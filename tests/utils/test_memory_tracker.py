import time
import warnings

import pytest
import torch

from hippocortex.cl.stats_buffer import StatsBuffer
from hippocortex.utils.memory_tracker import (
    MemorySnapshot, MemoryTracker, TaskMemoryRecord, SWRBufferProtocol,
)


def _make_snapshot(**kwargs):
    defaults = dict(
        label="test", timestamp=0.0, wall_time=time.time(),
        gpu_allocated_mb=0.0, gpu_reserved_mb=0.0, gpu_peak_allocated_mb=0.0,
        cpu_rss_mb=100.0, cpu_vms_mb=200.0, swr_buffer_mb=0.0,
    )
    defaults.update(kwargs)
    return MemorySnapshot(**defaults)


class TestMemorySnapshot:
    def test_construction(self):
        snap = _make_snapshot(label="foo", gpu_allocated_mb=256.5)
        assert snap.label == "foo"
        assert snap.gpu_allocated_mb == pytest.approx(256.5)

    def test_total_gpu_mb_alias(self):
        snap = _make_snapshot(gpu_allocated_mb=512.0)
        assert snap.total_gpu_mb == pytest.approx(512.0)

    def test_as_dict_has_required_keys(self):
        d = _make_snapshot().as_dict()
        for key in ("label", "gpu_allocated_mb", "cpu_rss_mb", "swr_buffer_mb"):
            assert key in d

    def test_str_does_not_crash(self):
        snap = _make_snapshot()
        assert "GPU=" in str(snap) and "RAM=" in str(snap)

    def test_frozen_immutability(self):
        snap = _make_snapshot()
        with pytest.raises((AttributeError, TypeError)):
            snap.label = "modified"

    def test_as_dict_rounds_values(self):
        d = _make_snapshot(gpu_allocated_mb=123.456789).as_dict()
        assert d["gpu_allocated_mb"] == pytest.approx(123.457, abs=0.001)


class TestSWRBufferProtocol:
    def test_stats_buffer_satisfies_protocol(self):
        assert isinstance(StatsBuffer(), SWRBufferProtocol)

    def test_object_without_memory_bytes_does_not_satisfy_protocol(self):
        class NotABuffer:
            pass
        assert not isinstance(NotABuffer(), SWRBufferProtocol)

    def test_stats_buffer_byte_size_matches_formula(self):
        buf = StatsBuffer()
        H = torch.randn(10, 64)
        buf.update(0, H)
        assert buf.memory_bytes() == 64 * 2 * 4  # d_model * (mu, sigma2) * float32


class TestMemoryTrackerLifecycle:
    def test_start_creates_baseline_snapshot(self):
        tracker = MemoryTracker(device="cpu")
        tracker.start()
        assert len(tracker.snapshots) == 1
        assert tracker.snapshots[0].label == "start"
        tracker.stop()

    def test_stop_appends_stop_snapshot(self):
        tracker = MemoryTracker(device="cpu")
        tracker.start()
        tracker.stop()
        assert tracker.snapshots[-1].label == "stop"

    def test_context_manager(self):
        with MemoryTracker(device="cpu") as tracker:
            pass
        labels = [s.label for s in tracker.snapshots]
        assert labels[0] == "start" and labels[-1] == "stop"

    def test_double_start_warns(self):
        tracker = MemoryTracker(device="cpu")
        tracker.start()
        with pytest.warns(UserWarning):
            tracker.start()
        tracker.stop()

    def test_stop_without_start_is_no_op(self):
        tracker = MemoryTracker(device="cpu")
        tracker.stop()
        assert len(tracker.snapshots) == 0

    def test_start_clears_previous_state(self):
        tracker = MemoryTracker(device="cpu")
        tracker.start()
        tracker.snapshot("mid")
        tracker.stop()
        tracker.start()
        assert len(tracker.snapshots) == 1   # only the new "start"
        tracker.stop()


class TestMemoryTrackerDeviceGating:
    """device='cpu' must never touch the CUDA API or warn about it."""

    def test_cpu_device_never_warns_about_cuda(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # any warning becomes an error
            with MemoryTracker(device="cpu") as tracker:
                tracker.snapshot("a")
                tracker.snapshot("b")

    def test_cpu_device_gpu_fields_are_zero(self):
        with MemoryTracker(device="cpu") as tracker:
            snap = tracker.snapshot()
        assert snap.gpu_allocated_mb == pytest.approx(0.0)
        assert snap.gpu_reserved_mb == pytest.approx(0.0)
        assert snap.gpu_peak_allocated_mb == pytest.approx(0.0)

    def test_cuda_device_without_cuda_available_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with MemoryTracker(device="cuda") as tracker:
                tracker.snapshot("a")
                tracker.snapshot("b")
        cuda_warnings = [w for w in caught if "CUDA is not available" in str(w.message)]
        assert len(cuda_warnings) == 1   # warns once, not per-snapshot

    def test_is_cuda_device_detection(self):
        assert MemoryTracker(device="cuda")._is_cuda_device is True
        assert MemoryTracker(device="cuda:0")._is_cuda_device is True
        assert MemoryTracker(device="cpu")._is_cuda_device is False

    def test_accepts_torch_device_object(self):
        tracker = MemoryTracker(device=torch.device("cpu"))
        with tracker:
            snap = tracker.snapshot()
        assert snap.gpu_allocated_mb == pytest.approx(0.0)


class TestMemoryTrackerSnapshot:
    def test_manual_snapshot_appended(self):
        with MemoryTracker(device="cpu") as tracker:
            tracker.snapshot("after_forward")
        assert "after_forward" in [s.label for s in tracker.snapshots]

    def test_snapshot_returns_instance(self):
        with MemoryTracker(device="cpu") as tracker:
            snap = tracker.snapshot("check")
        assert isinstance(snap, MemorySnapshot) and snap.label == "check"

    def test_timestamps_non_decreasing(self):
        with MemoryTracker(device="cpu") as tracker:
            tracker.snapshot("a")
            time.sleep(0.01)
            tracker.snapshot("b")
        ts = [s.timestamp for s in tracker.snapshots]
        assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))

    def test_cpu_rss_positive(self):
        with MemoryTracker(device="cpu") as tracker:
            snap = tracker.snapshot()
        assert snap.cpu_rss_mb > 0.0


class TestMemoryTrackerTaskTracking:
    def test_task_records_populated(self):
        with MemoryTracker(device="cpu") as tracker:
            for t in range(3):
                tracker.task_start(t)
                tracker.task_end(t)
        assert len(tracker.task_records) == 3

    def test_task_record_has_both_snapshots(self):
        with MemoryTracker(device="cpu") as tracker:
            tracker.task_start(0)
            tracker.task_end(0)
        rec = tracker.task_records[0]
        assert rec.start_snapshot is not None and rec.end_snapshot is not None

    def test_task_start_label(self):
        with MemoryTracker(device="cpu") as tracker:
            tracker.task_start(42)
            tracker.task_end(42)
        labels = [s.label for s in tracker.snapshots]
        assert "task_42_start" in labels and "task_42_end" in labels

    def test_task_record_deltas_are_floats(self):
        with MemoryTracker(device="cpu") as tracker:
            tracker.task_start(0)
            tracker.task_end(0)
        rec = tracker.task_records[0]
        assert isinstance(rec.gpu_delta_mb, float)
        assert isinstance(rec.cpu_delta_mb, float)

    def test_task_record_deltas_none_without_end(self):
        tracker = MemoryTracker(device="cpu")
        tracker.start()
        tracker.task_start(0)
        assert tracker.task_records[0].gpu_delta_mb is None
        tracker.stop()

    def test_task_end_without_start_warns(self):
        with MemoryTracker(device="cpu") as tracker:
            with pytest.warns(UserWarning):
                tracker.task_end(99)

    def test_swr_delta_zero_without_buffer(self):
        with MemoryTracker(device="cpu") as tracker:
            tracker.task_start(0)
            tracker.task_end(0)
        assert tracker.task_records[0].swr_delta_mb == pytest.approx(0.0)


class TestMemoryTrackerSWRIntegration:
    """Exercises MemoryTracker against the real StatsBuffer used in consolidation.py."""

    def test_register_valid_buffer(self):
        tracker = MemoryTracker(device="cpu")
        tracker.register_swr_buffer(StatsBuffer())

    def test_register_invalid_object_raises(self):
        with pytest.raises(TypeError):
            MemoryTracker(device="cpu").register_swr_buffer("not_a_buffer")

    def test_swr_size_reflected_in_snapshot(self):
        d_model = 128
        buf = StatsBuffer()
        buf.update(0, torch.randn(10, d_model))
        tracker = MemoryTracker(device="cpu")
        tracker.register_swr_buffer(buf)
        tracker.start()
        snap = tracker.snapshot("swr_check")
        tracker.stop()
        assert snap.swr_buffer_mb == pytest.approx(buf.memory_bytes() / (1024**2), rel=1e-5)

    def test_swr_growth_tracked_across_tasks(self):
        d_model = 128
        buf = StatsBuffer()
        tracker = MemoryTracker(device="cpu")
        tracker.register_swr_buffer(buf)
        tracker.start()
        for t in range(4):
            tracker.task_start(t)
            buf.update(t, torch.randn(10, d_model))
            tracker.task_end(t)
        tracker.stop()
        per_task_mb = d_model * 2 * 4 / (1024**2)   # (mu, sigma2), float32
        assert tracker.summary()["swr_buffer_growth_mb"] == pytest.approx(4 * per_task_mb, rel=1e-4)

    def test_swr_growth_independent_of_sample_count(self):
        """Matches StatsBuffer's constant-memory invariant: more samples per
        task must not change the buffer's tracked size."""
        d_model = 64
        buf = StatsBuffer()
        tracker = MemoryTracker(device="cpu")
        tracker.register_swr_buffer(buf)
        tracker.start()
        tracker.task_start(0)
        buf.update(0, torch.randn(5, d_model))
        snap_small = tracker.snapshot("small_batch")
        buf.update(0, torch.randn(500, d_model))   # same task, many more samples
        snap_large = tracker.snapshot("large_batch")
        tracker.task_end(0)
        tracker.stop()
        assert snap_small.swr_buffer_mb == pytest.approx(snap_large.swr_buffer_mb)


class TestMemoryTrackerSummary:
    def test_summary_has_required_keys(self):
        with MemoryTracker(device="cpu") as tracker:
            tracker.task_start(0)
            tracker.task_end(0)
        s = tracker.summary()
        for key in ("peak_gpu_mb", "peak_cpu_rss_mb", "swr_buffer_growth_mb",
                    "num_snapshots", "num_tasks", "task_gpu_deltas_mb",
                    "task_swr_deltas_mb", "constant_footprint_ok"):
            assert key in s

    def test_num_tasks_in_summary(self):
        with MemoryTracker(device="cpu") as tracker:
            for t in range(3):
                tracker.task_start(t)
                tracker.task_end(t)
        assert tracker.summary()["num_tasks"] == 3

    def test_peak_cpu_rss_positive(self):
        with MemoryTracker(device="cpu") as tracker:
            pass
        assert tracker.summary()["peak_cpu_rss_mb"] > 0.0

    def test_constant_footprint_true_when_no_gpu_growth(self):
        with MemoryTracker(device="cpu") as tracker:
            for t in range(5):
                tracker.task_start(t)
                tracker.task_end(t)
        assert tracker.summary()["constant_footprint_ok"] is True

    def test_empty_summary_before_start(self):
        assert "error" in MemoryTracker(device="cpu").summary()

    def test_print_summary_does_not_crash(self, capsys):
        with MemoryTracker(device="cpu") as tracker:
            tracker.task_start(0)
            tracker.task_end(0)
        tracker.print_summary()
        assert "MemoryTracker" in capsys.readouterr().out


class TestMemoryTrackerPeakProperties:
    def test_peak_cpu_rss_is_max_across_snapshots(self):
        with MemoryTracker(device="cpu") as tracker:
            tracker.snapshot("a")
            tracker.snapshot("b")
        assert tracker.peak_cpu_rss_mb == max(s.cpu_rss_mb for s in tracker.snapshots)

    def test_peak_gpu_mb_zero_on_cpu(self):
        with MemoryTracker(device="cpu") as tracker:
            pass
        assert tracker.peak_gpu_mb == pytest.approx(0.0)

    def test_peak_properties_zero_before_start(self):
        tracker = MemoryTracker(device="cpu")
        assert tracker.peak_gpu_mb == pytest.approx(0.0)
        assert tracker.peak_cpu_rss_mb == pytest.approx(0.0)


class TestTaskMemoryRecord:
    def test_deltas_none_when_no_snapshots(self):
        rec = TaskMemoryRecord(task_id=0)
        assert rec.gpu_delta_mb is None
        assert rec.cpu_delta_mb is None
        assert rec.swr_delta_mb is None

    def test_gpu_delta_computed_correctly(self):
        start = _make_snapshot(label="start", gpu_allocated_mb=100.0)
        end = _make_snapshot(label="end", gpu_allocated_mb=150.0)
        rec = TaskMemoryRecord(task_id=0, start_snapshot=start, end_snapshot=end)
        assert rec.gpu_delta_mb == pytest.approx(50.0)
