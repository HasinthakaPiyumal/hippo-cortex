"""
MemoryTracker — GPU/CPU/SWR-buffer memory profiling for training runs.

Tracks memory footprint across a continual-learning run so the
constant-memory claim (StatsBuffer grows O(T x d_model), not with sample
count or backbone activations) can be measured empirically alongside GPU/CPU
usage. Any buffer registered via register_swr_buffer() must satisfy
SWRBufferProtocol, which hippocortex.cl.stats_buffer.StatsBuffer already
implements (memory_bytes() -> int).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import psutil
import torch


@runtime_checkable
class SWRBufferProtocol(Protocol):
    """Any SWR buffer registered with MemoryTracker must implement this."""
    def memory_bytes(self) -> int: ...


@dataclass(frozen=True)
class MemorySnapshot:
    """One point-in-time memory reading. All sizes in MB (MiB)."""
    label: str
    timestamp: float
    wall_time: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_peak_allocated_mb: float
    cpu_rss_mb: float
    cpu_vms_mb: float
    swr_buffer_mb: float

    @property
    def total_gpu_mb(self) -> float:
        return self.gpu_allocated_mb

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label":                 self.label,
            "timestamp_s":           round(self.timestamp, 4),
            "gpu_allocated_mb":      round(self.gpu_allocated_mb, 3),
            "gpu_reserved_mb":       round(self.gpu_reserved_mb, 3),
            "gpu_peak_allocated_mb": round(self.gpu_peak_allocated_mb, 3),
            "cpu_rss_mb":            round(self.cpu_rss_mb, 3),
            "cpu_vms_mb":            round(self.cpu_vms_mb, 3),
            "swr_buffer_mb":         round(self.swr_buffer_mb, 6),
        }

    def __str__(self) -> str:
        return (
            f"[{self.label}] "
            f"GPU={self.gpu_allocated_mb:.1f}MB "
            f"(peak {self.gpu_peak_allocated_mb:.1f}MB) | "
            f"RAM={self.cpu_rss_mb:.1f}MB | "
            f"SWR={self.swr_buffer_mb:.4f}MB "
            f"@ t={self.timestamp:.2f}s"
        )


@dataclass
class TaskMemoryRecord:
    """Memory snapshots bracketing one task's training."""
    task_id: int
    start_snapshot: Optional[MemorySnapshot] = None
    end_snapshot:   Optional[MemorySnapshot] = None

    @property
    def gpu_delta_mb(self) -> Optional[float]:
        if self.start_snapshot and self.end_snapshot:
            return self.end_snapshot.gpu_allocated_mb - self.start_snapshot.gpu_allocated_mb
        return None

    @property
    def cpu_delta_mb(self) -> Optional[float]:
        if self.start_snapshot and self.end_snapshot:
            return self.end_snapshot.cpu_rss_mb - self.start_snapshot.cpu_rss_mb
        return None

    @property
    def swr_delta_mb(self) -> Optional[float]:
        if self.start_snapshot and self.end_snapshot:
            return self.end_snapshot.swr_buffer_mb - self.start_snapshot.swr_buffer_mb
        return None


_BYTES_PER_MB = 1024 ** 2


class MemoryTracker:
    """
    Tracks GPU, CPU, and SWR-buffer memory throughout a training run.

    Usage:
        with MemoryTracker(device="cuda") as tracker:
            tracker.register_swr_buffer(stats_buffer)
            for t in range(num_tasks):
                tracker.task_start(t)
                train(t)
                tracker.task_end(t)
        tracker.print_summary()
    """

    def __init__(
        self,
        device: Union[str, "torch.device"] = "cuda",
        reset_peak_on_start: bool = True,
    ) -> None:
        self._device_str = str(device)
        self._is_cuda_device = self._device_str.startswith("cuda")
        self._reset_peak = reset_peak_on_start
        self._cuda_available = self._is_cuda_device and torch.cuda.is_available()
        self._cuda_warned = False
        self._process = psutil.Process()
        self._swr_buffer: Optional[SWRBufferProtocol] = None
        self._running = False
        self._start_monotonic: float = 0.0
        self._snapshots: List[MemorySnapshot] = []
        self._task_records: Dict[int, TaskMemoryRecord] = {}

    def register_swr_buffer(self, buffer: SWRBufferProtocol) -> None:
        if not isinstance(buffer, SWRBufferProtocol):
            raise TypeError("buffer must implement SWRBufferProtocol (memory_bytes() -> int)")
        self._swr_buffer = buffer

    def start(self) -> "MemoryTracker":
        if self._running:
            warnings.warn("MemoryTracker.start() called while already running.", stacklevel=2)
            return self
        self._snapshots.clear()
        self._task_records.clear()
        self._running = True
        self._start_monotonic = time.monotonic()
        if self._cuda_available and self._reset_peak:
            try:
                torch.cuda.reset_peak_memory_stats(self._device_str)
            except Exception:
                pass
        self.snapshot("start")
        return self

    def stop(self) -> "MemoryTracker":
        if not self._running:
            return self
        self.snapshot("stop")
        self._running = False
        return self

    def __enter__(self) -> "MemoryTracker":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def task_start(self, task_id: int) -> None:
        snap = self.snapshot(f"task_{task_id}_start")
        self._task_records[task_id] = TaskMemoryRecord(task_id=task_id, start_snapshot=snap)

    def task_end(self, task_id: int) -> None:
        snap = self.snapshot(f"task_{task_id}_end")
        if task_id in self._task_records:
            self._task_records[task_id].end_snapshot = snap
        else:
            warnings.warn(f"task_end({task_id}) called without matching task_start().", stacklevel=2)

    def snapshot(self, label: str = "manual") -> MemorySnapshot:
        now_mono = time.monotonic()
        timestamp = now_mono - self._start_monotonic if self._running else 0.0

        gpu_allocated_mb = gpu_reserved_mb = gpu_peak_mb = 0.0
        if self._is_cuda_device:
            if self._cuda_available:
                try:
                    dev = self._device_str
                    gpu_allocated_mb = torch.cuda.memory_allocated(dev) / _BYTES_PER_MB
                    gpu_reserved_mb  = torch.cuda.memory_reserved(dev)  / _BYTES_PER_MB
                    gpu_peak_mb      = torch.cuda.max_memory_allocated(dev) / _BYTES_PER_MB
                except Exception as exc:
                    if not self._cuda_warned:
                        warnings.warn(f"GPU memory query failed: {exc}", stacklevel=2)
                        self._cuda_warned = True
            elif not self._cuda_warned:
                warnings.warn(
                    f"device={self._device_str!r} requested but CUDA is not available. "
                    "GPU fields will be 0.0.",
                    UserWarning,
                    stacklevel=2,
                )
                self._cuda_warned = True

        mem_info = self._process.memory_info()
        cpu_rss_mb = mem_info.rss / _BYTES_PER_MB
        cpu_vms_mb = mem_info.vms / _BYTES_PER_MB

        swr_buffer_mb = 0.0
        if self._swr_buffer is not None:
            swr_buffer_mb = self._swr_buffer.memory_bytes() / _BYTES_PER_MB

        snap = MemorySnapshot(
            label=label, timestamp=timestamp, wall_time=time.time(),
            gpu_allocated_mb=gpu_allocated_mb, gpu_reserved_mb=gpu_reserved_mb,
            gpu_peak_allocated_mb=gpu_peak_mb,
            cpu_rss_mb=cpu_rss_mb, cpu_vms_mb=cpu_vms_mb,
            swr_buffer_mb=swr_buffer_mb,
        )
        self._snapshots.append(snap)
        return snap

    @property
    def snapshots(self) -> List[MemorySnapshot]:
        return list(self._snapshots)

    @property
    def task_records(self) -> Dict[int, TaskMemoryRecord]:
        return dict(self._task_records)

    @property
    def peak_gpu_mb(self) -> float:
        if not self._snapshots:
            return 0.0
        return max(s.gpu_allocated_mb for s in self._snapshots)

    @property
    def peak_cpu_rss_mb(self) -> float:
        if not self._snapshots:
            return 0.0
        return max(s.cpu_rss_mb for s in self._snapshots)

    def summary(self) -> Dict[str, Any]:
        if not self._snapshots:
            return {"error": "No snapshots recorded. Call start() first."}
        first = self._snapshots[0]
        last  = self._snapshots[-1]
        task_gpu_deltas = {
            tid: rec.gpu_delta_mb
            for tid, rec in self._task_records.items()
            if rec.gpu_delta_mb is not None
        }
        task_swr_deltas = {
            tid: rec.swr_delta_mb
            for tid, rec in self._task_records.items()
            if rec.swr_delta_mb is not None
        }
        threshold_mb = 5.0
        constant_footprint_ok = all(abs(d) < threshold_mb for d in task_gpu_deltas.values())
        return {
            "peak_gpu_mb":           round(self.peak_gpu_mb, 2),
            "peak_cpu_rss_mb":       round(self.peak_cpu_rss_mb, 2),
            "swr_buffer_growth_mb":  round(last.swr_buffer_mb - first.swr_buffer_mb, 6),
            "num_snapshots":         len(self._snapshots),
            "num_tasks":             len(self._task_records),
            "task_gpu_deltas_mb":    {k: round(v, 3) for k, v in task_gpu_deltas.items()},
            "task_swr_deltas_mb":    {k: round(v, 6) for k, v in task_swr_deltas.items()},
            "constant_footprint_ok": constant_footprint_ok,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n── MemoryTracker Summary ─────────────────────────────")
        print(f"  Snapshots taken      : {s.get('num_snapshots', 0)}")
        print(f"  Tasks tracked        : {s.get('num_tasks', 0)}")
        print(f"  Peak GPU allocated   : {s.get('peak_gpu_mb', 0):.1f} MB")
        print(f"  Peak CPU RSS         : {s.get('peak_cpu_rss_mb', 0):.1f} MB")
        print(f"  SWR buffer growth    : {s.get('swr_buffer_growth_mb', 0):.4f} MB")
        print(f"  Constant footprint   : {s.get('constant_footprint_ok', False)}")
        if s.get("task_gpu_deltas_mb"):
            print("  GPU delta per task   :", s["task_gpu_deltas_mb"])
        print("──────────────────────────────────────────────────────\n")
