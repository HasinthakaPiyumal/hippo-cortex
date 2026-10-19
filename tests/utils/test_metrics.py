import math
import pytest
from hippocortex.utils.metrics import (
    ResultMatrix, ContinualMetrics,
    compute_aa, compute_aia, compute_fm, compute_all_metrics,
)

# ── Hand-worked 3-task example ───────────────────────────────────────────────
#
#   R[i][j]     j=0    j=1    j=2
#   i=0         0.90    —      —
#   i=1         0.80   0.70    —
#   i=2         0.60   0.65   0.85
#
#   AA_0 = 0.90
#   AA_1 = (0.80+0.70)/2 = 0.750
#   AA_2 = (0.60+0.65+0.85)/3 = 0.700
#
#   AIA_0 = 0.900
#   AIA_1 = (0.900+0.750)/2 = 0.825
#   AIA_2 = (0.900+0.750+0.700)/3 ≈ 0.783
#
#   FM_0 = 0  (no old tasks)
#   FM_1: j=0 → peak=0.90, final=0.80, drop=0.10 → FM=0.10
#   FM_2: j=0 → peak=0.90, final=0.60, drop=0.30
#          j=1 → peak=0.70, final=0.65, drop=0.05
#          FM = (0.30+0.05)/2 = 0.175
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def three_task_matrix():
    R = ResultMatrix(num_tasks=3)
    R.update(0, 0, 0.90)
    R.update(1, 0, 0.80);  R.update(1, 1, 0.70)
    R.update(2, 0, 0.60);  R.update(2, 1, 0.65);  R.update(2, 2, 0.85)
    return R

@pytest.fixture
def perfect_matrix():
    R = ResultMatrix(num_tasks=4)
    for i in range(4):
        for j in range(i + 1):
            R.update(i, j, 1.0)
    return R

@pytest.fixture
def catastrophic_matrix():
    R = ResultMatrix(num_tasks=3)
    R.update(0, 0, 1.00)
    R.update(1, 0, 0.00);  R.update(1, 1, 1.00)
    R.update(2, 0, 0.00);  R.update(2, 1, 0.00);  R.update(2, 2, 1.00)
    return R


class TestResultMatrix:
    def test_basic_update_and_get(self):
        R = ResultMatrix(num_tasks=2)
        R.update(0, 0, 0.75)
        assert R.get(0, 0) == pytest.approx(0.75)

    def test_raises_on_invalid_num_tasks(self):
        with pytest.raises(ValueError):
            ResultMatrix(num_tasks=0)

    def test_raises_on_out_of_range_task_i(self):
        R = ResultMatrix(num_tasks=3)
        with pytest.raises(IndexError):
            R.update(3, 0, 0.5)

    def test_raises_on_j_greater_than_i(self):
        R = ResultMatrix(num_tasks=3)
        with pytest.raises(IndexError):
            R.update(0, 1, 0.5)

    def test_warns_on_accuracy_above_one(self):
        R = ResultMatrix(num_tasks=2)
        with pytest.warns(UserWarning, match="fraction"):
            R.update(0, 0, 95.0)

    def test_get_raises_on_unset_entry(self):
        R = ResultMatrix(num_tasks=3)
        with pytest.raises(ValueError, match="not been set"):
            R.get(0, 0)

    def test_to_tensor_shape(self):
        R = ResultMatrix(num_tasks=3)
        R.update(0, 0, 0.8)
        assert R.to_tensor().shape == (3, 3)

    def test_overwrite_entry(self):
        R = ResultMatrix(num_tasks=2)
        R.update(0, 0, 0.5)
        R.update(0, 0, 0.9)
        assert R.get(0, 0) == pytest.approx(0.9)

    def test_repr_does_not_crash(self, three_task_matrix):
        assert "ResultMatrix" in repr(three_task_matrix)


class TestComputeAA:
    def test_aa_after_task_0(self, three_task_matrix):
        assert compute_aa(three_task_matrix, 0) == pytest.approx(0.9000, abs=1e-6)

    def test_aa_after_task_1(self, three_task_matrix):
        assert compute_aa(three_task_matrix, 1) == pytest.approx(0.7500, abs=1e-6)

    def test_aa_after_task_2(self, three_task_matrix):
        assert compute_aa(three_task_matrix, 2) == pytest.approx((0.60+0.65+0.85)/3, abs=1e-6)

    def test_aa_perfect_model(self, perfect_matrix):
        for k in range(4):
            assert compute_aa(perfect_matrix, k) == pytest.approx(1.0, abs=1e-6)

    def test_aa_catastrophic_model(self, catastrophic_matrix):
        assert compute_aa(catastrophic_matrix, 2) == pytest.approx((0.0+0.0+1.0)/3, abs=1e-6)

    def test_aa_single_task(self):
        R = ResultMatrix(num_tasks=1)
        R.update(0, 0, 0.77)
        assert compute_aa(R, 0) == pytest.approx(0.77, abs=1e-6)

    def test_aa_raises_on_out_of_range_k(self, three_task_matrix):
        with pytest.raises(IndexError):
            compute_aa(three_task_matrix, 5)


class TestComputeAIA:
    def test_aia_at_task_0_equals_aa(self, three_task_matrix):
        assert compute_aia(three_task_matrix, 0) == pytest.approx(compute_aa(three_task_matrix, 0))

    def test_aia_at_task_1(self, three_task_matrix):
        assert compute_aia(three_task_matrix, 1) == pytest.approx((0.90+0.75)/2, abs=1e-6)

    def test_aia_at_task_2(self, three_task_matrix):
        expected = (0.90 + 0.75 + (0.60+0.65+0.85)/3) / 3
        assert compute_aia(three_task_matrix, 2) == pytest.approx(expected, abs=1e-6)

    def test_aia_perfect_model(self, perfect_matrix):
        for k in range(4):
            assert compute_aia(perfect_matrix, k) == pytest.approx(1.0, abs=1e-6)

    def test_aia_declines_for_catastrophic_model(self, catastrophic_matrix):
        vals = [compute_aia(catastrophic_matrix, k) for k in range(3)]
        assert vals[0] > vals[1] > vals[2]

    def test_aia_raises_on_out_of_range_k(self, three_task_matrix):
        with pytest.raises(IndexError):
            compute_aia(three_task_matrix, 10)


class TestComputeFM:
    def test_fm_at_task_0_is_zero(self, three_task_matrix):
        assert compute_fm(three_task_matrix, 0) == pytest.approx(0.0, abs=1e-9)

    def test_fm_at_task_1(self, three_task_matrix):
        assert compute_fm(three_task_matrix, 1) == pytest.approx(0.10, abs=1e-6)

    def test_fm_at_task_2(self, three_task_matrix):
        assert compute_fm(three_task_matrix, 2) == pytest.approx((0.30+0.05)/2, abs=1e-6)

    def test_fm_perfect_model_is_zero(self, perfect_matrix):
        for k in range(4):
            assert compute_fm(perfect_matrix, k) == pytest.approx(0.0, abs=1e-6)

    def test_fm_catastrophic_model(self, catastrophic_matrix):
        assert compute_fm(catastrophic_matrix, 2) == pytest.approx(1.0, abs=1e-6)

    def test_fm_can_be_negative(self):
        R = ResultMatrix(num_tasks=2)
        R.update(0, 0, 0.50)
        R.update(1, 0, 0.90)   # improved on task 0
        R.update(1, 1, 0.80)
        assert compute_fm(R, 1) == pytest.approx(-0.40, abs=1e-6)

    def test_fm_raises_on_out_of_range_k(self, three_task_matrix):
        with pytest.raises(IndexError):
            compute_fm(three_task_matrix, 99)


class TestComputeAllMetrics:
    def test_return_structure(self, three_task_matrix):
        result = compute_all_metrics(three_task_matrix, num_tasks=3)
        assert set(result.keys()) == {"aa", "aia", "fm"}
        assert len(result["aa"]) == len(result["aia"]) == len(result["fm"]) == 3

    def test_values_match_individual_functions(self, three_task_matrix):
        result = compute_all_metrics(three_task_matrix, num_tasks=3)
        for k in range(3):
            assert result["aa"][k]  == pytest.approx(compute_aa(three_task_matrix, k))
            assert result["aia"][k] == pytest.approx(compute_aia(three_task_matrix, k))
            assert result["fm"][k]  == pytest.approx(compute_fm(three_task_matrix, k))

    def test_partial_num_tasks(self, three_task_matrix):
        result = compute_all_metrics(three_task_matrix, num_tasks=2)
        assert len(result["aa"]) == 2

    def test_raises_if_num_tasks_exceeds_matrix(self, three_task_matrix):
        with pytest.raises(ValueError):
            compute_all_metrics(three_task_matrix, num_tasks=10)

    def test_all_metrics_perfect(self, perfect_matrix):
        result = compute_all_metrics(perfect_matrix, num_tasks=4)
        for k in range(4):
            assert result["aa"][k]  == pytest.approx(1.0)
            assert result["aia"][k] == pytest.approx(1.0)
            assert result["fm"][k]  == pytest.approx(0.0)


class TestContinualMetrics:
    def _fill(self, cm):
        cm.record(0, 0, 0.90)
        cm.record(1, 0, 0.80);  cm.record(1, 1, 0.70)
        cm.record(2, 0, 0.60);  cm.record(2, 1, 0.65);  cm.record(2, 2, 0.85)

    def test_summary_has_required_keys(self):
        cm = ContinualMetrics(num_tasks=3)
        self._fill(cm)
        s = cm.summary()
        for key in ("final_aa", "final_aia", "final_fm", "num_tasks_completed"):
            assert key in s

    def test_final_aa(self):
        cm = ContinualMetrics(num_tasks=3)
        self._fill(cm)
        assert cm.aa() == pytest.approx((0.60+0.65+0.85)/3, abs=1e-6)

    def test_final_aia(self):
        cm = ContinualMetrics(num_tasks=3)
        self._fill(cm)
        expected = (0.90 + 0.75 + (0.60+0.65+0.85)/3) / 3
        assert cm.aia() == pytest.approx(expected, abs=1e-6)

    def test_final_fm(self):
        cm = ContinualMetrics(num_tasks=3)
        self._fill(cm)
        assert cm.fm() == pytest.approx((0.30+0.05)/2, abs=1e-6)

    def test_explicit_k_parameter(self):
        cm = ContinualMetrics(num_tasks=3)
        self._fill(cm)
        assert cm.aa(k=0) == pytest.approx(0.90, abs=1e-6)
        assert cm.aa(k=1) == pytest.approx(0.75, abs=1e-6)

    def test_summary_raises_before_any_records(self):
        cm = ContinualMetrics(num_tasks=3)
        with pytest.raises(RuntimeError, match="No data"):
            cm.summary()

    def test_result_matrix_property(self):
        cm = ContinualMetrics(num_tasks=3)
        self._fill(cm)
        assert isinstance(cm.result_matrix, ResultMatrix)

    def test_num_tasks_completed_updates(self):
        cm = ContinualMetrics(num_tasks=5)
        cm.record(0, 0, 0.8)
        assert cm.summary()["num_tasks_completed"] == 1
        cm.record(1, 0, 0.7);  cm.record(1, 1, 0.9)
        assert cm.summary()["num_tasks_completed"] == 2

    def test_fm_first_element_zero(self):
        cm = ContinualMetrics(num_tasks=3)
        self._fill(cm)
        assert cm.summary()["fm_per_task"][0] == pytest.approx(0.0)

    def test_per_task_lists_correct_length(self):
        cm = ContinualMetrics(num_tasks=3)
        self._fill(cm)
        s = cm.summary()
        assert len(s["aa_per_task"]) == len(s["aia_per_task"]) == len(s["fm_per_task"]) == 3


class TestMathematicalProperties:
    def test_aa_bounded_zero_to_one(self, three_task_matrix):
        for k in range(3):
            assert 0.0 <= compute_aa(three_task_matrix, k) <= 1.0

    def test_fm_single_task_zero(self):
        R = ResultMatrix(num_tasks=1)
        R.update(0, 0, 0.8)
        assert compute_fm(R, 0) == pytest.approx(0.0)

    def test_large_matrix_no_nan(self):
        n = 20
        R = ResultMatrix(num_tasks=n)
        for i in range(n):
            for j in range(i + 1):
                R.update(i, j, max(0.0, 0.9 - 0.02 * (i - j)))
        result = compute_all_metrics(R, num_tasks=n)
        assert all(not math.isnan(v) for v in result["aa"])
        assert all(not math.isnan(v) for v in result["fm"])
