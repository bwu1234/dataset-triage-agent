import pandas as pd

from triage.config import ValidationPolicy
from triage.executor import apply_op
from triage.ops import CastType, FilterRows, Impute, StandardizeMissing
from triage.validate import check_output

POLICY = ValidationPolicy(max_rows_removed_fraction=0.25)
BEFORE = pd.DataFrame({
    "q": [1, 2, 3, 4, -1, -2, 5, 6],
    "d": ["2025-01-01", "2025-01-02", None, "2025-01-04", "2025-01-05", "2025-01-06",
          "2025-01-07", "2025-01-08"],
    "t": ["a", "N/A", "b", "c", "d", "e", "f", "g"],
}).astype(object)


def _check(after, ops, skipped=None, rejected=(), nulled=None):
    return check_output(BEFORE, after, ops, skipped or {}, rejected, nulled or {}, POLICY)


def test_clean_run_passes():
    ops = [StandardizeMissing(column="t", tokens=["N/A"], reason="r"),
           FilterRows(column="q", operator=">=", value=0.0, reason="r")]
    after = apply_op(apply_op(BEFORE, ops[0]), ops[1])
    assert _check(after, ops, nulled={"t": 1}) == []


def test_too_many_rows_removed():
    op = FilterRows(column="q", operator=">=", value=3.0, reason="r")
    [failure] = _check(apply_op(BEFORE, op), [op])
    assert failure.startswith("4 of 8 rows removed (50%)")


def test_nulls_not_explained_by_an_op_on_that_column():
    after = BEFORE.copy()
    after.loc[0, "q"] = None
    [failure] = _check(after, [])
    assert "'q' has 1 more nulls" in failure


def test_cast_target_undone_by_a_later_op():
    ops = [CastType(column="d", to="datetime", reason="r"),
           Impute(column="d", strategy="constant", value="unknown", reason="r")]
    after = apply_op(apply_op(BEFORE, ops[0]), ops[1])
    [failure] = _check(after, ops)
    assert failure.startswith("column 'd' should be datetime after op #0 but is object")


def test_skipped_op_reported_once_and_rejected_cast_not_at_all():
    ops = [CastType(column="t", to="float", reason="r"), CastType(column="q", to="int", reason="r")]
    failures = _check(BEFORE, ops, skipped={1: "boom"}, rejected={0})
    assert failures == ["op #1 (cast_type) could not be applied: boom"]
