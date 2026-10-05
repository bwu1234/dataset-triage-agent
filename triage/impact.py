"""Measure what an op does to a frame, and decide whether it needs approval."""

import pandas as pd
from pydantic import BaseModel

from triage.config import RiskPolicy
from triage.executor import apply_op
from triage.ops import AnyOp


class Impact(BaseModel):
    rows_removed: int
    columns_removed: int
    values_nulled: int  # non-null -> null
    values_filled: int  # null -> non-null
    values_modified: int  # non-null -> different non-null


def measure(before: pd.DataFrame, after: pd.DataFrame) -> Impact:
    """Compare two frames whose rows share an index (the executor preserves it).

    Values are compared by their string form, so a cast from "7" to 7 is not
    counted as a modification but "7" to 7.5 is.
    """
    kept = after.index
    common = [c for c in before.columns if c in after.columns]
    nulled = filled = modified = 0
    for column in common:
        b = before.loc[kept, column]
        a = after[column]
        b_null, a_null = b.isna().to_numpy(), a.isna().to_numpy()
        nulled += int((~b_null & a_null).sum())
        filled += int((b_null & ~a_null).sum())
        both = ~b_null & ~a_null
        modified += int((b.astype(str).to_numpy()[both] != a.astype(str).to_numpy()[both]).sum())
    return Impact(
        rows_removed=len(before) - len(after),
        columns_removed=len(before.columns) - len(common),
        values_nulled=nulled,
        values_filled=filled,
        values_modified=modified,
    )


def assess(df: pd.DataFrame, op: AnyOp) -> Impact:
    """Dry-run an op on a copy and report its impact. Raises ``OpError`` if it cannot apply."""
    return measure(df, apply_op(df, op))


def needs_approval(impact: Impact, policy: RiskPolicy) -> bool:
    return (
        impact.rows_removed > policy.max_rows_removed
        or impact.columns_removed > policy.max_columns_removed
        or impact.values_nulled > policy.max_values_nulled
    )
