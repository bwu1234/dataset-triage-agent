"""Checks on a finished run's output.

Each failure is one sentence the planner can act on; ``triage.graph`` feeds
them back when it replans. The checks compare the run's input frame with its
output, using what the audit log says each op did.
"""

from collections.abc import Callable, Collection, Mapping, Sequence

import pandas as pd
from pandas.api import types

from triage.config import ValidationPolicy
from triage.ops import AnyOp, CastType

_DTYPE_CHECKS: dict[str, Callable[[pd.Series], bool]] = {
    "int": types.is_integer_dtype,
    "float": types.is_float_dtype,
    "datetime": types.is_datetime64_any_dtype,
    "bool": types.is_bool_dtype,
    "string": types.is_string_dtype,
}


def check_output(
    before: pd.DataFrame,
    after: pd.DataFrame,
    ops: Sequence[AnyOp],
    skipped: Mapping[int, str],
    rejected: Collection[int],
    nulled: Mapping[str, int],
    policy: ValidationPolicy,
) -> list[str]:
    """Return the failures, empty if the output passes.

    ``skipped`` maps op index to the ``OpError`` that stopped it, ``rejected``
    holds indexes a person rejected, and ``nulled`` counts values each column
    lost to applied ops (from their measured impact).
    """
    failures = [f"op #{i} ({ops[i].op}) could not be applied: {err}"
                for i, err in sorted(skipped.items())]

    removed = len(before) - len(after)
    if before.shape[0] and removed / len(before) > policy.max_rows_removed_fraction:
        failures.append(
            f"{removed} of {len(before)} rows removed ({removed / len(before):.0%}); at most "
            f"{policy.max_rows_removed_fraction:.0%} may be removed in total"
        )

    # A cross-check on the audit log: route already gated each op's nulls, so
    # this fires only if an op nulled values outside its own column.
    for column in (c for c in after.columns if c in before.columns):
        added = int(after[column].isna().sum() - before.loc[after.index, column].isna().sum())
        if added > nulled.get(column, 0):
            failures.append(f"column {column!r} has {added} more nulls than the input, but "
                            f"applied ops on it nulled {nulled.get(column, 0)}")

    # A person's rejection is a decision, not a planner mistake, and a skipped
    # cast is already reported above.
    for i, op in enumerate(ops):
        if (isinstance(op, CastType) and i not in skipped and i not in rejected
                and op.column in after.columns
                and not _DTYPE_CHECKS[op.to](after[op.column])):
            failures.append(f"column {op.column!r} should be {op.to} after op #{i} but is "
                            f"{after[op.column].dtype}; a later op changed it")
    return failures
