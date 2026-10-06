"""Deterministic executor for cleaning ops.

``apply_op`` is pure: it never mutates its input and keeps the original row
index, so ``triage.impact`` can line rows up before and after.
"""

from functools import singledispatch

import pandas as pd

from triage.ops import (
    AnyOp,
    CastType,
    Dedupe,
    DropColumn,
    FilterRows,
    Impute,
    StandardizeMissing,
    StripWhitespace,
)

_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


class OpError(ValueError):
    """The op cannot be applied to this frame. The message is fed back to the planner."""


def apply_op(df: pd.DataFrame, op: AnyOp) -> pd.DataFrame:
    for column in _columns_of(op):
        if column not in df.columns:
            raise OpError(f"{op.op}: column {column!r} not found; columns are {list(df.columns)}")
    return _apply(op, df.copy())


def _columns_of(op: AnyOp) -> list[str]:
    if isinstance(op, Dedupe):
        return list(op.subset or [])
    return [op.column]


# singledispatch picks the implementation from the op's class, so adding an op
# means adding one registered function rather than growing an if/elif chain.
@singledispatch
def _apply(op, df: pd.DataFrame) -> pd.DataFrame:
    raise OpError(f"no executor for op {type(op).__name__}")


@_apply.register
def _(op: DropColumn, df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[op.column])


@_apply.register
def _(op: CastType, df: pd.DataFrame) -> pd.DataFrame:
    df[op.column] = _cast(df[op.column], op)
    return df


def _cast(s: pd.Series, op: CastType) -> pd.Series:
    if op.to == "string":
        return s.astype("string")
    if op.to == "float":
        return pd.to_numeric(s, errors="coerce").astype("Float64")
    if op.to == "int":
        numeric = pd.to_numeric(s, errors="coerce")
        # Non-integral values become null rather than being silently truncated.
        numeric = numeric.where(numeric.isna() | (numeric % 1 == 0))
        return numeric.astype("Int64")
    if op.to == "datetime":
        return pd.to_datetime(s, errors="coerce", format=op.datetime_format or "mixed")
    if op.to == "bool":
        lowered = s.astype("string").str.strip().str.lower()
        out = pd.Series(pd.NA, index=s.index, dtype="boolean")
        out[lowered.isin(_TRUE)] = True
        out[lowered.isin(_FALSE)] = False
        return out
    raise OpError(f"cast_type: unsupported target {op.to!r}")


@_apply.register
def _(op: Impute, df: pd.DataFrame) -> pd.DataFrame:
    s = df[op.column]
    if op.strategy in ("mean", "median"):
        if not pd.api.types.is_numeric_dtype(s):
            raise OpError(f"impute: {op.strategy} needs a numeric column; {op.column!r} is {s.dtype}")
        fill = s.mean() if op.strategy == "mean" else s.median()
        if pd.api.types.is_integer_dtype(s) and fill % 1 != 0:
            s = s.astype("Float64")
    elif op.strategy == "mode":
        modes = s.mode(dropna=True)
        if modes.empty:
            raise OpError(f"impute: {op.column!r} has no non-null values to take a mode from")
        fill = modes.iloc[0]
    else:
        fill = op.value
    if pd.isna(fill):
        raise OpError(f"impute: {op.column!r} has no non-null values")
    try:
        df[op.column] = s.fillna(fill)
    except TypeError as e:  # e.g. a text constant into a Float64 column.
        raise OpError(f"impute: cannot fill {op.column!r} ({s.dtype}) with {fill!r}") from e
    return df


@_apply.register
def _(op: Dedupe, df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=op.subset, keep=op.keep)


@_apply.register
def _(op: FilterRows, df: pd.DataFrame) -> pd.DataFrame:
    s = df[op.column]
    if op.operator == "not_null":
        return df[s.notna()]
    value = op.value
    if isinstance(value, float):
        s = pd.to_numeric(s, errors="coerce")
    compare = {
        "==": s.eq, "!=": s.ne, "<": s.lt, "<=": s.le, ">": s.gt, ">=": s.ge,
    }[op.operator]
    try:
        keep = compare(value)
    except TypeError as e:
        raise OpError(f"filter_rows: cannot compare {op.column!r} ({s.dtype}) with {value!r}") from e
    return df[keep.fillna(False).astype(bool) & s.notna()]


@_apply.register
def _(op: StandardizeMissing, df: pd.DataFrame) -> pd.DataFrame:
    s = df[op.column]
    tokens = {t.strip().lower() for t in op.tokens}
    is_marker = s.map(lambda v: isinstance(v, str) and v.strip().lower() in tokens)
    df[op.column] = s.mask(is_marker.astype(bool))
    return df


@_apply.register
def _(op: StripWhitespace, df: pd.DataFrame) -> pd.DataFrame:
    df[op.column] = df[op.column].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df
