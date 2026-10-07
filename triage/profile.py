"""Deterministic dataset profile: the planner's only view of the data.

Sample values are copied from the file, so they are untrusted text. The
planner prompt must present them as data, never as instructions.
"""

import pandas as pd
from pydantic import BaseModel

from triage.config import ProfilerConfig


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    distinct_count: int
    sample_values: list[str]
    # Hints for text columns: counts over non-null string values.
    missing_token_count: int = 0
    # The markers counted above, one spelling per configured token. Samples
    # alone hid them: region's first five distinct values held one of its
    # three markers, so no plan could list the others (M7 eval).
    missing_tokens_found: list[str] = []
    whitespace_padded_count: int = 0
    numeric_parse_rate: float | None = None
    # Present for numeric columns.
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    # Values below zero, for numeric columns and for text that parses as
    # numbers. min alone did not show how many values were negative or that
    # they were a minority: 27b plans left quantity's negative values alone in
    # 17 of 20 runs (M7 eval).
    negative_count: int | None = None


class DatasetProfile(BaseModel):
    n_rows: int
    n_columns: int
    duplicate_rows: int
    columns: list[ColumnProfile]


def profile(df: pd.DataFrame, config: ProfilerConfig) -> DatasetProfile:
    return DatasetProfile(
        n_rows=len(df),
        n_columns=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        columns=[_profile_column(str(c), df[c], config) for c in df.columns],
    )


def _profile_column(name: str, s: pd.Series, config: ProfilerConfig) -> ColumnProfile:
    non_null = s.dropna()
    samples = [
        _truncate(str(v), config.max_sample_chars)
        for v in non_null.drop_duplicates().head(config.sample_values)
    ]
    col = ColumnProfile(
        name=name,
        dtype=str(s.dtype),
        null_count=int(s.isna().sum()),
        distinct_count=int(non_null.nunique()),
        sample_values=samples,
    )
    if pd.api.types.is_bool_dtype(s):
        return col
    if pd.api.types.is_numeric_dtype(s):
        if not non_null.empty:
            col.min, col.max, col.mean = (float(non_null.min()), float(non_null.max()),
                                          float(non_null.mean()))
            col.negative_count = int((non_null < 0).sum())
        return col
    strings = non_null[non_null.map(lambda v: isinstance(v, str))]
    if not strings.empty:
        tokens = {t.strip().lower() for t in config.missing_tokens}
        stripped = strings.str.strip()
        lowered = stripped.str.lower()
        is_token = lowered.isin(tokens)
        col.missing_token_count = int(is_token.sum())
        found = stripped[is_token].groupby(lowered[is_token], sort=True).first()
        col.missing_tokens_found = [_truncate(v, config.max_sample_chars) for v in found]
        col.whitespace_padded_count = int((stripped != strings).sum())
        parsed = pd.to_numeric(stripped, errors="coerce")
        col.numeric_parse_rate = round(float(parsed.notna().mean()), 4)
        if parsed.notna().any():
            col.negative_count = int((parsed < 0).sum())
    return col


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
