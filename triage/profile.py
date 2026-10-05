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
    whitespace_padded_count: int = 0
    numeric_parse_rate: float | None = None
    # Present for numeric columns.
    min: float | None = None
    max: float | None = None
    mean: float | None = None


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
        return col
    strings = non_null[non_null.map(lambda v: isinstance(v, str))]
    if not strings.empty:
        tokens = {t.strip().lower() for t in config.missing_tokens}
        stripped = strings.str.strip()
        col.missing_token_count = int(stripped.str.lower().isin(tokens).sum())
        col.whitespace_padded_count = int((stripped != strings).sum())
        parsed = pd.to_numeric(stripped, errors="coerce")
        col.numeric_parse_rate = round(float(parsed.notna().mean()), 4)
    return col


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
