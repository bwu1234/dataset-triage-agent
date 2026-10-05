"""CSV loading. Only empty cells become null on load, so markers such as 'N/A'
stay visible as strings for the profiler and planner to deal with."""

from pathlib import Path

import pandas as pd


def load_csv(path: Path, na_values: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=na_values)
