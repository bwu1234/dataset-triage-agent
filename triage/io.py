"""CSV loading. Only empty cells become null on load, so markers such as 'N/A'
stay visible as strings for the profiler and planner to deal with."""

from pathlib import Path

import pandas as pd


def load_csv(path: Path, na_values: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=na_values)


# Intermediate frames use pickle: it round-trips every dtype the executor can
# produce, including mixed object columns that Parquet rejects. The files are
# written by this process under ``Settings.runs_dir`` and trusted like the
# checkpoint database; never point ``load_frame`` at a file from elsewhere.
def save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_pickle(tmp)
    tmp.replace(path)  # Atomic, so a crash never leaves a half-written frame.


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)
