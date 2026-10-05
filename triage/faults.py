"""Synthetic dirty CSVs with known injected faults, plus outcome checks.

Each fixture is a seeded clean "orders" table with faults injected into it, and
a manifest recording what was injected. The manifest is the answer key: the
evaluation reads it, the planner must never see it.

Checks judge the cleaned output, not which ops were chosen, so any plan that
actually fixes a fault gets credit for it.

    python -m triage.faults --out fixtures --seeds 0 1 2
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

FAULT_KINDS = (
    "missing_tokens",
    "whitespace_padding",
    "numeric_as_text",
    "impossible_values",
    "missing_values",
    "mixed_date_formats",
    "constant_column",
    "duplicate_rows",  # Injected last so copies match the rest of the faults.
)


class Fault(BaseModel):
    kind: str
    column: str | None
    n_injected: int


class Manifest(BaseModel):
    seed: int
    n_clean_rows: int
    faults: list[Fault]


def clean_orders(rng: np.random.Generator, n_rows: int) -> pd.DataFrame:
    days = rng.integers(0, 365, n_rows)
    return pd.DataFrame({
        "order_id": np.arange(1, n_rows + 1),
        "customer": rng.choice(["Ana Ruiz", "Ben Okafor", "Chen Wei", "Dana Kim", "Eli Novak"], n_rows),
        "region": rng.choice(["north", "south", "east", "west"], n_rows),
        "amount": np.round(rng.uniform(5, 500, n_rows), 2),
        "quantity": rng.integers(1, 11, n_rows),
        "rating": np.round(rng.uniform(1, 5, n_rows), 1),
        "order_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(days, unit="D"),
        "is_member": rng.choice([True, False], n_rows),
    })


def make_dirty(seed: int, n_rows: int = 200, rate: float = 0.08) -> tuple[pd.DataFrame, Manifest]:
    rng = np.random.default_rng(seed)
    df = clean_orders(rng, n_rows).astype(object)
    k = max(1, int(n_rows * rate))
    faults: list[Fault] = []

    def rows() -> np.ndarray:
        return rng.choice(n_rows, size=k, replace=False)

    df.loc[rows(), "region"] = rng.choice(["N/A", "?", "unknown"], k)
    faults.append(Fault(kind="missing_tokens", column="region", n_injected=k))

    idx = rows()
    df.loc[idx, "customer"] = [f"  {v} " for v in df.loc[idx, "customer"]]
    faults.append(Fault(kind="whitespace_padding", column="customer", n_injected=k))

    df.loc[rows(), "amount"] = rng.choice(["unknown", "n/a"], k)
    faults.append(Fault(kind="numeric_as_text", column="amount", n_injected=k))

    df.loc[rows(), "quantity"] = -rng.integers(1, 5, k)
    faults.append(Fault(kind="impossible_values", column="quantity", n_injected=k))

    df.loc[rows(), "rating"] = np.nan
    faults.append(Fault(kind="missing_values", column="rating", n_injected=k))

    dates = pd.to_datetime(df["order_date"])
    idx = rows()
    df["order_date"] = dates.dt.strftime("%Y-%m-%d")
    df.loc[idx, "order_date"] = dates[idx].dt.strftime("%d %b %Y")
    faults.append(Fault(kind="mixed_date_formats", column="order_date", n_injected=k))

    df["channel"] = "web"
    faults.append(Fault(kind="constant_column", column="channel", n_injected=n_rows))

    df = pd.concat([df, df.iloc[rows()]], ignore_index=True)
    faults.append(Fault(kind="duplicate_rows", column=None, n_injected=k))

    return df, Manifest(seed=seed, n_clean_rows=n_rows, faults=faults)


def write_fixture(out_dir: Path, seed: int, n_rows: int = 200) -> Path:
    df, manifest = make_dirty(seed, n_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"orders_{seed}.csv"
    df.to_csv(csv_path, index=False)
    (out_dir / f"orders_{seed}.manifest.json").write_text(manifest.model_dump_json(indent=2))
    return csv_path


# --- Outcome checks: True when the fault is gone from the cleaned frame. ---

_MARKERS = {"n/a", "?", "unknown"}


def _no_markers(df: pd.DataFrame, f: Fault) -> bool:
    if f.column not in df:
        return False
    values = df[f.column].dropna().astype(str).str.strip().str.lower()
    return not values.isin(_MARKERS).any()


def _no_padding(df: pd.DataFrame, f: Fault) -> bool:
    if f.column not in df:
        return False
    s = df[f.column].dropna().astype(str)
    return bool((s == s.str.strip()).all())


def _is_numeric(df: pd.DataFrame, f: Fault) -> bool:
    return f.column in df and pd.api.types.is_numeric_dtype(df[f.column])


def _no_negatives(df: pd.DataFrame, f: Fault) -> bool:
    if f.column not in df:
        return False
    return not (pd.to_numeric(df[f.column], errors="coerce") < 0).any()


def _no_nulls(df: pd.DataFrame, f: Fault) -> bool:
    return f.column in df and not df[f.column].isna().any()


def _is_datetime(df: pd.DataFrame, f: Fault) -> bool:
    return f.column in df and pd.api.types.is_datetime64_any_dtype(df[f.column])


def _dropped(df: pd.DataFrame, f: Fault) -> bool:
    return f.column not in df


def _no_duplicates(df: pd.DataFrame, f: Fault) -> bool:
    return not df.duplicated().any()


CHECKS: dict[str, Callable[[pd.DataFrame, Fault], bool]] = {
    "missing_tokens": _no_markers,
    "whitespace_padding": _no_padding,
    "numeric_as_text": _is_numeric,
    "impossible_values": _no_negatives,
    "missing_values": _no_nulls,
    "mixed_date_formats": _is_datetime,
    "constant_column": _dropped,
    "duplicate_rows": _no_duplicates,
}


def check_all(df: pd.DataFrame, manifest: Manifest) -> dict[str, bool]:
    return {f.kind: CHECKS[f.kind](df, f) for f in manifest.faults}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write synthetic dirty CSV fixtures.")
    parser.add_argument("--out", type=Path, default=Path("fixtures"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args()
    for seed in args.seeds:
        print(write_fixture(args.out, seed, args.rows))


if __name__ == "__main__":
    main()

