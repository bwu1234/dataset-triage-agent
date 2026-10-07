import pandas as pd
import pytest

from triage.config import Settings
from triage.executor import apply_op
from triage.faults import FAULT_KINDS, check_all, make_dirty, write_fixture
from triage.io import load_csv
from triage.ops import (
    CastType,
    Dedupe,
    DropColumn,
    FilterRows,
    Impute,
    StandardizeMissing,
    StripWhitespace,
)
from triage.profile import profile

SETTINGS = Settings()

# A hand-written plan that fixes every injected fault. Test-only: it proves the
# op set and checks agree, and it must never reach the planner.
REFERENCE_PLAN = [
    StripWhitespace(column="customer", reason="r"),
    StandardizeMissing(column="region", tokens=["N/A", "?", "unknown"], reason="r"),
    StandardizeMissing(column="amount", tokens=["unknown", "n/a"], reason="r"),
    CastType(column="amount", to="float", reason="r"),
    FilterRows(column="quantity", operator=">=", value=0.0, reason="r"),
    Impute(column="rating", strategy="median", reason="r"),
    CastType(column="order_date", to="datetime", reason="r"),
    DropColumn(column="channel", reason="r"),
    Dedupe(reason="r"),
]


@pytest.fixture(params=[0, 1, 2])
def fixture_frame(request, tmp_path):
    path = write_fixture(tmp_path, seed=request.param)
    _, manifest = make_dirty(request.param)
    return load_csv(path, SETTINGS.csv_na_values), manifest


def test_generation_is_deterministic():
    a, ma = make_dirty(7)
    b, mb = make_dirty(7)
    pd.testing.assert_frame_equal(a, b)
    assert ma == mb


def test_every_fault_kind_is_injected():
    _, manifest = make_dirty(0)
    assert [f.kind for f in manifest.faults] == list(FAULT_KINDS)


def test_all_faults_are_detected_on_dirty_data(fixture_frame):
    df, manifest = fixture_frame
    assert not any(check_all(df, manifest).values())


def test_reference_plan_fixes_every_fault(fixture_frame):
    df, manifest = fixture_frame
    for op in REFERENCE_PLAN:
        df = apply_op(df, op)
    assert check_all(df, manifest) == dict.fromkeys(FAULT_KINDS, True)


def test_profile_surfaces_fault_hints(fixture_frame):
    df, manifest = fixture_frame
    p = profile(df, SETTINGS.profiler)
    cols = {c.name: c for c in p.columns}
    assert p.duplicate_rows > 0
    assert cols["region"].missing_token_count > 0
    # Every injected marker is named, not just those among the samples.
    assert set(cols["region"].missing_tokens_found) == set(df["region"][
        df["region"].str.lower().isin(["n/a", "?", "unknown"])])
    assert cols["customer"].missing_tokens_found == []
    assert cols["customer"].whitespace_padded_count > 0
    assert 0 < cols["amount"].numeric_parse_rate < 1
    assert cols["rating"].null_count > 0
    assert cols["quantity"].min < 0
    assert cols["quantity"].negative_count == int((df["quantity"] < 0).sum())
    assert cols["order_id"].negative_count == 0
    assert cols["channel"].distinct_count == 1


def test_profile_truncates_samples():
    frame = pd.DataFrame({"t": ["x" * 100]})
    p = profile(frame, SETTINGS.profiler)
    assert len(p.columns[0].sample_values[0]) == SETTINGS.profiler.max_sample_chars


def test_missing_tokens_found_lists_one_spelling_per_token():
    s = pd.Series(["a", " N/A", "n/a ", "?", "b", "NULL"])
    [col] = profile(pd.DataFrame({"x": s}), SETTINGS.profiler).columns
    assert col.missing_token_count == 4
    # Sorted by lowercased token; the first spelling seen wins.
    assert col.missing_tokens_found == ["?", "N/A", "NULL"]


def test_negative_count_covers_numeric_text_only():
    frame = pd.DataFrame({
        "n": [-1.5, 2.0, None, -3.0],
        "t": ["-2", " 5", "n/a", "-0.5"],
        "words": ["a", "b", "c", "d"],
        "flag": [True, False, True, False],
    })
    cols = {c.name: c for c in profile(frame, SETTINGS.profiler).columns}
    assert cols["n"].negative_count == 2
    # Text is counted on its parsed values; markers do not parse.
    assert cols["t"].negative_count == 2
    assert cols["words"].negative_count is None
    assert cols["flag"].negative_count is None
