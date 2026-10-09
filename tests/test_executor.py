import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

from triage.executor import OpError, apply_op
from triage.ops import (
    CastType,
    CleaningPlan,
    Dedupe,
    DropColumn,
    FilterRows,
    Impute,
    Op,
    StandardizeMissing,
    StripWhitespace,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({
        "a": ["1", "2", "x", None],
        "b": [1.0, None, 3.0, 4.0],
        "c": [" p ", "q", "N/A", "q"],
    })


def test_apply_op_does_not_mutate_input(df):
    before = df.copy()
    apply_op(df, CastType(column="a", to="int", reason="r"))
    pd.testing.assert_frame_equal(df, before)


def test_missing_column_raises_op_error(df):
    with pytest.raises(OpError, match="not found"):
        apply_op(df, DropColumn(column="zzz", reason="r"))


def test_drop_column(df):
    assert "a" not in apply_op(df, DropColumn(column="a", reason="r"))


@pytest.mark.parametrize("to, expected", [
    ("int", [1, 2, pd.NA, pd.NA]),
    ("float", [1.0, 2.0, pd.NA, pd.NA]),
])
def test_cast_numeric_nulls_unparseable(df, to, expected):
    out = apply_op(df, CastType(column="a", to=to, reason="r"))
    assert out["a"].tolist() == pd.array(expected, dtype=out["a"].dtype).tolist()


def test_cast_int_nulls_non_integral_instead_of_truncating():
    out = apply_op(pd.DataFrame({"a": [1.0, 2.5]}), CastType(column="a", to="int", reason="r"))
    assert out["a"].isna().tolist() == [False, True]


def test_cast_datetime_accepts_mixed_formats():
    frame = pd.DataFrame({"d": ["2025-03-01", "02 Mar 2025"]})
    out = apply_op(frame, CastType(column="d", to="datetime", reason="r"))
    assert out["d"].dt.day.tolist() == [1, 2]


def test_cast_bool():
    frame = pd.DataFrame({"m": ["Yes", "no", "maybe"]})
    out = apply_op(frame, CastType(column="m", to="bool", reason="r"))
    assert out["m"].tolist() == [True, False, pd.NA]


def test_impute_median(df):
    assert apply_op(df, Impute(column="b", strategy="median", reason="r"))["b"].tolist() == [1.0, 3.0, 3.0, 4.0]


def test_impute_mean_on_text_column_raises(df):
    with pytest.raises(OpError, match="numeric"):
        apply_op(df, Impute(column="c", strategy="mean", reason="r"))


def test_impute_text_constant_into_nullable_float_raises_op_error():
    # pandas raises TypeError here; route only catches OpError, so a raw
    # TypeError would crash the run instead of skipping the op.
    frame = pd.DataFrame({"x": pd.array([1.5, None], dtype="Float64")})
    with pytest.raises(OpError, match="cannot fill"):
        apply_op(frame, Impute(column="x", strategy="constant", value="zero", reason="r"))


def test_impute_validates_constant_value():
    with pytest.raises(ValidationError):
        Impute(column="b", strategy="constant", reason="r")
    with pytest.raises(ValidationError):
        Impute(column="b", strategy="mean", value=1.0, reason="r")


def test_dedupe_keeps_original_index():
    frame = pd.DataFrame({"a": [1, 1, 2]})
    assert apply_op(frame, Dedupe(reason="r")).index.tolist() == [0, 2]


def test_filter_rows_drops_nulls_and_failures(df):
    out = apply_op(df, FilterRows(column="b", operator=">=", value=3.0, reason="r"))
    assert out["b"].tolist() == [3.0, 4.0]


def test_filter_rows_not_null_rejects_value():
    with pytest.raises(ValidationError):
        FilterRows(column="b", operator="not_null", value=1.0, reason="r")


def test_standardize_missing_is_case_and_space_insensitive(df):
    out = apply_op(df, StandardizeMissing(column="c", tokens=["n/a"], reason="r"))
    assert out["c"].isna().tolist() == [False, False, True, False]


def test_strip_whitespace(df):
    assert apply_op(df, StripWhitespace(column="c", reason="r"))["c"].tolist()[0] == "p"


def test_plan_parses_ops_by_discriminator():
    plan = CleaningPlan.model_validate_json(
        '{"ops": [{"op": "dedupe", "reason": "r"}, {"op": "drop_column", "column": "x", "reason": "r"}]}'
    )
    assert [type(o) for o in plan.ops] == [Dedupe, DropColumn]


def test_unknown_op_and_extra_fields_are_rejected():
    adapter = TypeAdapter(Op)
    with pytest.raises(ValidationError):
        adapter.validate_python({"op": "run_python", "reason": "r"})
    with pytest.raises(ValidationError):
        adapter.validate_python({"op": "drop_column", "column": "x", "reason": "r", "sql": "1"})


@pytest.mark.parametrize("fmt", ["%Y-%m-%Y", "%Q"])
def test_bad_datetime_format_is_an_op_error(fmt):
    # Both crashed the graph in the M7 evaluation: route catches OpError only.
    df = pd.DataFrame({"d": ["2025-01-02", "3 Jan 2025"]})
    with pytest.raises(OpError, match="datetime_format"):
        apply_op(df, CastType(column="d", to="datetime", datetime_format=fmt, reason="r"))


@pytest.mark.parametrize("fmt", ["YYYY-MM-DD", "yyyy/mm/dd", "2025-01-02"])
def test_datetime_format_without_a_directive_is_refused(fmt):
    # pandas matches no value against literal text and nulls them all.
    with pytest.raises(ValidationError, match="no strftime directive"):
        CastType(column="d", to="datetime", datetime_format=fmt, reason="r")


@pytest.mark.parametrize(("fmt", "value"), [(None, "2025-01-02"), ("%Y-%m-%d", "2025-01-02"),
                                           ("%d %b %Y", "2 Jan 2025"), ("mixed", "2 Jan 2025"),
                                           ("ISO8601", "2025-01-02")])
def test_datetime_formats_pandas_understands_are_accepted(fmt, value):
    df = pd.DataFrame({"d": [value]})
    out = apply_op(df, CastType(column="d", to="datetime", datetime_format=fmt, reason="r"))
    assert out["d"].notna().all()
