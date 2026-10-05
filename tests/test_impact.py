import pandas as pd

from triage.config import RiskPolicy
from triage.impact import assess, needs_approval
from triage.ops import CastType, Dedupe, DropColumn, Impute, StripWhitespace

POLICY = RiskPolicy()


def test_lossless_cast_is_safe():
    frame = pd.DataFrame({"a": ["1", "2"]})
    impact = assess(frame, CastType(column="a", to="int", reason="r"))
    assert (impact.values_nulled, impact.values_modified) == (0, 0)
    assert not needs_approval(impact, POLICY)


def test_lossy_cast_needs_approval():
    frame = pd.DataFrame({"a": ["1", "oops"]})
    impact = assess(frame, CastType(column="a", to="int", reason="r"))
    assert impact.values_nulled == 1
    assert needs_approval(impact, POLICY)


def test_impute_counts_fills_and_is_safe():
    impact = assess(pd.DataFrame({"b": [1.0, None]}), Impute(column="b", strategy="mean", reason="r"))
    assert impact.values_filled == 1
    assert not needs_approval(impact, POLICY)


def test_strip_counts_modifications_and_is_safe():
    impact = assess(pd.DataFrame({"c": [" x", "y"]}), StripWhitespace(column="c", reason="r"))
    assert impact.values_modified == 1
    assert not needs_approval(impact, POLICY)


def test_removals_need_approval():
    frame = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
    assert assess(frame, Dedupe(reason="r")).rows_removed == 1
    assert assess(frame, DropColumn(column="b", reason="r")).columns_removed == 1
    assert needs_approval(assess(frame, Dedupe(reason="r")), POLICY)


def test_policy_limits_are_configurable():
    impact = assess(pd.DataFrame({"a": [1, 1]}), Dedupe(reason="r"))
    assert not needs_approval(impact, RiskPolicy(max_rows_removed=1))
