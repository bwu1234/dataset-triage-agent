import json

import pandas as pd
import pytest

from tests.test_graph import _planner, _planners
from tests.test_profile_and_faults import REFERENCE_PLAN
from triage import evaluate
from triage.config import EvaluationConfig, Settings
from triage.evaluate import EvalRow, collateral_rows, rule_based, run_one, summarize
from triage.faults import make_dirty
from triage.graph import ApprovalRequest
from triage.impact import Impact
from triage.ops import DropColumn, FilterRows, StandardizeMissing

NULL_REGION_FILTER = [
    StandardizeMissing(column="region", tokens=["N/A", "?", "unknown"], reason="r"),
    FilterRows(column="region", operator="not_null", reason="r"),
]


def _settings(tmp_path):
    return Settings(runs_dir=tmp_path / "runs")


@pytest.mark.parametrize("condition", ["approve_all", "rule_based"])
def test_reference_plan_scores_every_fault_with_no_collateral(tmp_path, condition):
    row = run_one(_settings(tmp_path), _planner(REFERENCE_PLAN), condition, 0, tmp_path / "fx")
    assert row.status == "done"
    assert row.faults_fixed == row.faults_total == 8
    assert row.collateral_rows == 0
    # Two standardize_missing, the quantity filter, drop channel, dedupe.
    assert (row.approvals, row.rejections) == (5, 0)
    assert row.model_calls == 1 and row.replans == 0


def test_dropping_rows_with_a_null_region_is_collateral(tmp_path):
    row = run_one(_settings(tmp_path), _planner(NULL_REGION_FILTER), "approve_all", 0,
                  tmp_path / "fx")
    dirty, _ = make_dirty(0)
    markers = dirty["region"].isin(["N/A", "?", "unknown"])
    # Distinct rows: a dropped row and its injected copy are one clean row. A
    # row that also has a negative quantity should go anyway, so is not counted.
    lost = dirty[markers & (pd.to_numeric(dirty["quantity"]) >= 0)].drop_duplicates()
    assert row.collateral_rows == len(lost) > 0
    assert row.rows_removed == markers.sum()


def test_rule_based_rejects_the_null_filter_and_keeps_the_rows(tmp_path):
    row = run_one(_settings(tmp_path), _planner(NULL_REGION_FILTER), "rule_based", 0,
                  tmp_path / "fx")
    assert row.status == "done"
    assert (row.approvals, row.rejections) == (2, 1)
    assert row.collateral_rows == 0


def test_dedupe_keep_last_is_not_collateral():
    dirty, manifest = make_dirty(0)
    start = dirty.reset_index(drop=True)
    final = start[~start.duplicated(keep="last")]
    quantity = pd.to_numeric(final["quantity"])
    final = final[quantity >= 0]
    assert collateral_rows(start, final, manifest) == 0
    assert collateral_rows(start, final.drop(final.index[:3]), manifest) == 3


def _request(op, rows_removed=0):
    return ApprovalRequest(op_index=0, op=op, impact=Impact(
        rows_removed=rows_removed, columns_removed=0, values_nulled=0, values_filled=0,
        values_modified=0))


def test_rule_based_approver():
    frame = pd.DataFrame({"a": range(100), "c": ["web"] * 100})
    decide = rule_based(EvaluationConfig(reject_rows_removed_fraction=0.1))
    filt = FilterRows(column="a", operator=">", value=5.0, reason="r")
    assert decide(_request(filt, 10), frame).action == "approve"
    assert decide(_request(filt, 11), frame).action == "reject"
    assert decide(_request(DropColumn(column="c", reason="r")), frame).action == "approve"
    assert decide(_request(DropColumn(column="a", reason="r")), frame).action == "reject"
    null_filter = FilterRows(column="a", operator="not_null", reason="r")
    assert decide(_request(null_filter, 1), frame).action == "reject"
    lenient = rule_based(EvaluationConfig(reject_null_filters=False))
    assert lenient(_request(null_filter, 1), frame).action == "approve"


def test_a_crash_stays_in_the_denominator(tmp_path, monkeypatch):
    def boom(_request, _frame):
        raise RuntimeError("approver died")
    monkeypatch.setattr(evaluate, "approve_all", boom)
    row = run_one(_settings(tmp_path), _planner(REFERENCE_PLAN), "approve_all", 0, tmp_path / "fx")
    assert row.status == "crashed" and "approver died" in row.error
    assert row.faults_fixed == 0 and row.faults_total == 8
    assert row.model_calls == 1
    assert row.plan and any(line.startswith("planned") for line in row.audit)


def test_failed_run_scores_zero_and_counts_replans(tmp_path):
    # A numeric comparison on a text column is skipped (OpError) on every
    # attempt, and a skipped op fails validation.
    bad = [FilterRows(column="customer", operator=">", value=1.0, reason="r")]
    settings = _settings(tmp_path)
    row = run_one(settings, _planners([bad] * 3), "approve_all", 0, tmp_path / "fx")
    assert row.status == "failed" and row.error
    assert row.faults_fixed == 0 and row.collateral_rows is None
    assert row.replans == settings.max_plan_retries == 2
    assert row.model_calls == 3


def test_summary_keeps_failures_in_the_denominator():
    rows = [
        EvalRow(model="m", think=False, condition="approve_all", seed=0, status="done",
                faults_fixed=8, faults_total=8, approvals=4, collateral_rows=0, model_calls=1,
                wall_seconds=10.0),
        EvalRow(model="m", think=False, condition="approve_all", seed=1, status="failed",
                faults_total=8, replans=2, model_calls=3, wall_seconds=30.0),
        EvalRow(model="m", think=False, condition="approve_all", seed=2, status="crashed",
                faults_total=8, wall_seconds=1.0),
    ]
    table = summarize(rows).splitlines()
    assert len(table) == 3
    cells = [c.strip() for c in table[2].strip("|").split("|")]
    assert cells[3:] == ["3", "1 / 1 / 1", "8/24", "1/3", "4", "0", "2", "1", "0 (0)", "4",
                         "10.0"]


def test_main_writes_results_and_summarizes_them(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "make_planner", lambda _config: _planner(REFERENCE_PLAN))
    assert evaluate.main(["--models", "fake", "--seeds", "0", "--out", str(tmp_path)]) == 0
    [results] = tmp_path.glob("*/results.jsonl")
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert [(r["condition"], r["faults_fixed"]) for r in rows] == [("approve_all", 8),
                                                                    ("rule_based", 8)]
    assert (results.parent / "summary.md").exists()
    capsys.readouterr()
    assert evaluate.main(["--summarize", str(results)]) == 0
    assert "| `fake` | False | rule_based | 1 |" in capsys.readouterr().out


@pytest.mark.live
def test_live_run_is_scored(tmp_path):
    from tests.test_planner import _model_available
    from triage.planner import make_planner

    model = "qwen3.5:9b-mlx"
    if not _model_available(model):
        pytest.skip(f"Ollama or {model} not available")
    settings = Settings(runs_dir=tmp_path / "runs")
    settings.model = settings.model.model_copy(update={"name": model})
    row = run_one(settings, make_planner(settings.model), "rule_based", 0, tmp_path / "fx")
    # Plan quality varies run to run; this checks the pipeline, not the score.
    assert row.status in ("done", "failed"), row.error
    assert row.model_calls >= 1 and row.audit
