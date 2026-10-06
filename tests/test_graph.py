import logging

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from tests.test_profile_and_faults import REFERENCE_PLAN
from triage.config import RiskPolicy, Settings
from triage.faults import check_all, make_dirty, write_fixture
from triage.graph import build_graph, run_config
from triage.ops import CleaningPlan, DropColumn, StripWhitespace

PERMISSIVE = RiskPolicy(max_rows_removed=10**6, max_columns_removed=10**6, max_values_nulled=10**6)


def _planner(ops):
    plan = CleaningPlan(ops=ops)
    return RunnableLambda(lambda _: {"raw": AIMessage("{}"), "parsed": plan, "parsing_error": None})


def _run(tmp_path, ops, risk=None, thread="t1", planner=None):
    settings = Settings(runs_dir=tmp_path / "runs", **({"risk": risk} if risk else {}))
    graph = build_graph(settings, planner=planner or _planner(ops))
    csv = write_fixture(tmp_path / "in", seed=0)
    final = graph.invoke({"input_path": str(csv)}, run_config(thread))
    return graph, final


def _actions(final):
    return [(e.action, e.op_index) for e in final["audit"]]


def test_reference_plan_runs_end_to_end(tmp_path):
    _, final = _run(tmp_path, REFERENCE_PLAN, risk=PERMISSIVE)
    assert final["status"] == "done"
    assert [a for a, _ in _actions(final)].count("applied") == len(REFERENCE_PLAN)
    out = pd.read_csv(final["output_path"], keep_default_na=False, na_values=[""],
                      parse_dates=["order_date"])
    _, manifest = make_dirty(0)
    failed = {k for k, ok in check_all(out, manifest).items() if not ok}
    assert not failed


def test_default_policy_holds_destructive_ops(tmp_path):
    _, final = _run(tmp_path, REFERENCE_PLAN)
    held = {i for a, i in _actions(final) if a == "held"}
    # standardize_missing x2, filter_rows, drop_column, dedupe: each would null
    # values or remove rows/columns on this fixture. The amount cast (3) is
    # held too: with its markers still in place, they would parse to null.
    assert held == {1, 2, 3, 4, 7, 8}
    held_entry = next(e for e in final["audit"] if e.action == "held")
    assert held_entry.impact is not None and held_entry.impact.values_nulled > 0
    assert "channel" in pd.read_csv(final["output_path"]).columns


def test_op_error_is_logged_and_run_continues(tmp_path):
    ops = [DropColumn(column="nope", reason="r"), StripWhitespace(column="customer", reason="r")]
    _, final = _run(tmp_path, ops, risk=PERMISSIVE)
    assert _actions(final)[2:4] == [("skipped", 0), ("applied", 1)]
    assert "'nope' not found" in final["audit"][2].detail
    assert final["status"] == "done"


def test_planner_failure_finishes_failed(tmp_path):
    bad = RunnableLambda(lambda _: {"raw": AIMessage("{x"), "parsed": None,
                                    "parsing_error": ValueError("bad json")})
    _, final = _run(tmp_path, [], planner=bad)
    assert final["status"] == "failed" and final["last_error"] == "bad json"
    assert "output_path" not in final


def test_checkpoint_revives_typed_state_without_warnings(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        graph, _ = _run(tmp_path, REFERENCE_PLAN)
        values = graph.get_state(run_config("t1")).values
    assert isinstance(values["plan"], CleaningPlan)
    assert values["plan"].ops == REFERENCE_PLAN
    assert all(type(e).__name__ == "AuditEntry" for e in values["audit"])
    assert not [r for r in caplog.records if "langgraph" in r.name]


def test_rerun_reuses_step_files(tmp_path):
    graph, first = _run(tmp_path, REFERENCE_PLAN, risk=PERMISSIVE)
    run_dir = tmp_path / "runs" / "t1"
    files = sorted(p.name for p in run_dir.glob("step_*.pkl"))
    second = graph.invoke({"input_path": first["input_path"]}, run_config("t1"))
    assert sorted(p.name for p in run_dir.glob("step_*.pkl")) == files
    assert second["current_path"] == first["current_path"]
