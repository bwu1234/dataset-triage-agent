import logging

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from tests.test_profile_and_faults import REFERENCE_PLAN
from triage.config import RiskPolicy, Settings
from triage.faults import check_all, make_dirty, write_fixture
from triage.graph import build_graph, run_config
from triage.ops import CleaningPlan, DropColumn, FilterRows, StripWhitespace

PERMISSIVE = RiskPolicy(max_rows_removed=10**6, max_columns_removed=10**6, max_values_nulled=10**6)


def _planner(ops):
    plan = CleaningPlan(ops=ops)
    return RunnableLambda(lambda _: {"raw": AIMessage("{}"), "parsed": plan, "parsing_error": None})


def _reject(_req):
    return {"action": "reject"}


def _run(tmp_path, ops, risk=None, thread="t1", planner=None, decide=_reject):
    """Run to the end, answering each approval with ``decide(request)``."""
    final, _ = _drive(tmp_path, ops, risk, thread, planner, decide)
    return final


def _drive(tmp_path, ops, risk=None, thread="t1", planner=None, decide=_reject):
    settings = Settings(runs_dir=tmp_path / "runs", **({"risk": risk} if risk else {}))
    graph = build_graph(settings, planner=planner or _planner(ops))
    csv = write_fixture(tmp_path / "in", seed=0)
    config, inp, asked = run_config(thread), {"input_path": str(csv)}, []
    while True:
        graph.invoke(inp, config)
        state = graph.get_state(config)
        if not state.interrupts:
            return state.values, (graph, asked)
        request = state.interrupts[0].value
        asked.append(request.op_index)
        inp = Command(resume=decide(request))


def _actions(final):
    return [(e.action, e.op_index) for e in final["audit"]]


def test_reference_plan_runs_end_to_end(tmp_path):
    final, (_, asked) = _drive(tmp_path, REFERENCE_PLAN, risk=PERMISSIVE)
    assert asked == []
    assert final["status"] == "done"
    assert [a for a, _ in _actions(final)].count("applied") == len(REFERENCE_PLAN)
    _assert_all_faults_fixed(final)


def _assert_all_faults_fixed(final):
    out = pd.read_csv(final["output_path"], keep_default_na=False, na_values=[""],
                      parse_dates=["order_date"])
    _, manifest = make_dirty(0)
    failed = {k for k, ok in check_all(out, manifest).items() if not ok}
    assert not failed


def test_destructive_ops_pause_for_approval_and_rejects_are_not_applied(tmp_path):
    final, (_, asked) = _drive(tmp_path, REFERENCE_PLAN)
    # standardize_missing x2, filter_rows, drop_column, dedupe: each would null
    # values or remove rows/columns on this fixture. The amount cast (3) asks
    # too: with its markers rejected, they would parse to null.
    assert asked == [1, 2, 3, 4, 7, 8]
    rejected = [e for e in final["audit"] if e.action == "rejected"]
    assert [e.op_index for e in rejected] == asked
    assert rejected[0].impact.values_nulled > 0
    assert "channel" in pd.read_csv(final["output_path"]).columns


def test_approving_everything_fixes_every_fault(tmp_path):
    final, (_, asked) = _drive(tmp_path, REFERENCE_PLAN, decide=lambda _: {"action": "approve"})
    # Once the markers are nulled with approval, the amount cast is lossless.
    assert asked == [1, 2, 4, 7, 8]
    _assert_all_faults_fixed(final)


def test_approve_records_one_entry_despite_node_rerun(tmp_path):
    # approve runs twice per approval (pause, then resume from the top); only
    # the resumed run's return value reaches state.
    final, _ = _drive(tmp_path, [DropColumn(column="channel", reason="r")],
                      decide=lambda _: {"action": "approve", "note": "unused column"})
    assert _actions(final)[2:5] == [("approved", 0), ("applied", 0), ("finished", None)]
    assert final["audit"][2].detail == "unused column"


def test_edit_replaces_op_and_reassesses_it(tmp_path):
    keep_all = {"op": "filter_rows", "column": "quantity", "operator": ">=", "value": -1e9,
                "reason": "edited"}
    answers = iter([{"action": "edit", "op": keep_all}])
    final, (_, asked) = _drive(
        tmp_path, [FilterRows(column="quantity", operator=">=", value=0.0, reason="r")],
        decide=lambda _: next(answers),
    )
    # The edit removes no rows, so route sends it to apply without asking again.
    assert asked == [0]
    assert _actions(final)[2:4] == [("edited", 0), ("applied", 0)]
    assert final["plan"].ops[0].value == -1e9


def test_edit_that_is_still_risky_asks_again(tmp_path):
    still_drops = {"op": "filter_rows", "column": "quantity", "operator": ">=", "value": 1.0,
                   "reason": "edited"}
    answers = iter([{"action": "edit", "op": still_drops}, {"action": "reject"}])
    final, (_, asked) = _drive(
        tmp_path, [FilterRows(column="quantity", operator=">=", value=0.0, reason="r")],
        decide=lambda _: next(answers),
    )
    assert asked == [0, 0]
    assert _actions(final)[2:4] == [("edited", 0), ("rejected", 0)]


def test_op_error_is_logged_and_run_continues(tmp_path):
    ops = [DropColumn(column="nope", reason="r"), StripWhitespace(column="customer", reason="r")]
    final = _run(tmp_path, ops, risk=PERMISSIVE)
    assert _actions(final)[2:4] == [("skipped", 0), ("applied", 1)]
    assert "'nope' not found" in final["audit"][2].detail
    assert final["status"] == "done"


def test_planner_failure_finishes_failed(tmp_path):
    bad = RunnableLambda(lambda _: {"raw": AIMessage("{x"), "parsed": None,
                                    "parsing_error": ValueError("bad json")})
    final = _run(tmp_path, [], planner=bad)
    assert final["status"] == "failed" and final["last_error"] == "bad json"
    assert "output_path" not in final


def test_checkpoint_revives_typed_state_without_warnings(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        values, _ = _drive(tmp_path, REFERENCE_PLAN)
    assert isinstance(values["plan"], CleaningPlan)
    assert values["plan"].ops == REFERENCE_PLAN
    assert all(type(e).__name__ == "AuditEntry" for e in values["audit"])
    assert not [r for r in caplog.records if "langgraph" in r.name]


def test_rerun_reuses_step_files(tmp_path):
    first, (graph, _) = _drive(tmp_path, REFERENCE_PLAN, risk=PERMISSIVE)
    run_dir = tmp_path / "runs" / "t1"
    files = sorted(p.name for p in run_dir.glob("step_*.pkl"))
    second = graph.invoke({"input_path": first["input_path"]}, run_config("t1"))
    assert sorted(p.name for p in run_dir.glob("step_*.pkl")) == files
    assert second["current_path"] == first["current_path"]
