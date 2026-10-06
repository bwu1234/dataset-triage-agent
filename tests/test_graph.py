import logging

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from tests.test_profile_and_faults import REFERENCE_PLAN
from triage.config import RiskPolicy, Settings
from triage.faults import check_all, make_dirty, write_fixture
from triage.graph import build_graph, run_config
from triage.ops import CleaningPlan, Dedupe, DropColumn, FilterRows, StripWhitespace

PERMISSIVE = RiskPolicy(max_rows_removed=10**6, max_columns_removed=10**6, max_values_nulled=10**6)


def _planner(ops):
    plan = CleaningPlan(ops=ops)
    return RunnableLambda(lambda _: {"raw": AIMessage("{}"), "parsed": plan, "parsing_error": None})


def _planners(plans, seen=None):
    """A fake planner returning each plan in turn; records the prompt it got."""
    it = iter(CleaningPlan(ops=ops) for ops in plans)

    def call(messages):
        if seen is not None:
            seen.append(messages[-1].content)
        return {"raw": AIMessage("{}"), "parsed": next(it), "parsing_error": None}
    return RunnableLambda(call)


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
    assert _actions(final)[2:6] == [("approved", 0), ("applied", 0), ("validated", None),
                                    ("finished", None)]
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


def test_op_error_is_logged_and_run_continues_to_validation(tmp_path):
    ops = [DropColumn(column="nope", reason="r"), StripWhitespace(column="customer", reason="r")]
    final = _run(tmp_path, ops, risk=PERMISSIVE, planner=_planners([ops, ops[1:]]))
    assert _actions(final)[2:5] == [("skipped", 0), ("applied", 1), ("invalid", None)]
    assert "'nope' not found" in final["audit"][2].detail
    assert final["status"] == "done" and final["retries"] == 1


def test_planner_failure_finishes_failed(tmp_path):
    bad = RunnableLambda(lambda _: {"raw": AIMessage("{x"), "parsed": None,
                                    "parsing_error": ValueError("bad json")})
    final = _run(tmp_path, [], planner=bad)
    assert final["status"] == "failed" and final["last_error"] == "ValueError: bad json"
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


def test_failed_validation_replans_from_the_loaded_file_with_feedback(tmp_path):
    # Removes about half the rows: each filter is approved on its own, but the
    # total is over the validation policy.
    greedy = [FilterRows(column="quantity", operator=">=", value=6.0, reason="r")]
    fixed = [FilterRows(column="quantity", operator=">=", value=0.0, reason="r")]
    seen = []
    final = _run(tmp_path, [], planner=_planners([greedy, fixed], seen),
                 decide=lambda _: {"action": "approve"})
    assert [a for a, _ in _actions(final)] == [
        "loaded", "planned", "approved", "applied", "invalid",
        "planned", "approved", "applied", "validated", "finished"]
    assert final["status"] == "done" and final["retries"] == 1
    # The second attempt applied to the loaded file, not to the greedy output.
    assert final["audit"][7].impact.rows_removed == 16
    assert "<feedback>" not in seen[0]
    assert "rows removed" in seen[1] and '"value":6.0' in seen[1]


def test_feedback_names_ops_a_person_rejected(tmp_path):
    ops = [DropColumn(column="channel", reason="r"), DropColumn(column="nope", reason="r")]
    seen = []
    final = _run(tmp_path, [], planner=_planners([ops, []], seen))
    assert final["status"] == "done"
    assert "A person rejected these ops" in seen[1] and '"column":"channel"' in seen[1]
    assert "'nope' not found" in seen[1]


def test_retries_exhausted_finishes_failed_without_output(tmp_path):
    bad = [DropColumn(column="nope", reason="r")]
    calls = []
    final = _run(tmp_path, [], risk=PERMISSIVE, planner=_planners([bad] * 3, calls))
    assert len(calls) == 3  # first call plus max_plan_retries=2
    assert final["status"] == "failed" and final["retries"] == 2
    assert "output_path" not in final and "'nope' not found" in final["last_error"]
    assert _actions(final)[-1] == ("finished", None)
    assert not (tmp_path / "runs" / "t1" / "cleaned.csv").exists()


def test_planner_failure_is_retried_with_the_error_as_feedback(tmp_path):
    seen = []
    replies = iter([{"parsed": None, "parsing_error": ValueError("bad json")},
                    {"parsed": CleaningPlan(ops=[]), "parsing_error": None}])

    def call(messages):
        seen.append(messages[-1].content)
        return {"raw": AIMessage("{x"), **next(replies)}
    final = _run(tmp_path, [], planner=RunnableLambda(call))
    assert final["status"] == "done"
    assert [a for a, _ in _actions(final)][1:3] == ["plan_failed", "planned"]
    assert "bad json" in seen[1]


DEDUPE = Dedupe(reason="first wording")
SKIP = DropColumn(column="nope", reason="r")  # forces a replan


@pytest.mark.parametrize("second, reuse, asked_twice", [
    # Same op, new reason, same input bytes: the earlier answer is reused.
    ([Dedupe(reason="second wording")], True, False),
    # Same op on different bytes (strip runs first): asked again.
    ([StripWhitespace(column="customer", reason="r"), DEDUPE], True, True),
    # Reuse switched off: asked again.
    ([DEDUPE], False, True),
])
def test_replan_reuses_a_decision_only_for_the_same_op_on_the_same_data(
        tmp_path, second, reuse, asked_twice):
    risk = RiskPolicy(reuse_decisions=reuse)
    final, (_, asked) = _drive(tmp_path, [], risk=risk, planner=_planners([[DEDUPE, SKIP], second]),
                               decide=lambda _: {"action": "approve"})
    assert final["status"] == "done"
    assert len(asked) == (2 if asked_twice else 1)
    approvals = [e.detail for e in final["audit"] if e.action == "approved"]
    assert (approvals[-1] or "").startswith("reused") is not asked_twice


def test_reused_rejection_is_not_applied(tmp_path):
    drop = DropColumn(column="channel", reason="r")
    final, (_, asked) = _drive(tmp_path, [], planner=_planners([[drop, SKIP],
                                                                [drop.model_copy(update={"reason": "again"})]]))
    assert asked == [0]
    assert final["audit"][-3].action == "rejected" and final["audit"][-3].detail.startswith("reused")
    assert "channel" in pd.read_csv(final["output_path"]).columns
