from contextlib import ExitStack

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from tests.cli_child import PLAN, PLANNER
from triage import cli
from triage.config import Settings
from triage.faults import write_fixture
from triage.graph import build_graph, open_checkpointer, run_config
from triage.io import load_frame
from triage.ops import CleaningPlan, StripWhitespace
from triage.writeup import run_path


def _approve(_request):
    return cli.ApprovalDecision(action="approve")


def _reject(_request):
    return cli.ApprovalDecision(action="reject", note="keep it")


def _stop(_request):
    raise EOFError  # what Ctrl-D at the prompt raises


@pytest.fixture
def env(tmp_path):
    """A finished run on thread 't' (strip, then an approved drop of 'channel')
    in a SQLite checkpoint store, and a function that opens the graph on a new
    connection, as a new process would."""
    settings = Settings(runs_dir=tmp_path / "runs", checkpoint_db=tmp_path / "c.sqlite")
    csv = write_fixture(tmp_path / "in", 0)
    with ExitStack() as stack:
        def graph(planner=PLANNER):
            return build_graph(settings, planner,
                               stack.enter_context(open_checkpointer(settings.checkpoint_db)))

        assert cli.run(csv, "t", graph(), _approve, lambda _: None) == 0
        yield graph, csv


def _checkpoints(g):
    return list(g.get_state_history(run_config("t")))


def _at(g, pred):
    return next(cli._checkpoint_id(s.config) for s in _checkpoints(g) if pred(s))


def _actions(values):
    return [(e.action, e.op_index) for e in values["audit"]]


def test_fork_at_an_approval_asks_again_and_keeps_both_branches(env):
    graph, _ = env
    g = graph()
    first = g.get_state(run_config("t")).values
    asked, lines = [], []

    def ask(request):
        asked.append(request.op_index)
        return _reject(request)

    approval = _at(g, lambda s: s.interrupts)
    assert cli.fork("t", approval, g, ask, lines.append) == 0
    assert asked == [1] and "forking at approve" in lines

    fork = g.get_state(run_config("t")).values
    assert _actions(fork) == [("loaded", None), ("planned", None), ("applied", 0),
                              ("rejected", 1), ("validated", None), ("finished", None)]
    assert "channel" in pd.read_csv(fork["output_path"]).columns
    # The original branch's output is a different file and still as it was.
    assert fork["output_path"] != first["output_path"]
    assert "channel" not in pd.read_csv(first["output_path"]).columns

    shown = []
    assert cli.history("t", g, shown.append) == 0
    rows = shown[2:]
    assert len(rows) == len(_checkpoints(g))
    assert any(r.startswith("*") and f"fork of {approval}" in r for r in rows)
    old_end = cli._checkpoint_id(next(s for s in _checkpoints(g)
                                      if s.values.get("output_path") == first["output_path"]
                                      and not s.next).config)
    assert any(r.startswith(" ") and old_end in r for r in rows)
    assert sum(r.startswith("*") and "asks: #1 drop_column" in r for r in rows) == 2

    # The diagram follows the fork, from the shared steps through the new answer.
    diagram = run_path(g, "t")
    assert f"<b>fork</b><br/>from checkpoint {approval}" in diagram
    assert ":::rejected" in diagram and ":::approved" not in diagram


def test_resume_command_at_an_old_checkpoint_keeps_the_old_answer(env):
    # Why ``fork`` streams ``None`` instead: pinned so a LangGraph upgrade
    # that changes it is noticed.
    graph, _ = env
    g = graph()
    at = {"configurable": {"thread_id": "t", "checkpoint_id": _at(g, lambda s: s.interrupts)}}
    g.invoke(Command(resume={"action": "reject"}), at)
    assert ("approved", 1) in _actions(g.get_state(run_config("t")).values)


def test_fork_refuses_unknown_and_final_checkpoints(env):
    graph, _ = env
    g = graph()
    count, lines = len(_checkpoints(g)), []
    assert cli.fork("t", "nope", g, _approve, lines.append) == 2
    assert cli.fork("t", _at(g, lambda s: not s.next), g, _approve, lines.append) == 2
    assert "no checkpoint" in lines[0] and "end of a run" in lines[1]
    assert len(_checkpoints(g)) == count
    assert cli.history("nope", g, lines.append) == 2


def test_stopped_fork_is_continued_by_resume(env):
    graph, _ = env
    g = graph()
    with pytest.raises(EOFError):
        cli.fork("t", _at(g, lambda s: s.interrupts), g, _stop, lambda _: None)
    g = graph()
    lines = []
    assert cli.resume("t", g, _reject, lines.append) == 0
    assert "resuming at approve" in lines
    assert ("rejected", 1) in _actions(g.get_state(run_config("t")).values)


def test_fork_before_plan_asks_the_model_again(env):
    graph, _ = env
    other = CleaningPlan(ops=[StripWhitespace(column="customer", reason="only this")])
    plans = iter([other])
    planner = RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": next(plans),
                                        "parsing_error": None})
    g = graph(planner)
    lines = []
    assert cli.fork("t", _at(g, lambda s: s.next == ("plan",)), g, _approve, lines.append) == 0
    assert "forking at plan; this asks the model for a new plan" in lines
    values = g.get_state(run_config("t")).values
    assert values["plan"] == other and values["plan"] != PLAN


def test_fork_from_load_after_the_csv_changed_leaves_the_old_branch_intact(env, tmp_path):
    graph, csv = env
    g = graph()
    first = g.get_state(run_config("t")).values
    before = load_frame(first["start_path"]).copy()
    csv.write_bytes(write_fixture(tmp_path / "other", 1).read_bytes())

    assert cli.fork("t", _at(g, lambda s: s.next == ("load",)), g, _approve,
                    lambda _: None) == 0
    fork = g.get_state(run_config("t")).values
    assert fork["start_path"] != first["start_path"]
    pd.testing.assert_frame_equal(load_frame(first["start_path"]), before)


def test_main_wires_history_and_fork(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRIAGE_CHECKPOINT_DB", str(tmp_path / "c.sqlite"))
    monkeypatch.setenv("TRIAGE_RUNS_DIR", str(tmp_path / "runs"))
    assert cli.main(["history", "--thread", "nope"]) == 2
    assert cli.main(["fork", "--thread", "nope", "--checkpoint", "x"]) == 2
    out = capsys.readouterr().out
    assert "no run with thread 'nope'" in out and "no checkpoint 'x'" in out
    with pytest.raises(SystemExit):
        cli.main(["fork", "--thread", "t"])
