import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from triage.cli import ask_approval, main, run, safe
from triage.config import Settings
from triage.faults import write_fixture
from triage.graph import ApprovalRequest, build_graph, check_thread_id
from triage.impact import Impact
from triage.ops import CleaningPlan, DropColumn, StripWhitespace

REQUEST = ApprovalRequest(
    op_index=3, op=DropColumn(column="channel", reason="r"),
    impact=Impact(rows_removed=0, columns_removed=1, values_nulled=0, values_filled=0,
                  values_modified=0),
)


def _scripted(*answers):
    it = iter(answers)
    return lambda _prompt: next(it)


def test_safe_escapes_terminal_control_characters():
    assert safe("a\x1b[2Jb\nc") == "a\\x1b[2Jb\\nc"


@pytest.mark.parametrize("bad", ["../x", "a/b", ".hidden", "", "x" * 101])
def test_thread_id_must_be_a_plain_name(bad):
    with pytest.raises(ValueError):
        check_thread_id(bad)


def test_ask_approval_choices():
    assert ask_approval(REQUEST, _scripted("a"), lambda _: None).action == "approve"
    rejected = ask_approval(REQUEST, _scripted("r", "still used"), lambda _: None)
    assert (rejected.action, rejected.note) == ("reject", "still used")


def test_ask_approval_reprompts_on_invalid_edit():
    shown = []
    replacement = '{"op": "strip_whitespace", "column": "channel", "reason": "keep it"}'
    decision = ask_approval(
        REQUEST, _scripted("x", "e", '{"op": "run_shell"}', "e", replacement), shown.append
    )
    assert decision.action == "edit" and decision.op == StripWhitespace(column="channel",
                                                                       reason="keep it")
    assert any(line.startswith("Not a valid op") for line in shown)


def test_run_streams_progress_and_resumes_after_each_approval(tmp_path):
    plan = CleaningPlan(ops=[StripWhitespace(column="customer", reason="r"),
                             DropColumn(column="channel", reason="r")])
    planner = RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": plan, "parsing_error": None})
    graph = build_graph(Settings(runs_dir=tmp_path / "runs"), planner=planner)
    asked, lines = [], []

    def ask(request):
        asked.append(request.op_index)
        return ask_approval(request, _scripted("a"), lines.append)

    code = run(write_fixture(tmp_path, 0), "t", graph, ask, lines.append)
    assert code == 0 and asked == [1]
    actions = [line.split()[0] for line in lines if line and not line.startswith(" ")]
    assert actions == ["loaded", "planned", "applied", "Approval", "approved", "applied",
                       "validated", "finished", "done:"]


def test_main_rejects_missing_csv(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", str(tmp_path / "nope.csv"), "--thread", "t"])
    assert exc.value.code == 2
    assert "no such file" in capsys.readouterr().err


def test_failed_run_reports_each_validation_failure_on_its_own_line(tmp_path):
    bad = CleaningPlan(ops=[DropColumn(column="nope", reason="r"),
                            DropColumn(column="gone\x1b[2J", reason="r")])
    planner = RunnableLambda(lambda _: {"raw": AIMessage(""), "parsed": bad, "parsing_error": None})
    graph = build_graph(Settings(runs_dir=tmp_path / "runs", max_plan_retries=0), planner=planner)
    lines = []
    assert run(write_fixture(tmp_path, 0), "t", graph, lambda _: None, lines.append) == 1
    assert lines[-2].startswith("failed: op #0") and lines[-1].startswith("        op #1")
    assert "\x1b" not in "".join(lines)
