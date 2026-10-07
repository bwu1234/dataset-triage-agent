import json

import ollama
import pytest

from triage import trace
from triage.cli import resume, run
from triage.config import Settings
from triage.faults import write_fixture
from triage.graph import ApprovalDecision, build_graph
from triage.trace import Tracing, _block, tracing

# The first reply does not parse, so the planner is called again with
# feedback; the second plan's drop needs approval under the default policy.
REPLIES = ['{"ops": [{"op": "drop_column"}]}',
           json.dumps({"ops": [{"op": "drop_column", "column": "channel", "reason": "unused"}]})]


@pytest.fixture
def ollama_stub(monkeypatch):
    """A real ChatOllama planner whose client replies from ``REPLIES``, and a
    prompt renderer that never touches the network."""
    replies, requests = iter(REPLIES), []

    def chat(self, **params):
        yield {"model": params["model"], "done": True, "done_reason": "stop",
               "message": {"role": "assistant", "content": next(replies)},
               "prompt_eval_count": 1200, "eval_count": 40,
               "prompt_eval_duration": 2 * 10**9, "eval_duration": 10**9}

    def render(base_url, request, timeout_s):
        requests.append(request)
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                       for m in request["messages"]) + "<|im_start|>assistant\n"

    monkeypatch.setattr(ollama.Client, "chat", chat)
    monkeypatch.setattr(trace, "render_prompt", render)
    return requests


def _approve(_request):
    return ApprovalDecision(action="approve")


def test_trace_records_steps_model_calls_and_prompt_delta(tmp_path, ollama_stub):
    settings = Settings(runs_dir=tmp_path / "runs")
    graph = build_graph(settings)
    csv = write_fixture(tmp_path, 0)
    code = run(csv, "t", graph, _approve, lambda _: None,
               traces=tracing(settings, "t", "`run`"))
    assert code == 0
    text = (tmp_path / "runs" / "t" / "trace.md").read_text()

    assert text.startswith("# Trace of thread `t`\n")
    for node in ("load", "profile", "plan", "route", "approve", "apply", "validate", "finish"):
        assert f"· `{node}`" in text
    assert "#### Model call 1 ·" in text and "#### Model call 2 ·" in text
    # The request the trace rendered is the one ChatOllama builds.
    assert ollama_stub[0]["model"] == settings.model.name
    assert ollama_stub[0]["options"]["num_predict"] == settings.model.num_predict
    # Same system message, new user message carrying the feedback.
    assert "| 0 | system |" in text and "| same |" in text and "| changed |" in text
    assert "**Rendered prompt:** call 2 in `trace_prompts.md`" in text
    assert "<|im_start|>" not in text
    assert "prompt 1200 tokens; output 40 tokens" in text and "40.0 tok/s" in text
    assert "**Tool calls:** none." in text
    assert "**Paused for approval.**" in text and '"op": "drop_column"' in text

    # The prompts file: each call's prompt, delta and reply, in order.
    prompts = (tmp_path / "runs" / "t" / "trace_prompts.md").read_text()
    assert prompts.startswith("# Rendered prompts of thread `t`\n")
    assert "### Call 1 · step 3 · `plan`" in prompts and "### Call 2 · step 4 · `plan`" in prompts
    assert "<|im_start|>system\nYou are a data-cleaning planner." in prompts
    delta = prompts.split("**Delta from call 1:**")[1].split("**Reply to call 2**")[0]
    assert "+The previous plan failed." in delta
    assert prompts.index("### Call 1") < prompts.index("**Reply to call 1**") \
        < prompts.index("### Call 2") < prompts.index("**Reply to call 2**")
    assert "### Step" not in prompts and "state update" not in prompts


def test_trace_appends_a_section_per_command(tmp_path, ollama_stub):
    settings = Settings(runs_dir=tmp_path / "runs")
    graph = build_graph(settings)
    csv = write_fixture(tmp_path, 0)

    def stop(_request):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(csv, "t", graph, stop, lambda _: None, traces=tracing(settings, "t", "`run`"))
    assert resume("t", graph, _approve, lambda _: None,
                  traces=tracing(settings, "t", "`resume`")) == 0
    text = (tmp_path / "runs" / "t" / "trace.md").read_text()
    assert text.count("# Trace of thread") == 1
    assert text.index("## `run`") < text.index("## `resume`")
    after = text.split("## `resume`")[1]
    assert "**Resumed with**" in after and '"action": "approve"' in after


def test_failed_render_is_noted_and_the_run_goes_on(tmp_path, ollama_stub, monkeypatch):
    def down(*_args):
        raise ConnectionRefusedError("no daemon")

    monkeypatch.setattr(trace, "render_prompt", down)
    settings = Settings(runs_dir=tmp_path / "runs")
    code = run(write_fixture(tmp_path, 0), "t", build_graph(settings), _approve, lambda _: None,
               traces=tracing(settings, "t", "`run`"))
    assert code == 0
    text = (tmp_path / "runs" / "t" / "trace.md").read_text()
    assert "**Rendered prompt:** call 1 in `trace_prompts.md`, not available." in text
    prompts = (tmp_path / "runs" / "t" / "trace_prompts.md").read_text()
    assert "Not available (ConnectionRefusedError: no daemon)." in prompts


def test_tracing_can_be_turned_off(tmp_path):
    settings = Settings(runs_dir=tmp_path / "runs",
                        trace={"enabled": False, "debug_enabled": False})
    assert tracing(settings, "t", "`run`") == Tracing([], None, [])


def test_block_fence_outgrows_backticks_in_the_text():
    assert _block("a ```` b").startswith("`````\n")


def test_debug_file_holds_both_libraries_own_output(tmp_path, ollama_stub):
    settings = Settings(runs_dir=tmp_path / "runs")
    traces = tracing(settings, "t", "`run`")
    assert [p.name for p in traces.paths] == ["trace.md", "trace_prompts.md", "trace_debug.md"]
    assert run(write_fixture(tmp_path, 0), "t", build_graph(settings), _approve, lambda _: None,
               traces=traces) == 0
    text = (tmp_path / "runs" / "t" / "trace_debug.md").read_text()
    assert text.startswith("# LangChain and LangGraph debug output of thread `t`\n")
    assert "\x1b[" not in text
    # LangChain's console format: the chat messages flattened to one string.
    assert "### [langchain] [llm/start] [chain:LangGraph > chain:plan" in text
    assert '"System: You are a data-cleaning planner.' in text
    # LangGraph's debug events: tasks, their results, and checkpoints.
    assert "### [langgraph] step 3 · task · plan" in text
    assert "### [langgraph] step 3 · task_result · plan" in text
    assert "· checkpoint · next: route" in text
