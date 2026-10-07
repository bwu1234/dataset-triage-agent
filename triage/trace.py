"""A Markdown trace of a graph run: each node's state update, each approval
pause and resume, and each model call in full.

``MarkdownTrace`` is a LangChain callback handler. Passed in the run config's
``callbacks``, LangGraph hands it to every node and to the chat model inside
``plan``. For a model call it writes the messages, the prompt text Ollama
renders from them, how that prompt differs from the previous call's, and the
reply: thinking, content, tool calls, token counts and timings.

The rendered prompt comes from Ollama, not from a template copied here.
qwen3.x models carry ``TEMPLATE {{ .Prompt }}`` and a built-in Go
``RENDERER`` (see ``ollama show --modelfile``), so the chat template lives in
Ollama's code. The trace rebuilds the request ChatOllama sent, from the same
``chat_model`` config and the same messages and ``format``, and posts it with
``_debug_render_only``, which returns the prompt without generating. That
field is undocumented (checked on Ollama 0.35.1); if a render fails, the
trace says why and the run carries on.

The file is appended to: ``run``, ``resume`` and ``fork`` of one thread each
add a section. Prompt deltas compare calls within one process, so the first
call after a ``resume`` has no delta.

The rendered prompts go to a file of their own (``prompts_file_name``), one
section per model call with its delta and reply, so they read in sequence
without the graph steps in between; the main trace points to them by call
number.

``Tracing`` can also write the libraries' own debug output to a second file,
for comparison: LangChain's ``FunctionCallbackHandler`` (the handler
``set_debug(True)`` attaches, printing to the console) pointed at the file,
and LangGraph's ``stream_mode="debug"`` events. LangGraph's ``debug=True``
only prints ``updates`` and ``values`` to stdout, so the debug stream is the
fuller record that can go to a file.
"""

import difflib
import json
import re
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.tracers.stdout import FunctionCallbackHandler
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel

from triage.config import ModelConfig, Settings, TraceConfig
from triage.planner import chat_model


class Tracing(NamedTuple):
    """What a run passes to the graph: ``callbacks`` go in the run config,
    and ``graph_events``, when set, takes each ``stream_mode="debug"`` event."""

    callbacks: list[BaseCallbackHandler] = []
    graph_events: Callable[[dict[str, Any]], None] | None = None
    paths: list[Path] = []


NO_TRACING = Tracing()


def tracing(settings: Settings, thread: str, title: str) -> Tracing:
    """Traces for one CLI or evaluation run of ``thread``, per
    ``settings.trace``. ``thread`` must already have passed ``check_thread_id``."""
    config, run_dir = settings.trace, settings.runs_dir / thread
    result = Tracing([], None, [])
    if config.enabled:
        trace = MarkdownTrace(
            _MarkdownFile(run_dir / config.file_name, title, thread, settings.model),
            _MarkdownFile(run_dir / config.prompts_file_name, title, thread, settings.model,
                          kind="Rendered prompts"),
            settings.model, config)
        result.callbacks.append(trace)
        result.paths.extend([trace.out.path, trace.prompts.path])
    if config.debug_enabled:
        debug = LibraryDebug(_MarkdownFile(run_dir / config.debug_file_name, title, thread,
                                           settings.model, kind="LangChain and LangGraph debug output"))
        result.callbacks.append(FunctionCallbackHandler(debug.langchain))
        result = result._replace(graph_events=debug.langgraph)
        result.paths.append(debug.out.path)
    return result


def render_prompt(base_url: str, request: dict[str, Any], timeout_s: float) -> str:
    """The prompt text Ollama would feed the model for ``request``, a
    ``/api/chat`` body. Renders only; nothing is generated."""
    body = json.dumps({**request, "stream": False, "_debug_render_only": True}).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        reply = json.load(resp)
    rendered = (reply.get("_debug_info") or {}).get("rendered_template")
    if not isinstance(rendered, str):
        raise ValueError(f"no _debug_info.rendered_template in Ollama's reply: {sorted(reply)}")
    return rendered


class MarkdownTrace(BaseCallbackHandler):
    # Errors in a callback are logged by LangChain and never fail the run.
    raise_error = False

    def __init__(self, out: "_MarkdownFile", prompts: "_MarkdownFile", model: ModelConfig,
                 config: TraceConfig) -> None:
        self.out = out
        self.prompts = prompts
        self.model = model
        self.config = config
        self._write = out.write
        self._llm = chat_model(model)
        self._nodes: dict[UUID, tuple[str, float]] = {}
        # Start time and call number of each model call in flight.
        self._calls: dict[UUID, tuple[float, int]] = {}
        self._n_calls = 0
        self._prev_messages: list[dict[str, Any]] | None = None
        self._prev_prompt: str | None = None

    # Graph steps

    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: Any, *, run_id: UUID,
                       parent_run_id: UUID | None = None, tags: list[str] | None = None,
                       metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        name = kwargs.get("name")
        if parent_run_id is None:
            resume = getattr(inputs, "resume", None)
            if resume is not None:
                self._write(f"**Resumed with** the person's answer:\n\n{_json_block(resume)}")
            elif isinstance(inputs, dict) and inputs:
                self._write(f"**Input:**\n\n{_json_block(inputs)}")
            return
        node = (metadata or {}).get("langgraph_node")
        if node is None or name != node or not any(t.startswith("graph:step:") for t in tags or []):
            return
        self._nodes[run_id] = (node, time.monotonic())
        self._write(f"### Step {(metadata or {}).get('langgraph_step')} · `{node}`")

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if run_id not in self._nodes:
            return
        _, start = self._nodes.pop(run_id)
        keys = ", ".join(f"`{k}`" for k in outputs) if isinstance(outputs, dict) else "-"
        self._write(_details(f"state update ({keys}) · {time.monotonic() - start:.2f} s",
                             _json_block(outputs)))

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if run_id not in self._nodes:
            return
        node, _ = self._nodes.pop(run_id)
        if isinstance(error, GraphInterrupt):
            values = [getattr(i, "value", i) for i in (error.args[0] if error.args else ())]
            self._write("**Paused for approval.** `interrupt()` raised and the run stops "
                        f"here until a person answers; on resume LangGraph re-runs `{node}` "
                        "from the top.\n\n" + _json_block(values[0] if len(values) == 1 else values))
        else:
            self._write(f"**Error in `{node}`:** {type(error).__name__}: {error}")

    # Model calls

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]],
                            *, run_id: UUID, invocation_params: dict[str, Any] | None = None,
                            metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._n_calls += 1
        n = self._n_calls
        self._calls[run_id] = (time.monotonic(), n)
        meta = metadata or {}
        params = invocation_params or {}
        # Only the per-call arguments the planner binds; the rest of the
        # request comes from the model config, as in ChatOllama itself.
        extra = {k: params[k] for k in ("format", "tools") if params.get(k) is not None}
        request = self._llm._chat_params(messages[0], params.get("stop"), **extra)
        sent = request["messages"]
        shown = {k: v for k, v in request.items() if k not in ("messages", "format", "tools")}

        parts = [f"#### Model call {n} · `{request['model']}`",
                 "**Request:** the `/api/chat` body ChatOllama sends, less `messages` "
                 "and `format` (below).\n\n" + _json_block(shown),
                 f"**Messages** ({len(sent)}), compared with "
                 + (f"call {n - 1}:" if self._prev_messages is not None
                    else "nothing; first call in this process:"),
                 _message_table(sent, self._prev_messages)]
        where = f"step {meta.get('langgraph_step')} · `{meta.get('langgraph_node')}`"
        rendered = [f"### Call {n} · {where} · `{request['model']}`"]
        prompt = None
        if self.config.render_prompt:
            try:
                prompt = render_prompt(self.model.base_url, request, self.config.render_timeout_s)
            except Exception as exc:  # The trace is a side channel; never fail the run.
                rendered.append(f"Not available ({type(exc).__name__}: {exc}).")
        else:
            rendered.append("Not rendered (`render_prompt` is off).")
        if prompt is not None:
            rendered.append(f"**Prompt** ({len(prompt)} chars): the exact text Ollama feeds "
                            "the model, from `_debug_render_only`. The `format` schema is not "
                            "in it; Ollama passes that to the model runner separately.\n\n"
                            + _block(prompt, "text"))
            if self._prev_prompt is not None:
                diff = "\n".join(difflib.unified_diff(
                    self._prev_prompt.splitlines(), prompt.splitlines(),
                    f"call {n - 1}", f"call {n}", n=1, lineterm=""))
                rendered.append(f"**Delta from call {n - 1}:**\n\n"
                                + (_block(diff, "diff") if diff else "identical."))
            self._prev_prompt = prompt
        self.prompts.write("\n\n".join(rendered))
        parts.append(f"**Rendered prompt:** call {n} in `{self.prompts.path.name}`"
                     + (f" ({len(prompt)} chars)." if prompt is not None else ", not available."))
        if "format" in extra:
            parts.append(_details("format: the JSON schema the reply must match",
                                  _json_block(extra["format"])))
        if "tools" in extra:
            parts.append(_details("tools offered to the model", _json_block(extra["tools"])))
        self._prev_messages = sent
        self._write("\n\n".join(parts))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        start, n = self._calls.pop(run_id, (None, None))
        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        message = getattr(gen, "message", None)
        if message is None:
            return
        thinking = message.additional_kwargs.get("reasoning_content")
        content = message.content if isinstance(message.content, str) else json.dumps(message.content)
        meta = message.response_metadata or {}
        parts = ["**Thinking:** " + (
            f"{len(thinking)} chars.\n\n{_block(thinking, 'text')}" if thinking
            else f"none returned (`think={self.model.think}`).")]
        parts.append(f"**Output** ({len(content)} chars, as returned):\n\n{_block(content, 'json')}")
        tool_calls = getattr(message, "tool_calls", None)
        parts.append("**Tool calls:** " + (
            f"\n\n{_json_block(tool_calls)}" if tool_calls
            else "none. The planner asks for JSON through `format`, not through tools."))
        parts.append("**Tokens and timing:** " + _usage(meta, start))
        self._write("\n\n".join(parts))
        # The reply also follows its prompt, as Ollama's parser split it.
        self.prompts.write(f"**Reply to call {n}**, split by Ollama into thinking and "
                           "content:\n\n" + "\n\n".join(p for p in parts
                                                       if not p.startswith("**Tool calls:**")))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        start, n = self._calls.pop(run_id, (None, None))
        took = f" after {time.monotonic() - start:.1f} s" if start is not None else ""
        failed = f"**Model call {n} failed{took}:** {type(error).__name__}: {error}"
        self._write(failed)
        self.prompts.write(failed)


class LibraryDebug:
    """The libraries' own debug output, in the order it happens, each entry
    headed by its source."""

    def __init__(self, out: "_MarkdownFile") -> None:
        self.out = out

    def langchain(self, text: str) -> None:
        """One message from ``FunctionCallbackHandler``: a header line, then
        the run's input or output as JSON."""
        head, _, body = _ANSI.sub("", text).partition("\n")
        self.out.write(f"### [langchain] {head.rstrip(':')}"
                       + (f"\n\n{_block(body.strip(), 'json')}" if body.strip() else ""))

    def langgraph(self, event: dict[str, Any]) -> None:
        """One ``stream_mode="debug"`` event: a checkpoint saved, a task
        started, or a task's result."""
        payload = event.get("payload") or {}
        what = payload.get("name") if event.get("type") != "checkpoint" else (
            "next: " + ", ".join(payload.get("next") or []) or "-")
        self.out.write(f"### [langgraph] step {event.get('step')} · {event.get('type')} · {what}"
                       f"\n\n{_json_block(event)}")


class _MarkdownFile:
    """An append-only Markdown file with one section per command."""

    def __init__(self, path: Path, title: str, thread: str, model: ModelConfig,
                 kind: str = "Trace") -> None:
        self.path = path
        self._header = (f"# {kind} of thread `{thread}`\n",
                        f"\n## {title} · {{time}}\n\nModel `{model.name}`, think={model.think}.\n")
        self._lock = threading.Lock()
        self._opened = False

    def write(self, text: str) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                if not self._opened:
                    # Written on the first event, so a command that stops
                    # before the graph runs leaves no empty section.
                    if f.tell() == 0:
                        f.write(self._header[0])
                    f.write(self._header[1].format(time=time.strftime("%Y-%m-%d %H:%M:%S")))
                    self._opened = True
                f.write(f"\n{text}\n")


# The colour codes ``FunctionCallbackHandler`` puts around its headers.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _usage(meta: dict[str, Any], start: float | None) -> str:
    """Ollama's counts. ``eval_count`` covers thinking and answer together;
    Ollama does not report them separately."""
    def secs(key: str) -> float | None:
        ns = meta.get(key)
        return ns / 1e9 if isinstance(ns, int | float) else None

    items = [f"prompt {meta.get('prompt_eval_count', '?')} tokens",
             f"output {meta.get('eval_count', '?')} tokens (thinking and answer together)",
             f"done_reason `{meta.get('done_reason', '?')}`"]
    if (p := secs("prompt_eval_duration")) is not None:
        items.append(f"prompt eval {p:.1f} s")
    if (g := secs("eval_duration")) is not None:
        rate = f", {meta['eval_count'] / g:.1f} tok/s" if g and meta.get("eval_count") else ""
        items.append(f"generation {g:.1f} s{rate}")
    if (load := secs("load_duration")) is not None:
        items.append(f"model load {load:.1f} s")
    if start is not None:
        items.append(f"wall {time.monotonic() - start:.1f} s")
    return "; ".join(items) + "."


def _message_table(sent: Sequence[dict[str, Any]], prev: Sequence[dict[str, Any]] | None) -> str:
    rows = ["| # | role | chars | change | starts with |", "|---|---|---|---|---|"]
    for i, m in enumerate(sent):
        if prev is None:
            change = "-"
        elif i >= len(prev):
            change = "appended"
        elif m == prev[i]:
            change = "same"
        else:
            change = "changed"
        content = str(m.get("content", ""))
        head = content.strip().splitlines()[0][:60] if content.strip() else ""
        head = head.replace("|", "\\|").replace("`", "'")
        rows.append(f"| {i} | {m.get('role')} | {len(content)} | {change} | `{head}` |")
    if prev is not None and len(prev) > len(sent):
        rows.append(f"\n{len(prev) - len(sent)} message(s) from the previous call are gone.")
    return "\n".join(rows)


def _details(summary: str, body: str) -> str:
    # The blank line after <summary> lets Markdown render the fenced block.
    return f"<details><summary>{summary}</summary>\n\n{body}\n\n</details>"


def _json_block(value: Any) -> str:
    return _block(json.dumps(value, indent=1, default=_plain, ensure_ascii=False), "json")


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


def _block(text: str, lang: str = "") -> str:
    """A fenced block longer than any backtick run in ``text``, which comes
    from the file and the model and so may contain fences of its own."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text}\n{fence}"
