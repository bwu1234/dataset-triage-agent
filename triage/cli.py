"""Command line for the triage graph.

    uv run python -m triage.cli run <csv> --thread <id>
    uv run python -m triage.cli resume --thread <id>

``run`` streams progress and stops at each op that needs approval. Every step
is checkpointed to ``Settings.checkpoint_db``, so a run that was stopped or
killed, at a prompt or mid-node, carries on with ``resume`` in a new process.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from pydantic import TypeAdapter, ValidationError

from triage.config import Settings
from triage.graph import (
    ApprovalDecision,
    ApprovalRequest,
    AuditEntry,
    build_graph,
    check_thread_id,
    open_checkpointer,
    run_config,
)
from triage.impact import Impact
from triage.ops import AnyOp, Op

Read = Callable[[str], str]
Write = Callable[[str], None]

_OP = TypeAdapter(Op)


def safe(text: object) -> str:
    """Column names, values, and model-written reasons all trace back to the
    input file. Escape control characters so that text cannot move the cursor,
    recolour, or rewrite the terminal."""
    return "".join(c if c.isprintable() else repr(c)[1:-1] for c in str(text))


def _target(op: AnyOp) -> str:
    column = getattr(op, "column", None)
    if column is not None:
        return f"{op.op} {column!r}"
    subset = getattr(op, "subset", None)
    return f"{op.op} {subset}" if subset else f"{op.op} (all columns)"


def _impact(impact: Impact) -> str:
    parts = [f"{v} {k.replace('_', ' ')}" for k, v in impact.model_dump().items() if v]
    return ", ".join(parts) or "no change"


def describe(entry: AuditEntry) -> str:
    line = f"{entry.action:<11}"
    if entry.op_index is not None:
        line += f" #{entry.op_index}"
    if entry.op is not None:
        line += f" {_target(entry.op)}"
    if entry.impact is not None:
        line += f": {_impact(entry.impact)}"
    if entry.detail:
        line += f" ({entry.detail})"
    return safe(line)


def ask_approval(request: ApprovalRequest, read: Read = input, write: Write = print) -> ApprovalDecision:
    op = request.op
    write("")
    write(safe(f"Approval needed for #{request.op_index}: {_target(op)}"))
    write(safe(f"  reason: {op.reason}"))
    write(safe(f"  impact: {_impact(request.impact)}"))
    write(safe(f"  op:     {op.model_dump_json()}"))
    while True:
        choice = read("[a]pprove, [r]eject, or [e]dit? ").strip().lower()
        if choice in ("a", "approve"):
            return ApprovalDecision(action="approve")
        if choice in ("r", "reject"):
            return ApprovalDecision(action="reject", note=read("Why (optional): ").strip() or None)
        if choice in ("e", "edit"):
            text = read("Replacement op as JSON (same shape as 'op' above): ")
            try:
                return ApprovalDecision(action="edit", op=_OP.validate_json(text))
            except ValidationError as e:
                write(safe(f"Not a valid op: {e.errors(include_url=False)}"))


Ask = Callable[[ApprovalRequest], ApprovalDecision]


def _answer(state: StateSnapshot, ask: Ask) -> Command:
    # The resume value goes through the checkpointer, so send plain JSON;
    # ``approve`` validates it back into an ApprovalDecision.
    return Command(resume=ask(state.interrupts[0].value).model_dump(mode="json"))


def run(csv: Path, thread: str, graph: CompiledStateGraph, ask: Ask, write: Write = print,
        banner: str | None = None) -> int:
    config = run_config(thread)
    if graph.get_state(config).values:
        # New input on an existing thread would start over from ``load`` and
        # silently drop any pending approval, so make that a separate choice.
        write(safe(f"thread {thread!r} already exists; continue it with "
                   f"'resume --thread {thread}' or pick a new id"))
        return 2
    if banner:
        write(safe(banner))
    return _drive(graph, config, {"input_path": str(csv)}, ask, write)


def resume(thread: str, graph: CompiledStateGraph, ask: Ask, write: Write = print) -> int:
    """Continue a thread from its last checkpoint: answer the pending approval,
    or re-run the node that was in progress when the process stopped."""
    config = run_config(thread)
    state = graph.get_state(config)
    if not state.values:
        write(safe(f"no run with thread {thread!r}"))
        return 2
    for entry in state.values.get("audit", []):
        write(describe(entry))
    if not state.next:
        return _report(state.values, write)
    write(safe(f"resuming at {', '.join(state.next)}"))
    # ``None`` input continues from the checkpoint without new state.
    return _drive(graph, config, _answer(state, ask) if state.interrupts else None, ask, write)


def _drive(graph: CompiledStateGraph, config, inp: dict | Command | None, ask: Ask,
           write: Write) -> int:
    while True:
        for chunk in graph.stream(inp, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__" or not update:
                    continue
                for entry in update.get("audit", []):
                    write(describe(entry))
        state = graph.get_state(config)
        if not state.interrupts:
            return _report(state.values, write)
        inp = _answer(state, ask)


def _report(values: dict, write: Write) -> int:
    if values.get("status") == "done":
        write(safe(f"done: {values['output_path']}"))
        return 0
    write(safe(f"failed: {values.get('last_error')}"))
    return 1


def _thread_arg(value: str) -> str:
    try:
        return check_thread_id(value)
    except ValueError as e:  # argparse shows only ArgumentTypeError messages.
        raise argparse.ArgumentTypeError(str(e)) from e


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="triage a CSV, asking before destructive ops")
    run_p.add_argument("csv", type=Path)
    run_p.add_argument("--thread", required=True, type=_thread_arg,
                       help="new run id; output goes to runs/<thread>/")
    resume_p = sub.add_parser("resume", help="continue a stopped or killed run")
    resume_p.add_argument("--thread", required=True, type=_thread_arg)
    args = parser.parse_args(argv)

    if args.command == "run" and not args.csv.is_file():
        parser.error(f"no such file: {args.csv}")
    settings = Settings()
    try:
        with open_checkpointer(settings.checkpoint_db) as saver:
            graph = build_graph(settings, checkpointer=saver)
            if args.command == "resume":
                return resume(args.thread, graph, ask_approval)
            return run(args.csv, args.thread, graph, ask_approval,
                       banner=f"planning with {settings.model.name}; this can take a minute")
    except (KeyboardInterrupt, EOFError):
        print()
        print(safe(f"stopped. Continue with: uv run python -m triage.cli resume "
                   f"--thread {args.thread}"))
        return 130


if __name__ == "__main__":
    sys.exit(main())
