"""Command line for the triage graph.

    uv run python -m triage.cli run <csv> --thread <id>

Streams progress and stops at each op that needs approval. Runs use the
in-memory checkpointer until M4, so a stopped run cannot be resumed later.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import TypeAdapter, ValidationError

from triage.config import Settings
from triage.graph import (
    ApprovalDecision,
    ApprovalRequest,
    AuditEntry,
    build_graph,
    check_thread_id,
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


def run(
    csv: Path,
    thread: str,
    graph: CompiledStateGraph,
    ask: Callable[[ApprovalRequest], ApprovalDecision],
    write: Write = print,
) -> int:
    config = run_config(thread)
    inp: dict | Command = {"input_path": str(csv)}
    while True:
        for chunk in graph.stream(inp, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__" or not update:
                    continue
                for entry in update.get("audit", []):
                    write(describe(entry))
        state = graph.get_state(config)
        if not state.interrupts:
            break
        # The resume value goes through the checkpointer, so send plain JSON;
        # ``approve`` validates it back into an ApprovalDecision.
        inp = Command(resume=ask(state.interrupts[0].value).model_dump(mode="json"))
    values = state.values
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
                       help="run id; output goes to runs/<thread>/")
    args = parser.parse_args(argv)

    if not args.csv.is_file():
        parser.error(f"no such file: {args.csv}")
    settings = Settings()
    print(safe(f"planning with {settings.model.name}; this can take a minute"))
    try:
        return run(args.csv, args.thread, build_graph(settings), ask_approval)
    except (KeyboardInterrupt, EOFError):
        print("\nstopped. Runs are kept in memory until M4, so this one cannot be resumed.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
