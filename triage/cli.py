"""Command line for the triage graph.

    uv run python -m triage.cli run <csv> --thread <id>
    uv run python -m triage.cli resume --thread <id>
    uv run python -m triage.cli history --thread <id>
    uv run python -m triage.cli fork --thread <id> --checkpoint <checkpoint id>

``run`` streams progress and stops at each op that needs approval. Every step
is checkpointed to ``Settings.checkpoint_db``, so a run that was stopped or
killed, at a prompt or mid-node, carries on with ``resume`` in a new process.
``history`` lists those checkpoints and ``fork`` runs the thread again from
one of them, say to answer an approval differently. The fork is a new branch
of the same thread; ``resume`` and ``history`` follow the newest branch.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from langchain_core.runnables import RunnableConfig
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


def describe(entry: AuditEntry, width: int = 11) -> str:
    line = f"{entry.action:<{width}}"
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


def history(thread: str, graph: CompiledStateGraph, write: Write = print) -> int:
    """List a thread's checkpoints, newest first, with what each step added.
    After a fork the thread is a tree; ``*`` marks the newest checkpoint and
    its ancestors, the branch ``resume`` continues."""
    snapshots = list(graph.get_state_history(run_config(thread)))
    if not snapshots:
        write(safe(f"no run with thread {thread!r}"))
        return 2
    by_id = {_checkpoint_id(s.config): s for s in snapshots}
    current, node = set(), snapshots[0]
    while node is not None:
        current.add(_checkpoint_id(node.config))
        node = by_id.get(_checkpoint_id(node.parent_config)) if node.parent_config else None
    write(safe(f"thread {thread!r}: {len(snapshots)} checkpoints, newest first; "
               f"* marks the branch 'resume' continues"))
    write(f"  {'step':>4}  {'checkpoint':<36}  {'next':<9}  event")
    for s in snapshots:
        cid = _checkpoint_id(s.config)
        parent = by_id.get(_checkpoint_id(s.parent_config)) if s.parent_config else None
        write(safe(f"{'*' if cid in current else ' '} {s.metadata['step']:>4}  {cid:<36}  "
                   f"{', '.join(s.next) or '-':<9}  {_event(s, parent)}".rstrip()))
    return 0


def _checkpoint_id(config) -> str:
    return config["configurable"]["checkpoint_id"]


def _event(s: StateSnapshot, parent: StateSnapshot | None) -> str:
    """One line on what a checkpoint holds: the audit entries its step added,
    the approval it asked for, or the checkpoint it forked from."""
    parts = []
    if s.metadata.get("source") == "fork" and s.parent_config:
        parts.append(f"fork of {_checkpoint_id(s.parent_config)}")
    else:
        before = len(parent.values.get("audit", [])) if parent else 0
        parts += [describe(e, width=0) for e in s.values.get("audit", [])[before:]]
    for intr in s.interrupts:
        req = intr.value
        parts.append(f"asks: #{req.op_index} {_target(req.op)}: {_impact(req.impact)}")
    return "; ".join(parts)


def fork(thread: str, checkpoint: str, graph: CompiledStateGraph, ask: Ask,
         write: Write = print) -> int:
    """Run a thread again from an earlier checkpoint, as a new branch. The old
    branch stays in ``history``; the fork's checkpoints are the newest, so
    ``resume`` continues the fork if it is stopped."""
    config = run_config(thread)
    at: RunnableConfig = {"configurable": {**config["configurable"], "checkpoint_id": checkpoint}}
    state = graph.get_state(at)
    if state.metadata is None:
        write(safe(f"no checkpoint {checkpoint!r} in thread {thread!r}; see 'history --thread {thread}'"))
        return 2
    if not state.next:
        # Streaming from here would still write an empty fork checkpoint,
        # which would then shadow the run's real end for ``resume``.
        write(safe(f"checkpoint {checkpoint!r} is the end of a run; fork an earlier one"))
        return 2
    for entry in state.values.get("audit", []):
        write(describe(entry))
    write(safe(f"forking at {', '.join(state.next)}"
               + ("; this asks the model for a new plan" if "plan" in state.next else "")))
    # Stream ``None`` at the old checkpoint, never ``Command(resume=...)``.
    # LangGraph 1.2 treats ``None`` there as time travel: it saves a fork
    # checkpoint and drops the old answer, so a pending approval is asked
    # again. A resume command at the same checkpoint keeps the old answer's
    # writes and ignores the new one (checked in tests/test_replay.py).
    return _drive(graph, config, None, ask, write, start=at)


def _drive(graph: CompiledStateGraph, config, inp: dict | Command | None, ask: Ask,
           write: Write, start: RunnableConfig | None = None) -> int:
    """Stream until the run ends, asking at each approval. ``start`` (a fork's
    checkpoint) applies to the first stream only; after that the thread's
    newest checkpoint is the one to continue."""
    while True:
        for chunk in graph.stream(inp, start or config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__" or not update:
                    continue
                for entry in update.get("audit", []):
                    write(describe(entry))
        start = None
        state = graph.get_state(config)
        if not state.interrupts:
            return _report(state.values, write)
        inp = _answer(state, ask)


def _report(values: dict, write: Write) -> int:
    if values.get("status") == "done":
        write(safe(f"done: {values['output_path']}"))
        return 0
    # Validation failures are one per line; escape each line, not the breaks.
    for i, line in enumerate(str(values.get("last_error")).splitlines() or [""]):
        write(safe(f"{'failed:' if i == 0 else '       '} {line}"))
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
    history_p = sub.add_parser("history", help="list a run's checkpoints")
    history_p.add_argument("--thread", required=True, type=_thread_arg)
    fork_p = sub.add_parser("fork", help="run again from an earlier checkpoint, as a new branch")
    fork_p.add_argument("--thread", required=True, type=_thread_arg)
    fork_p.add_argument("--checkpoint", required=True, help="a checkpoint id from 'history'")
    args = parser.parse_args(argv)

    if args.command == "run" and not args.csv.is_file():
        parser.error(f"no such file: {args.csv}")
    settings = Settings()
    try:
        with open_checkpointer(settings.checkpoint_db) as saver:
            graph = build_graph(settings, checkpointer=saver)
            if args.command == "history":
                return history(args.thread, graph)
            if args.command == "resume":
                return resume(args.thread, graph, ask_approval)
            if args.command == "fork":
                return fork(args.thread, args.checkpoint, graph, ask_approval)
            return run(args.csv, args.thread, graph, ask_approval,
                       banner=f"planning with {settings.model.name}; this can take a minute")
    except (KeyboardInterrupt, EOFError):
        print()
        print(safe(f"stopped. Continue with: uv run python -m triage.cli resume "
                   f"--thread {args.thread}"))
        return 130


if __name__ == "__main__":
    sys.exit(main())
