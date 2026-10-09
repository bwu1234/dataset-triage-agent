"""Generated parts of the README (M8).

    uv run python -m triage.writeup mermaid
    uv run python -m triage.writeup path --thread <id>
    uv run python -m triage.writeup demo --seed 0 --out docs/demo.md

``mermaid`` prints ``graph.get_graph().draw_mermaid()``; the README holds a
copy, and ``tests/test_writeup.py`` fails when they drift apart.

``path`` draws one run instead of the graph: a Mermaid flowchart of the
steps a thread took, read from its checkpoints (``get_state_history``) on the
branch ``resume`` continues. Each box is a node that ran, with the audit
entries it added; ``route`` steps that only decided become edge labels.

``demo`` records a real ``triage.cli run`` on one fixture with the configured
model. The run goes through the CLI's own ``run`` and ``ask_approval``; only
the keyboard is replaced, by a scripted person (``scripted_answer``). The
transcript shows each prompt with the answer typed after it.
"""

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph.state import CompiledStateGraph

from triage.cli import (
    ask_approval,
    current_branch,
    describe,
    impact_text,
    op_target,
    run,
    safe,
)
from triage.config import Settings
from triage.faults import write_fixture
from triage.graph import (
    ApprovalDecision,
    ApprovalRequest,
    AuditEntry,
    build_graph,
    open_checkpointer,
    run_config,
)
from triage.ops import CastType, FilterRows


def mermaid(settings: Settings) -> str:
    # The diagram depends on nodes and edges only, so no model is needed.
    graph = build_graph(settings, planner=RunnableLambda(lambda _: None))
    return graph.get_graph().draw_mermaid()


# What a ``route`` step that added no audit entry decided, as an edge label.
_ROUTED = {"apply": "within limits", "approve": "over limits", "done": "all ops routed"}

# Box styles by what happened in the step; the first matching action wins.
_CLASSES = [("rejected", "rejected"), ("edited", "edited"), ("approved", "approved"),
            ("skipped", "problem"), ("plan_failed", "problem"), ("invalid", "problem")]
_CLASS_DEFS = {
    "approved": "fill:#dcfce7,stroke:#15803d,color:#14532d",
    "rejected": "fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d",
    "edited": "fill:#fef3c7,stroke:#b45309,color:#78350f",
    "problem": "fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d,stroke-dasharray:4 3",
    "pending": "fill:#e0e7ff,stroke:#4338ca,color:#312e81,stroke-dasharray:4 3",
}


def run_path(graph: CompiledStateGraph, thread: str, width: int = 90) -> str | None:
    """A Mermaid flowchart of the steps ``thread`` took, oldest first, or None
    if there is no such thread. Lines in a box are cut at ``width``."""
    snapshots = list(graph.get_state_history(run_config(thread)))
    if not snapshots:
        return None
    branch = current_branch(snapshots)[::-1]
    out = ["flowchart TD", '  start(["start"])']
    used: set[str] = set()
    prev, label = "start", ""

    def box(name: str, lines: list[str], cls: str | None) -> None:
        nonlocal prev, label
        node = f"s{sum(line.startswith('  s') for line in out) // 2 + 1}"
        text = "<br/>".join([f"<b>{_mermaid_text(name, width)}</b>",
                             *(_mermaid_text(line, width) for line in lines)])
        out.append(f'  {node}["{text}"]' + (f":::{cls}" if cls else ""))
        out.append(f"  {prev} -->{f'|{label}|' if label else ''} {node}")
        used.update([cls] if cls else [])
        prev, label = node, ""

    for before, after in pairwise(branch):
        if after.metadata.get("source") == "fork":
            box("fork", [f"from checkpoint {after.parent_config['configurable']['checkpoint_id']}"],
                None)
            continue
        if not before.next or before.next == ("__start__",):
            continue
        node = ", ".join(before.next)
        added: list[AuditEntry] = after.values.get("audit", [])[len(before.values.get("audit", [])):]
        if node == "route" and not added:
            label = _ROUTED.get(after.values.get("decision"), "")
            continue
        lines = [_entry_line(e) for e in added]
        if node == "approve" and before.interrupts:
            request: ApprovalRequest = before.interrupts[0].value
            lines = [_asked(request)]
            lines += [_edit_line(request, e) if e.action == "edited"
                      else e.action + (f": {e.detail}" if e.detail else "") for e in added]
        actions = {e.action for e in added}
        if "finished" in actions and after.values.get("status") == "failed":
            actions.add("invalid")
        box(node, lines, next((cls for action, cls in _CLASSES if action in actions), None))

    last = branch[-1]
    if last.next:
        lines = [_asked(i.value) for i in last.interrupts if isinstance(i.value, ApprovalRequest)]
        box(f"stopped before {', '.join(last.next)}", lines, "pending")
    else:
        out.append(f'  done(["end: {_mermaid_text(last.values.get("status", ""), width)}"])')
        out.append(f"  {prev} --> done")
    out += [f"  classDef {cls} {_CLASS_DEFS[cls]}" for cls in _CLASS_DEFS if cls in used]
    return "\n".join(out)


# Actions that say no more than the node that adds them; their box shows
# only what follows the action.
_IMPLIED = {"loaded", "planned", "applied", "validated", "finished"}


def stored_path(settings: Settings, thread: str) -> str | None:
    """``run_path`` for a thread in ``settings.checkpoint_db``."""
    with open_checkpointer(settings.checkpoint_db) as saver:
        graph = build_graph(settings, planner=RunnableLambda(lambda _: None), checkpointer=saver)
        return run_path(graph, thread)


def path_section(thread: str, diagram: str) -> str:
    """The demo page's diagram of the recorded run."""
    return ("\n## Path through the graph\n\n"
            f"Drawn from the run's checkpoints with `uv run python -m triage.writeup path "
            f"--thread {thread}`. Each box is a node that ran, with what it added to the "
            "audit log; edge labels are what `route` decided. Green is an approved op, red a "
            "rejected one, amber an edited one.\n\n"
            f"```mermaid\n{diagram}\n```\n")


def _entry_line(entry: AuditEntry) -> str:
    if entry.action == "finished" and entry.detail and not entry.detail.startswith("failed"):
        # The output path; its directory is the thread's run directory.
        return Path(entry.detail).name
    if entry.action not in _IMPLIED:
        return describe(entry, width=0)
    parts = [f"#{entry.op_index} {op_target(entry.op)}" if entry.op else "",
             f": {impact_text(entry.impact)}" if entry.impact else "", entry.detail or ""]
    return safe("".join(parts)) or "passed"


def _asked(request: ApprovalRequest) -> str:
    return safe(f"#{request.op_index} {op_target(request.op)}: {impact_text(request.impact)}")


def _edit_line(request: ApprovalRequest, entry: AuditEntry) -> str:
    """What the person changed, field by field, leaving out ``reason``."""
    old = request.op.model_dump(exclude={"reason"})
    new = entry.op.model_dump(exclude={"reason"}) if entry.op else {}
    changes = [f"{k}: {old.get(k)!r} → {new.get(k)!r}"
               for k in dict.fromkeys([*old, *new]) if old.get(k) != new.get(k)]
    return "edited: " + ("; ".join(changes) or "reason only")


def _mermaid_text(text: object, width: int) -> str:
    """Untrusted text (column names, values, notes) made inert inside a quoted
    Mermaid label: ``#`` first, since the others become ``#...;`` entities."""
    line = safe(text)
    line = line if len(line) <= width else line[:width - 1] + "…"
    for char, entity in [("#", "#35;"), ('"', "#quot;"), ("<", "#lt;"), (">", "#gt;"),
                         ("&", "#amp;"), ("`", "#96;"), ("|", "#124;")]:
        line = line.replace(char, entity)
    return line


def scripted_answer(request: ApprovalRequest) -> list[str]:
    """The demo's stand-in person, as the lines they would type:

    - a cast that turns values into nulls and names a ``datetime_format``:
      edit it to drop the format, so mixed formats parse (``route`` then
      measures the edited op);
    - ``filter_rows`` with ``not_null``: reject, since a missing value is not a
      reason to delete the row;
    - anything else: approve.
    """
    op = request.op
    if isinstance(op, CastType) and op.datetime_format and request.impact.values_nulled:
        edited = op.model_copy(update={
            "datetime_format": None,
            "reason": f"{op.reason} Edited: no fixed format, so mixed formats parse."})
        return ["e", edited.model_dump_json()]
    if isinstance(op, FilterRows) and op.operator == "not_null":
        return ["r", "keep the rows; a missing value is not a reason to delete them"]
    return ["a"]


def scripted_ask(write: Callable[[str], None]) -> Callable[[ApprovalRequest], ApprovalDecision]:
    """``ask_approval`` with each answer typed by ``scripted_answer`` and
    echoed after its prompt, as a terminal would show it."""
    def ask(request: ApprovalRequest) -> ApprovalDecision:
        answers = iter(scripted_answer(request))

        def read(prompt: str) -> str:
            answer = next(answers, "")
            write(safe(prompt + answer))
            return answer
        return ask_approval(request, read=read, write=write)
    return ask


def record(settings: Settings, seed: int, thread: str, fixtures: Path = Path("fixtures"),
           planner: Runnable | None = None) -> tuple[list[str], int]:
    """Run the CLI on fixture ``seed``; return the transcript and exit code."""
    lines: list[str] = []
    csv = write_fixture(fixtures, seed)
    lines.append(f"$ uv run python -m triage.cli run {csv} --thread {thread}")
    with open_checkpointer(settings.checkpoint_db) as saver:
        graph = build_graph(settings, planner=planner, checkpointer=saver)
        code = run(csv, thread, graph, scripted_ask(lines.append), write=lines.append,
                   banner=f"planning with {settings.model.name}; this can take a minute")
    return lines, code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage.writeup")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mermaid", help="print the graph as Mermaid")
    path_p = sub.add_parser("path", help="print the steps of one run as Mermaid")
    path_p.add_argument("--thread", required=True)
    demo_p = sub.add_parser("demo", help="record a CLI run with a scripted person")
    demo_p.add_argument("--seed", type=int, default=0)
    demo_p.add_argument("--out", type=Path, default=Path("docs/demo.md"))
    args = parser.parse_args(argv)

    settings = Settings()
    if args.command == "mermaid":
        print(mermaid(settings))
        return 0
    if args.command == "path":
        diagram = stored_path(settings, args.thread)
        if diagram is None:
            print(safe(f"no run with thread {args.thread!r}"), file=sys.stderr)
            return 2
        print(diagram)
        return 0
    stamp = datetime.now(UTC)
    thread = f"demo-s{args.seed}-{stamp:%Y%m%dT%H%M%SZ}"
    lines, code = record(settings, args.seed, thread)
    transcript = "\n".join(lines)
    args.out.write_text(
        "# Demo run\n\n"
        f"Recorded {stamp:%Y-%m-%d} with `uv run python -m triage.writeup demo --seed "
        f"{args.seed}`: model `{settings.model.name}`, fixture seed {args.seed}, exit code "
        f"{code}. The CLI's prompts and output are real; the answers are typed by a script "
        "(`triage.writeup.scripted_answer`) that edits a date cast which would null values "
        "so it accepts mixed formats, rejects deleting rows for a missing value, and approves "
        "everything else. Model output varies from run to run, so a new recording will "
        "differ.\n\n"
        f"```text\n{transcript}\n```\n"
        f"{path_section(thread, stored_path(settings, thread) or '')}")
    print(transcript)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
