"""Generated parts of the README (M8).

    uv run python -m triage.writeup mermaid
    uv run python -m triage.writeup demo --seed 0 --out docs/demo.md

``mermaid`` prints ``graph.get_graph().draw_mermaid()``; the README holds a
copy, and ``tests/test_writeup.py`` fails when they drift apart.

``demo`` records a real ``triage.cli run`` on one fixture with the configured
model. The run goes through the CLI's own ``run`` and ``ask_approval``; only
the keyboard is replaced, by a scripted person (``scripted_answer``). The
transcript shows each prompt with the answer typed after it.
"""

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.runnables import Runnable, RunnableLambda

from triage.cli import ask_approval, run, safe
from triage.config import Settings
from triage.faults import write_fixture
from triage.graph import (
    ApprovalDecision,
    ApprovalRequest,
    build_graph,
    open_checkpointer,
)
from triage.ops import CastType, FilterRows


def mermaid(settings: Settings) -> str:
    # The diagram depends on nodes and edges only, so no model is needed.
    graph = build_graph(settings, planner=RunnableLambda(lambda _: None))
    return graph.get_graph().draw_mermaid()


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
    demo_p = sub.add_parser("demo", help="record a CLI run with a scripted person")
    demo_p.add_argument("--seed", type=int, default=0)
    demo_p.add_argument("--out", type=Path, default=Path("docs/demo.md"))
    args = parser.parse_args(argv)

    settings = Settings()
    if args.command == "mermaid":
        print(mermaid(settings))
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
        f"```text\n{transcript}\n```\n")
    print(transcript)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
