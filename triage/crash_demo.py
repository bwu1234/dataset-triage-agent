"""Crash-and-resume demo: kill a run while it waits for approval, then finish
it from a new process.

    uv run python -m triage.crash_demo [--seed 0] [--thread <id>]

The first process runs ``triage.cli run`` on a generated fixture and is
killed with SIGKILL (no handlers, no cleanup) as soon as it asks for
approval. A second process runs ``triage.cli resume`` on the same thread and
approves everything. The model is called once, in the first process; the
audit log at the end shows the run was not re-planned.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from triage.cli import describe, safe
from triage.config import Settings
from triage.faults import write_fixture
from triage.graph import build_graph, open_checkpointer, run_config

Write = Callable[[str], None]

APPROVAL_LINE = "Approval needed"


def kill_at_first_approval(argv: Sequence[str], write: Write, timeout_s: float,
                           env: Mapping[str, str] | None = None) -> bool:
    """Start ``argv``, echo its output, and SIGKILL it when it prints an
    approval prompt. Returns False if it ended without asking. The CLI prints
    the prompt only after the interrupt is checkpointed, so the kill cannot
    land before the pending approval is on disk."""
    # As a context manager, Popen closes the pipes and reaps the child on exit.
    with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, env=env) as proc:
        watchdog = threading.Timer(timeout_s, proc.kill)
        watchdog.start()
        try:
            for line in proc.stdout:
                write(safe(f"  | {line.rstrip()}"))
                if line.startswith(APPROVAL_LINE):
                    proc.kill()
                    write(f"killed pid {proc.pid} (SIGKILL) while it waited for approval")
                    return True
            return False
        finally:
            watchdog.cancel()


def run_to_end(argv: Sequence[str], answers: str, write: Write, timeout_s: float,
               env: Mapping[str, str] | None = None) -> int:
    """Run ``argv`` with ``answers`` on stdin, echo its output, return its exit code."""
    done = subprocess.run(argv, input=answers, capture_output=True, text=True, env=env,
                          timeout=timeout_s)
    for line in (done.stdout + done.stderr).splitlines():
        write(safe(f"  | {line}"))
    return done.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage.crash_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    parser.add_argument("--thread", default=f"crash-demo-{time.strftime('%Y%m%d-%H%M%S')}")
    args = parser.parse_args(argv)

    settings = Settings()
    csv = write_fixture(args.fixtures, args.seed)
    cli = [sys.executable, "-u", "-m", "triage.cli"]
    # Loading and one planner call, each bounded by the model timeout.
    timeout_s = 2 * settings.model.timeout_s

    print(f"1. run {csv} as thread {args.thread!r} with {settings.model.name}")
    if not kill_at_first_approval([*cli, "run", str(csv), "--thread", args.thread], print,
                                  timeout_s, env=os.environ):
        print("the run ended without asking for approval, so there was nothing to kill")
        return 1

    print(f"2. resume thread {args.thread!r} in a new process, approving everything")
    # More answers than any plan has ops; the CLI reads one per approval.
    code = run_to_end([*cli, "resume", "--thread", args.thread], "a\n" * 100, print,
                      timeout_s, env=os.environ)

    with open_checkpointer(settings.checkpoint_db) as saver:
        graph = build_graph(settings, planner=RunnableLambda(_never_called), checkpointer=saver)
        audit = graph.get_state(run_config(args.thread)).values.get("audit", [])
    planned = sum(e.action == "planned" for e in audit)
    print(f"3. final audit log ({planned} planner call(s), exit code {code}):")
    for entry in audit:
        print(f"  {describe(entry)}")
    return code if planned == 1 else 1


def _never_called(_: object) -> dict:
    raise AssertionError("reading state must not call the planner")


if __name__ == "__main__":
    sys.exit(main())
