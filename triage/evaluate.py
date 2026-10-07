"""M7 evaluation: run the whole graph on synthetic fixtures and score the output.

For each model, approver condition, and fixture seed, runs the graph from
``load`` to ``finish`` with a scripted approver in place of a person, then
scores the final frame with ``triage.faults.CHECKS``. Every run stays in the
denominator: a run that ends ``failed`` or raises counts as zero faults fixed.

    uv run python -m triage.evaluate --models qwen3.5:9b-mlx qwen3.8:27b-mlx \\
        --seeds 0 1 2 3 4 5 6 7 8 9
    uv run python -m triage.evaluate --summarize runs/eval/<stamp>/results.jsonl

Conditions (``docs/evaluation-plan.md``):

- ``approve_all``: every approval is granted, which measures the plans as
  proposed.
- ``rule_based``: rejects an op that removes more than
  ``EvaluationConfig.reject_rows_removed_fraction`` of the rows it sees, drops
  a column with more than one distinct value, or (``reject_null_filters``)
  deletes rows for having a null.

The manifest is the answer key. It is read here, after the run, to score;
it never reaches the planner.
"""

import argparse
import re
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

import pandas as pd
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.types import Command
from pydantic import BaseModel

from triage.cli import describe
from triage.config import EvaluationConfig, Settings
from triage.faults import Manifest, check_all, make_dirty, write_fixture
from triage.graph import ApprovalDecision, ApprovalRequest, build_graph, run_config
from triage.io import load_frame
from triage.ops import DropColumn, FilterRows
from triage.planner import make_planner
from triage.trace import tracing

Condition = Literal["approve_all", "rule_based"]
CONDITIONS: tuple[Condition, ...] = ("approve_all", "rule_based")

# Answers one approval, given the frame the op would run on.
Approver = Callable[[ApprovalRequest, pd.DataFrame], ApprovalDecision]


def approve_all(_request: ApprovalRequest, _frame: pd.DataFrame) -> ApprovalDecision:
    return ApprovalDecision(action="approve")


def rule_based(config: EvaluationConfig) -> Approver:
    def decide(request: ApprovalRequest, frame: pd.DataFrame) -> ApprovalDecision:
        removed = request.impact.rows_removed
        if removed > config.reject_rows_removed_fraction * len(frame):
            return ApprovalDecision(action="reject",
                                    note=f"removes {removed} of {len(frame)} rows")
        op = request.op
        if isinstance(op, DropColumn) and op.column in frame and frame[op.column].nunique() > 1:
            return ApprovalDecision(action="reject", note=f"{op.column!r} is not constant")
        if (config.reject_null_filters and isinstance(op, FilterRows)
                and op.operator == "not_null"):
            return ApprovalDecision(action="reject", note="impute or keep the nulls instead")
        return ApprovalDecision(action="approve")
    return decide


def approver_for(condition: Condition, config: EvaluationConfig) -> Approver:
    return approve_all if condition == "approve_all" else rule_based(config)


class EvalRow(BaseModel):
    model: str
    think: bool | str | None
    condition: Condition
    seed: int
    # ``crashed``: the graph raised. Its error is in ``error``.
    status: Literal["done", "failed", "crashed"]
    error: str | None = None
    # Per fault kind; empty unless status is ``done``.
    faults: dict[str, bool] = {}
    faults_fixed: int = 0
    faults_total: int
    approvals: int = 0
    rejections: int = 0
    # Planner calls after the first (parse or validation failures).
    replans: int = 0
    model_calls: int = 0
    model_seconds: float = 0.0
    wall_seconds: float = 0.0
    rows_removed: int | None = None
    # Clean rows lost: rows that were neither duplicates nor impossible values
    # and are gone from the output. None unless status is ``done``.
    collateral_rows: int | None = None
    # The last plan, and the audit log as ``triage.cli`` prints it.
    plan: list[dict] | None = None
    audit: list[str] = []


def collateral_rows(start: pd.DataFrame, final: pd.DataFrame, manifest: Manifest) -> int:
    """Rows of the loaded frame that should have survived but did not.

    Should be removed: later copies of duplicate rows, and rows with an
    injected impossible value. Rows are compared by content, not index, so a
    ``dedupe(keep='last')`` that keeps the copy instead of the original is not
    counted. The executor preserves the row index, so ``final.index`` names
    the loaded rows that survived.
    """
    key = pd.util.hash_pandas_object(start.astype(str), index=False)
    bad = pd.Series(False, index=start.index)
    for fault in manifest.faults:
        if fault.kind == "impossible_values" and fault.column in start:
            bad |= pd.to_numeric(start[fault.column], errors="coerce") < 0
    should_keep = set(key[~bad])
    kept = set(key[final.index])
    return len(should_keep - kept)


def _counting(planner: Runnable, log: list[float]) -> Runnable:
    """Wrap the planner to record each call's duration, including calls that
    raise (timeouts, transport errors), which ``plan_once`` turns into
    ``plan_failed``."""
    def call(messages: Any) -> Any:
        start = time.monotonic()
        try:
            return planner.invoke(messages)
        finally:
            log.append(time.monotonic() - start)
    return RunnableLambda(call)


def run_one(settings: Settings, planner: Runnable, condition: Condition, seed: int,
            fixtures: Path) -> EvalRow:
    """One graph run, start to finish, scored against the seed's manifest."""
    csv = write_fixture(fixtures, seed)
    _, manifest = make_dirty(seed)
    calls: list[float] = []
    graph = build_graph(settings, planner=_counting(planner, calls))
    approver = approver_for(condition, settings.evaluation)
    row = EvalRow(model=settings.model.name, think=settings.model.think, condition=condition,
                  seed=seed, status="crashed", faults_total=len(manifest.faults))
    thread = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{settings.model.name}-{condition}-s{seed}")
    traces = tracing(settings, thread, f"`evaluate`, condition {condition}, seed {seed}")
    config = {**run_config(thread), "callbacks": traces.callbacks}
    start = time.monotonic()
    try:
        inp: dict | Command = {"input_path": str(csv)}
        while True:
            if traces.graph_events:
                for event in graph.stream(inp, config, stream_mode="debug"):
                    traces.graph_events(event)
            else:
                graph.invoke(inp, config)
            state = graph.get_state(config)
            if not state.interrupts:
                break
            request = state.interrupts[0].value
            answer = approver(request, load_frame(Path(state.values["current_path"])))
            row.approvals += 1
            row.rejections += answer.action == "reject"
            inp = Command(resume=answer.model_dump(mode="json"))
    except Exception as exc:  # Scored as zero; the run stays in the denominator.
        row.error = f"{type(exc).__name__}: {exc}"
        _trace(row, graph.get_state(config).values)
        return _timed(row, start, calls)
    values = state.values
    _trace(row, values)
    row.status = values["status"]
    row.replans = values.get("retries", 0)
    if row.status != "done":
        row.error = values.get("last_error")
        return _timed(row, start, calls)
    begin, final = load_frame(Path(values["start_path"])), load_frame(Path(values["current_path"]))
    row.faults = check_all(final, manifest)
    row.faults_fixed = sum(row.faults.values())
    row.rows_removed = len(begin) - len(final)
    row.collateral_rows = collateral_rows(begin, final, manifest)
    return _timed(row, start, calls)


def _trace(row: EvalRow, values: dict) -> None:
    """Keep the plan and audit log, so a low score or a crash can be traced."""
    if "plan" in values:
        row.plan = values["plan"].model_dump(mode="json")["ops"]
    row.audit = [describe(e) for e in values.get("audit", [])]


def _timed(row: EvalRow, start: float, calls: list[float]) -> EvalRow:
    row.wall_seconds = round(time.monotonic() - start, 1)
    row.model_calls = len(calls)
    row.model_seconds = round(sum(calls), 1)
    return row


def summarize(rows: Iterable[EvalRow]) -> str:
    """A Markdown table, one line per model, thinking setting, and condition.
    Counts, not rates: ten fixtures do not support more precision."""
    groups: dict[tuple[str, str, str], list[EvalRow]] = {}
    for r in rows:
        groups.setdefault((r.model, str(r.think), r.condition), []).append(r)
    head = ("| Model | Think | Condition | Runs | Done / failed / crashed | Faults fixed "
            "| Fully cleaned | Approvals asked | Rejections | Replans | Hit retry limit "
            "| Collateral rows (runs) | Model calls | Median wall s |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for (model, think, condition), rs in groups.items():
        n = len(rs)
        by = {s: sum(r.status == s for r in rs) for s in ("done", "failed", "crashed")}
        fixed, total = sum(r.faults_fixed for r in rs), sum(r.faults_total for r in rs)
        cleaned = sum(r.status == "done" and r.faults_fixed == r.faults_total for r in rs)
        collateral = [r.collateral_rows for r in rs if r.collateral_rows]
        lines.append(
            f"| `{model}` | {think} | {condition} | {n} "
            f"| {by['done']} / {by['failed']} / {by['crashed']} | {fixed}/{total} "
            f"| {cleaned}/{n} | {sum(r.approvals for r in rs)} | {sum(r.rejections for r in rs)} "
            f"| {sum(r.replans for r in rs)} | {by['failed']} "
            f"| {sum(collateral)} ({len(collateral)}) | {sum(r.model_calls for r in rs)} "
            f"| {median(r.wall_seconds for r in rs):.1f} |")
    return "\n".join(lines)


def read_rows(paths: Iterable[Path]) -> list[EvalRow]:
    return [EvalRow.model_validate_json(line)
            for p in paths for line in p.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    base = Settings()
    parser = argparse.ArgumentParser(prog="triage.evaluate",
                                     description="Run and score the graph on synthetic fixtures.")
    parser.add_argument("--models", nargs="+", default=[base.model.name])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out", type=Path, default=Path("runs/eval"),
                        help="results go to <out>/<UTC timestamp>/")
    parser.add_argument("--summarize", type=Path, nargs="+", metavar="JSONL",
                        help="print the table for earlier results instead of running")
    args = parser.parse_args(argv)

    if args.summarize:
        print(summarize(read_rows(args.summarize)))
        return 0
    out = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    rows: list[EvalRow] = []
    with (out / "results.jsonl").open("w") as log:
        for model in args.models:
            settings = base.model_copy(update={
                "model": base.model.model_copy(update={"name": model}),
                "runs_dir": out / "runs",
            })
            planner = make_planner(settings.model)
            for condition in args.conditions:
                for seed in args.seeds:
                    row = run_one(settings, planner, condition, seed, out / "fixtures")
                    rows.append(row)
                    log.write(row.model_dump_json() + "\n")
                    log.flush()
                    print(f"{model} {condition} seed={seed} {row.status} "
                          f"fixed={row.faults_fixed}/{row.faults_total} asked={row.approvals} "
                          f"rejected={row.rejections} replans={row.replans} "
                          f"collateral={row.collateral_rows} calls={row.model_calls} "
                          f"{row.wall_seconds}s", flush=True)
    table = summarize(rows)
    (out / "summary.md").write_text(table + "\n")
    print(f"\n{table}\n\nresults: {out / 'results.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
