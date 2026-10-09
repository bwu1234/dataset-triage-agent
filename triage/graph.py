"""The triage graph: load, profile, plan, route each op to apply or approve,
then validate the output.

An op whose measured impact exceeds ``RiskPolicy`` goes to ``approve``, which
pauses the run with ``interrupt()`` until a person approves, rejects, or edits
it. Once every op is routed, ``validate`` checks the output
(``triage.validate``). On failure the run goes back to ``plan`` with the
failures as feedback, up to ``max_plan_retries`` times; a replan starts again
from the loaded file, because ops already applied cannot be undone by adding
more.

``open_checkpointer`` gives the durable SQLite saver the CLI uses: a run
stopped at an approval, or killed mid-node, continues from its last
checkpoint in a new process.

Frames never enter graph state. Each node reads the frame at ``current_path``
and ``apply`` writes a new file, so checkpoints hold only paths and small
pydantic models. Every file under ``run_dir`` is named by a hash of its
content's inputs, so a fork (``triage.cli fork``) that re-runs a node on a
branch of the same thread writes new files and never replaces one another
branch still points at.
"""

import hashlib
import json
import operator
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, get_args

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, model_validator

from triage.config import Settings
from triage.executor import OpError, apply_op
from triage.impact import Impact, assess, measure, needs_approval
from triage.io import load_csv, load_frame, save_frame
from triage.ops import AnyOp, CleaningPlan, Op
from triage.planner import make_planner, plan_once
from triage.profile import ColumnProfile, DatasetProfile, profile
from triage.validate import check_output


class AuditEntry(BaseModel):
    action: Literal["loaded", "planned", "plan_failed", "applied", "skipped", "approved",
                    "rejected", "edited", "validated", "invalid", "finished"]
    op_index: int | None = None
    op: Op | None = None
    impact: Impact | None = None
    detail: str | None = None


class ApprovalRequest(BaseModel):
    """What ``approve`` shows a person: the op and its measured impact."""

    op_index: int
    op: Op
    impact: Impact
    # Totals in the frame the op would run on, so an impact can be read as a
    # share: "216 values nulled" is every date, or one in ten. ``non_null``
    # counts the op's column, or the whole frame for an op without one. None
    # in a request saved before these fields existed.
    rows: int | None = None
    non_null: int | None = None


class ApprovalDecision(BaseModel):
    """The resume value for ``approve``. ``edit`` replaces the op; the
    replacement goes back through ``route`` so its own impact is measured."""

    action: Literal["approve", "reject", "edit"]
    op: Op | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _op_only_for_edit(self) -> "ApprovalDecision":
        if (self.action == "edit") != (self.op is not None):
            raise ValueError("op is required for action='edit' and not allowed otherwise")
        return self


class State(TypedDict, total=False):
    input_path: str
    run_dir: str
    # The loaded frame; every plan attempt starts from it.
    start_path: str
    current_path: str
    profile: DatasetProfile
    plan: CleaningPlan
    op_index: int
    # Set by ``plan``, ``route``, ``approve`` and ``validate`` for the
    # conditional edge that follows each.
    decision: Literal["route", "replan", "apply", "approve", "next", "done"]
    # ``operator.add`` is the reducer: nodes return new entries and LangGraph
    # appends them, instead of each node rewriting the whole list.
    audit: Annotated[list[AuditEntry], operator.add]
    # Planner calls after the first, for parse or validation failures.
    retries: int
    # Why the last plan failed; the next plan call gets it as feedback.
    last_error: str | None
    # A person's answers by ``op_key``, reused when the same op meets the same
    # data again (``RiskPolicy.reuse_decisions``).
    decisions: dict[str, Literal["approve", "reject"]]
    status: Literal["running", "done", "failed"]
    output_path: str


# Every pydantic type that can appear in state. Passing an explicit allowlist
# makes the checkpointer revive only these (plus LangGraph's built-in safe
# types) instead of importing whatever class a checkpoint names.
STATE_TYPES: tuple[type, ...] = (
    AuditEntry, Impact, DatasetProfile, ColumnProfile, CleaningPlan, ApprovalRequest,
    ApprovalDecision, *get_args(AnyOp),
)


def make_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)


@contextmanager
def open_checkpointer(db: Path) -> Iterator[SqliteSaver]:
    """A durable checkpointer at ``db``, closed on exit. LangGraph may call it
    from worker threads, hence ``check_same_thread=False``; SqliteSaver
    serialises access with its own lock."""
    db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db, check_same_thread=False)) as conn:
        yield SqliteSaver(conn, serde=make_serde())


def check_thread_id(thread_id: str) -> str:
    """The thread id names a directory under ``runs_dir``, so it must be a
    plain name: no separators, no leading dot."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", thread_id):
        raise ValueError(f"thread id must be letters, digits, '_', '.', '-': {thread_id!r}")
    return thread_id


def run_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def op_key(src: Path, op: AnyOp) -> str:
    """Identifies "this op on this data": the input file's bytes and the op
    without its ``reason``, which is the model's commentary and does not change
    what the op does. Names step files and keys reused decisions."""
    digest = hashlib.sha256(src.read_bytes())
    digest.update(op.model_dump_json(exclude={"reason"}).encode())
    return digest.hexdigest()[:16]


def this_attempt(audit: list[AuditEntry]) -> list[AuditEntry]:
    """Entries after the latest ``planned`` one: what happened to the current plan."""
    starts = [i for i, e in enumerate(audit) if e.action == "planned"]
    return audit[starts[-1] + 1:] if starts else []


def _outcomes(entries: list[AuditEntry]) -> tuple[dict[int, str], set[int], dict[str, int]]:
    """Skipped ops with their errors, rejected op indexes, and values nulled per
    column by applied ops. An edited op keeps its index, so the last entry for
    an index is its outcome."""
    final: dict[int, AuditEntry] = {}
    nulled: dict[str, int] = {}
    for e in entries:
        if e.op_index is not None and e.action in ("applied", "rejected", "skipped"):
            final[e.op_index] = e
        column = getattr(e.op, "column", None)
        if e.action == "applied" and column is not None and e.impact is not None:
            nulled[column] = nulled.get(column, 0) + e.impact.values_nulled
    skipped = {i: e.detail or "" for i, e in final.items() if e.action == "skipped"}
    rejected = {i for i, e in final.items() if e.action == "rejected"}
    return skipped, rejected, nulled


def _feedback(state: State) -> str:
    """What the planner is told after a failed attempt."""
    parts = []
    if "plan" in state:
        parts.append(f"Previous plan: {state['plan'].model_dump_json()}")
    parts.append(f"What went wrong:\n{state['last_error']}")
    rejected = [e.op for e in this_attempt(state["audit"]) if e.action == "rejected"]
    if rejected:
        parts.append("A person rejected these ops:\n"
                     + "\n".join(op.model_dump_json() for op in rejected))
    return "\n\n".join(parts)


def build_graph(
    settings: Settings,
    planner: Runnable[Any, dict[str, Any]] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the graph. ``planner`` defaults to the configured Ollama model;
    tests pass a fake. ``checkpointer`` defaults to an in-memory saver; pass
    one from ``open_checkpointer`` for runs that outlive the process."""
    planner = planner if planner is not None else make_planner(settings.model)

    def load(state: State, config: RunnableConfig) -> State:
        """Read the CSV into ``step_000_<key>.pkl``, keyed on the file's bytes
        and the load settings: a fork that re-runs ``load`` after the CSV
        changed must not replace the frame the original branch starts from."""
        run_dir = settings.runs_dir / check_thread_id(config["configurable"]["thread_id"])
        src = Path(state["input_path"])
        df = load_csv(src, settings.csv_na_values)
        key = hashlib.sha256(src.read_bytes())
        key.update(json.dumps(settings.csv_na_values).encode())
        path = run_dir / f"step_000_{key.hexdigest()[:16]}.pkl"
        save_frame(df, path)
        return {
            "run_dir": str(run_dir), "start_path": str(path), "current_path": str(path),
            "op_index": 0, "retries": 0,
            "last_error": None, "status": "running",
            "audit": [AuditEntry(action="loaded", detail=f"{len(df)} rows, {len(df.columns)} columns")],
        }

    def profile_node(state: State) -> State:
        df = load_frame(Path(state["current_path"]))
        return {"profile": profile(df, settings.profiler)}

    def plan(state: State) -> State:
        """Ask for a plan, with feedback if an earlier one failed. A new plan
        starts again from the loaded file."""
        retries = state.get("retries", 0)
        feedback = _feedback(state) if state.get("last_error") else None
        attempt = plan_once(planner, state["profile"], feedback,
                            max_error_chars=settings.model.max_error_chars)
        if attempt.plan is None:
            failed = AuditEntry(action="plan_failed", detail=attempt.error)
            if retries < settings.max_plan_retries:
                return {"decision": "replan", "retries": retries + 1, "last_error": attempt.error,
                        "audit": [failed]}
            return {"decision": "done", "status": "failed", "last_error": attempt.error,
                    "audit": [failed]}
        return {
            "decision": "route", "plan": attempt.plan, "op_index": 0,
            "current_path": state["start_path"], "last_error": None,
            "audit": [AuditEntry(action="planned",
                                 detail=f"{len(attempt.plan.ops)} ops in {attempt.seconds:.1f}s, "
                                        f"attempt {retries + 1}")],
        }

    def route(state: State) -> State:
        """Dry-run the next op and decide what happens to it. Reads only, so it
        is safe to re-run."""
        i, ops = state["op_index"], state["plan"].ops
        if i >= len(ops):
            return {"decision": "done"}
        op = ops[i]
        try:
            impact = assess(load_frame(Path(state["current_path"])), op)
        except OpError as e:
            return {"decision": "next", "op_index": i + 1,
                    "audit": [AuditEntry(action="skipped", op_index=i, op=op, detail=str(e))]}
        if not needs_approval(impact, settings.risk):
            return {"decision": "apply"}
        earlier = None
        if settings.risk.reuse_decisions:
            earlier = state.get("decisions", {}).get(op_key(Path(state["current_path"]), op))
        if earlier == "approve":
            return {"decision": "apply",
                    "audit": [AuditEntry(action="approved", op_index=i, op=op, impact=impact,
                                         detail="reused: approved earlier on the same data")]}
        if earlier == "reject":
            return {"decision": "next", "op_index": i + 1,
                    "audit": [AuditEntry(action="rejected", op_index=i, op=op, impact=impact,
                                         detail="reused: rejected earlier on the same data")]}
        return {"decision": "approve"}

    def approve(state: State) -> State:
        """Pause for a person. LangGraph re-runs this node from the top on
        resume, so everything before ``interrupt()`` only reads."""
        i, ops = state["op_index"], state["plan"].ops
        src = Path(state["current_path"])
        df = load_frame(src)
        impact = assess(df, ops[i])
        column = getattr(ops[i], "column", None)
        cells = df[column] if column in df.columns else df
        answer = interrupt(ApprovalRequest(op_index=i, op=ops[i], impact=impact, rows=len(df),
                                           non_null=int(cells.notna().to_numpy().sum())),
                           response_schema=ApprovalDecision)
        if answer.action in ("approve", "reject"):
            decisions = {**state.get("decisions", {}), op_key(src, ops[i]): answer.action}
        if answer.action == "approve":
            return {"decision": "apply", "decisions": decisions,
                    "audit": [AuditEntry(action="approved", op_index=i, op=ops[i], impact=impact,
                                         detail=answer.note)]}
        if answer.action == "reject":
            return {"decision": "next", "op_index": i + 1, "decisions": decisions,
                    "audit": [AuditEntry(action="rejected", op_index=i, op=ops[i], impact=impact,
                                         detail=answer.note)]}
        new_ops = [*ops[:i], answer.op, *ops[i + 1:]]
        return {"decision": "next", "plan": CleaningPlan(ops=new_ops),
                "audit": [AuditEntry(action="edited", op_index=i, op=answer.op,
                                     detail=answer.note or f"replaced {ops[i].op}")]}

    def apply(state: State) -> State:
        """Apply ``plan.ops[op_index]``. The output file is named by ``op_key``,
        so a re-run (say, after a crash) or a replan that repeats the op reuses
        it, while a fork that picks a different op, or a reused thread id on
        another CSV, gets a different file rather than a stale one."""
        i = state["op_index"]
        op = state["plan"].ops[i]
        src = Path(state["current_path"])
        out = Path(state["run_dir"]) / f"step_{i + 1:03d}_{op_key(src, op)}.pkl"
        before = load_frame(src)
        if out.exists():
            after = load_frame(out)
        else:
            after = apply_op(before, op)
            save_frame(after, out)
        return {"current_path": str(out), "op_index": i + 1,
                "audit": [AuditEntry(action="applied", op_index=i, op=op,
                                     impact=measure(before, after))]}

    def validate(state: State) -> State:
        """Check the output against the loaded file. Reads only."""
        skipped, rejected, nulled = _outcomes(this_attempt(state["audit"]))
        failures = check_output(
            load_frame(Path(state["start_path"])), load_frame(Path(state["current_path"])),
            state["plan"].ops, skipped, rejected, nulled, settings.validation,
        )
        if not failures:
            return {"decision": "done", "audit": [AuditEntry(action="validated")]}
        text = "\n".join(failures)
        invalid = AuditEntry(action="invalid", detail="; ".join(failures))
        retries = state.get("retries", 0)
        if retries < settings.max_plan_retries:
            return {"decision": "replan", "retries": retries + 1, "last_error": text,
                    "audit": [invalid]}
        return {"decision": "done", "status": "failed", "last_error": text, "audit": [invalid]}

    def finish(state: State) -> State:
        if state.get("status") == "failed":
            return {"audit": [AuditEntry(action="finished", detail=f"failed: {state['last_error']}")]}
        # Named by the final frame, so two branches of a thread that end
        # differently do not overwrite each other's output.
        src = Path(state["current_path"])
        key = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        out = Path(state["run_dir"]) / f"cleaned_{key}.csv"
        load_frame(src).to_csv(out, index=False)
        return {"status": "done", "output_path": str(out),
                "audit": [AuditEntry(action="finished", detail=str(out))]}

    builder = StateGraph(State)
    builder.add_node("load", load)
    builder.add_node("profile", profile_node)
    builder.add_node("plan", plan)
    builder.add_node("route", route)
    builder.add_node("approve", approve)
    builder.add_node("apply", apply)
    builder.add_node("validate", validate)
    builder.add_node("finish", finish)

    builder.add_edge(START, "load")
    builder.add_edge("load", "profile")
    builder.add_edge("profile", "plan")
    builder.add_conditional_edges(
        "plan", lambda s: s["decision"], {"route": "route", "replan": "plan", "done": "finish"}
    )
    builder.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"apply": "apply", "approve": "approve", "next": "route", "done": "validate"},
    )
    builder.add_conditional_edges(
        "approve", lambda s: s["decision"], {"apply": "apply", "next": "route"}
    )
    builder.add_edge("apply", "route")
    builder.add_conditional_edges(
        "validate", lambda s: s["decision"], {"replan": "plan", "done": "finish"}
    )
    builder.add_edge("finish", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver(serde=make_serde()))
