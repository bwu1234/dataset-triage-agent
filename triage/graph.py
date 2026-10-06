"""The triage graph: load, profile, plan, then route each op to apply or hold.

M2 scope. An op whose measured impact exceeds ``RiskPolicy`` is held: logged
and not applied. M3 replaces that branch with an ``approve`` node that calls
``interrupt()``. Validation and replanning (M5) are not wired in yet, so a run
ends at ``finish`` once every op has been routed.

Frames never enter graph state. Each node reads the frame at ``current_path``
and ``apply`` writes a new file, so checkpoints hold only paths and small
pydantic models.
"""

import hashlib
import operator
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, get_args

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from triage.config import Settings
from triage.executor import OpError, apply_op
from triage.impact import Impact, assess, measure, needs_approval
from triage.io import load_csv, load_frame, save_frame
from triage.ops import AnyOp, CleaningPlan, Op
from triage.planner import make_planner, plan_once
from triage.profile import ColumnProfile, DatasetProfile, profile


class AuditEntry(BaseModel):
    action: Literal["loaded", "planned", "plan_failed", "applied", "skipped", "held", "finished"]
    op_index: int | None = None
    op: Op | None = None
    impact: Impact | None = None
    detail: str | None = None


class State(TypedDict, total=False):
    input_path: str
    run_dir: str
    current_path: str
    profile: DatasetProfile
    plan: CleaningPlan
    op_index: int
    # Set by ``route`` for the conditional edge that follows it.
    decision: Literal["apply", "next", "done"]
    # ``operator.add`` is the reducer: nodes return new entries and LangGraph
    # appends them, instead of each node rewriting the whole list.
    audit: Annotated[list[AuditEntry], operator.add]
    retries: int
    last_error: str | None
    status: Literal["running", "done", "failed"]
    output_path: str


# Every pydantic type that can appear in state. Passing an explicit allowlist
# makes the checkpointer revive only these (plus LangGraph's built-in safe
# types) instead of importing whatever class a checkpoint names.
STATE_TYPES: tuple[type, ...] = (
    AuditEntry, Impact, DatasetProfile, ColumnProfile, CleaningPlan, *get_args(AnyOp),
)


def make_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)


def run_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def build_graph(
    settings: Settings,
    planner: Runnable[Any, dict[str, Any]] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the graph. ``planner`` defaults to the configured Ollama model;
    tests pass a fake. ``checkpointer`` defaults to an in-memory saver."""
    planner = planner if planner is not None else make_planner(settings.model)

    def load(state: State, config: RunnableConfig) -> State:
        run_dir = settings.runs_dir / config["configurable"]["thread_id"]
        df = load_csv(Path(state["input_path"]), settings.csv_na_values)
        path = run_dir / "step_000.pkl"
        save_frame(df, path)
        return {
            "run_dir": str(run_dir), "current_path": str(path), "op_index": 0, "retries": 0,
            "last_error": None, "status": "running",
            "audit": [AuditEntry(action="loaded", detail=f"{len(df)} rows, {len(df.columns)} columns")],
        }

    def profile_node(state: State) -> State:
        df = load_frame(Path(state["current_path"]))
        return {"profile": profile(df, settings.profiler)}

    def plan(state: State) -> State:
        attempt = plan_once(planner, state["profile"])
        if attempt.plan is None:
            return {
                "status": "failed", "last_error": attempt.error,
                "audit": [AuditEntry(action="plan_failed", detail=attempt.error)],
            }
        return {
            "plan": attempt.plan, "op_index": 0, "last_error": None,
            "audit": [AuditEntry(action="planned",
                                 detail=f"{len(attempt.plan.ops)} ops in {attempt.seconds:.1f}s")],
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
        if needs_approval(impact, settings.risk):
            # M3: route to `approve` here instead of holding.
            return {"decision": "next", "op_index": i + 1,
                    "audit": [AuditEntry(action="held", op_index=i, op=op, impact=impact,
                                         detail="exceeds risk policy; not applied without approval")]}
        return {"decision": "apply"}

    def apply(state: State) -> State:
        """Apply ``plan.ops[op_index]``. The output file is named by a hash of
        its input path and the op, so a re-run reuses it and a fork that picks
        a different op writes a different file rather than reading a stale one."""
        i = state["op_index"]
        op = state["plan"].ops[i]
        src = Path(state["current_path"])
        key = hashlib.sha256(f"{src}\n{op.model_dump_json()}".encode()).hexdigest()[:16]
        out = Path(state["run_dir"]) / f"step_{i + 1:03d}_{key}.pkl"
        before = load_frame(src)
        if out.exists():
            after = load_frame(out)
        else:
            after = apply_op(before, op)
            save_frame(after, out)
        return {"current_path": str(out), "op_index": i + 1,
                "audit": [AuditEntry(action="applied", op_index=i, op=op,
                                     impact=measure(before, after))]}

    def finish(state: State) -> State:
        if state.get("status") == "failed":
            return {"audit": [AuditEntry(action="finished", detail=f"failed: {state['last_error']}")]}
        out = Path(state["run_dir"]) / "cleaned.csv"
        load_frame(Path(state["current_path"])).to_csv(out, index=False)
        return {"status": "done", "output_path": str(out),
                "audit": [AuditEntry(action="finished", detail=str(out))]}

    builder = StateGraph(State)
    builder.add_node("load", load)
    builder.add_node("profile", profile_node)
    builder.add_node("plan", plan)
    builder.add_node("route", route)
    builder.add_node("apply", apply)
    builder.add_node("finish", finish)

    builder.add_edge(START, "load")
    builder.add_edge("load", "profile")
    builder.add_edge("profile", "plan")
    builder.add_conditional_edges(
        "plan", lambda s: "finish" if s.get("status") == "failed" else "route", ["route", "finish"]
    )
    builder.add_conditional_edges(
        "route", lambda s: s["decision"], {"apply": "apply", "next": "route", "done": "finish"}
    )
    builder.add_edge("apply", "route")
    builder.add_edge("finish", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver(serde=make_serde()))
