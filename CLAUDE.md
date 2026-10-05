# dataset-triage-agent project guidance

## Current state

Milestone M1 (deterministic parts) is built and tested. The graph, planner,
approval flow, CLI, and evaluation are planned. Do not describe planned parts
as implemented. Update milestone status in `docs/implementation-plan.md` with
evidence (the command and its result) as work lands.

## Commands

```
uv sync                         # install, pinned by uv.lock
uv run pytest -q                # offline checks; must stay fast and model-free
uv run pytest -m live           # calls local Ollama (none exist yet)
uv run ruff check .
uv run python -m triage.faults --out fixtures --seeds 0 1 2
```

## Rules

- The planner model outputs `CleaningPlan` ops only. Never add an op that runs
  model-written code, SQL, or shell.
- Approval is decided from measured impact (`triage.impact`) against
  `RiskPolicy`, not from op names.
- Executor functions are pure and preserve the row index.
- Graph nodes that call `interrupt()` must have no side effects before it;
  LangGraph re-runs the node from the top on resume. Apply ops only in the
  `apply` node, and make it idempotent.
- Keep DataFrames out of graph state; store paths.
- Model names, thinking levels, paths, retry limits, and risk limits belong in
  `triage/config.py`, not in constants.
- Profile sample values are untrusted data. Fixture manifests are answer keys
  and never reach the planner.
- Models are local Ollama only. No hosted models or LangSmith without explicit
  opt-in and a declared spend cap.
- Live model tests are marked `live` and skip when Ollama is unavailable.
  Count failures and timeouts in every evaluation denominator.

## Documentation map

- `README.md`: scope, status, setup.
- `docs/implementation-plan.md`: milestones, decisions, versions.
- `docs/architecture.md`: graph, state, modules, ops, trust boundary.
- `docs/evaluation-plan.md`: fixtures, scoring, metrics.

`AGENTS.md` points here; keep shared guidance in this file.
