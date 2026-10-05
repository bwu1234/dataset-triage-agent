# Implementation plan

A two-week build. Each milestone records its status with evidence. Do not mark
a milestone done without a command that shows it.

## Goal

A LangGraph agent that takes a messy CSV, proposes a cleaning plan as typed
ops, applies safe ops automatically, pauses for human approval on destructive
ones, survives a crash mid-run, validates its own output, and writes an audit
log. Everything runs locally on synthetic data.

The project exists to use LangGraph where its features fit: typed state,
conditional routing, `interrupt()`, durable checkpoints, and replay. The
companion project `ds-research-agent` uses a hand-written loop instead; the
README contrasts the two choices.

## Milestones

| ID | Days | Milestone | Status |
|---|---|---|---|
| M1 | 1–2 | Op schema, profiler, executor, impact measurement, fault generator, offline tests | Done |
| M2 | 3–4 | Graph with planner node and risk routing, in-memory checkpointer | In progress (gate passed) |
| M3 | 5 | Human approval via `interrupt()`, CLI | Planned |
| M4 | 6–7 | `SqliteSaver`, crash-and-resume demo | Planned |
| M5 | 8 | Validation node and bounded replan loop | Planned |
| M6 | 9 | Replay from earlier checkpoints | Planned |
| M7 | 10 | Evaluation on synthetic fixtures | Planned |
| M8 | 10 | README, graph diagram, LangGraph-vs-hand-written write-up | Planned |

### M1: deterministic parts (done)

Delivered: `triage/ops.py` (seven ops as a pydantic discriminated union),
`triage/executor.py`, `triage/impact.py`, `triage/profile.py`, `triage/io.py`,
`triage/faults.py`, `triage/config.py`.

Evidence (2026-10-04): `uv run pytest -q` gives 36 passed, also with `-W error`;
`uv run ruff check .` is clean. `tests/test_profile_and_faults.py` shows, on
seeds 0–2, that every injected fault fails its check on the dirty file and
passes after a hand-written reference plan, and that the profiler surfaces a
hint for each fault.

### M2: graph and planner

- `State` as a `TypedDict`: input path, profile, plan, pending op index,
  audit log (list with an `operator.add` reducer), retry count, last error.
- Nodes: `load`, `profile`, `plan`, `route`, `apply`, `validate`, `finish`.
- `plan` calls `ChatOllama(...).with_structured_output(CleaningPlan)`.
- `route` dry-runs each op with `assess()` and uses a conditional edge:
  safe ops go to `apply`, ops over the risk policy go to `approve` (M3).
- An `OpError` from the dry run goes into the audit log and on to the next op.
- Compile with `InMemorySaver`.
- **Compatibility gate (first task):** confirm `with_structured_output` returns
  valid `CleaningPlan` objects from `qwen3.5:9b-mlx` and `qwen3.8:27b-mlx`. If
  output is unreliable, add a parse-and-retry step that feeds the pydantic
  error back to the model, counted against `max_plan_retries`.
- Live tests are marked `live` and skip when Ollama or the model is missing.

**Gate result (2026-10-05): passed; no parse-and-retry step needed.**
`uv run python -m triage.gate --models qwen3.5:9b-mlx qwen3.8:27b-mlx --seeds 0 1 2 3 4`,
one call per fixture, no retry, `think=False`:

| Model | Parsed | Applied without `OpError` | Faults fixed (info) | Mean s |
|---|---|---|---|---|
| `qwen3.5:9b-mlx` | 5/5 | 3/5 | 3–4 of 8 | 15.9 |
| `qwen3.8:27b-mlx` | 5/5 | 5/5 | 5–6 of 8 | 38.2 |

- Both 9b failures were op order (`impute` median on `amount` while still
  text), not malformed output. Schema retry would not fix that; the
  `OpError`-to-audit-log rule here and the M5 replan loop do.
- With the model's default thinking on, `qwen3.5:9b-mlx` spent a 3000-token
  budget reasoning and emitted no plan; without a cap it ran past 10 minutes.
  `ModelConfig` now defaults to `think=False` and `num_predict=4096`.
- `uv run pytest -q -m live` gives 2 passed (one per model, seed 0);
  `uv run pytest -q` gives 40 passed, 2 deselected; ruff clean.

### M3: human approval

- An `approve` node calls `interrupt()` with the op, its reason, and its
  measured `Impact`. The resume value is `approve`, `reject`, or `edit` with a
  replacement op validated against `Op`.
- Keep side effects out of `approve`. On resume LangGraph re-runs the node
  from the top, so anything before `interrupt()` runs twice. Ops are applied
  only in `apply`.
- CLI: `uv run python -m triage.cli run <csv> --thread <id>` streams progress,
  shows each pending approval, and resumes with `Command(resume=...)`.

### M4: durable runs

- Swap in `SqliteSaver` at `Settings.checkpoint_db`.
- Demo script: start a run, kill the process while it waits for approval,
  restart with the same `thread_id`, and finish.
- The DataFrame is not stored in graph state. State holds file paths;
  `apply` writes each intermediate frame to `runs/<thread_id>/step_<n>.parquet`.
  This keeps checkpoints small and makes `apply` idempotent: it skips work
  whose output file already exists.

### M5: validation and replanning

- `validate` re-profiles the output and checks invariants: rows removed within
  a configured fraction, no column with more nulls than before unless an
  approved op caused it, and the plan's target dtypes reached.
- On failure, route back to `plan` with the failure text, up to
  `max_plan_retries`; then finish with status `failed`.

### M6: replay

- `uv run python -m triage.cli history --thread <id>` lists checkpoints from
  `graph.get_state_history(config)`.
- `... fork --thread <id> --checkpoint <cid>` resumes from an earlier
  checkpoint so a different approval decision can be tried.

### M7: evaluation

See `docs/evaluation-plan.md`.

### M8: write-up

README with the Mermaid graph from `graph.get_graph().draw_mermaid()`, a
recorded demo run, the evaluation table, and a section on when LangGraph
earned its place here and why `ds-research-agent` does not use it.

## Decisions

- **The model outputs typed ops, never code.** That removes the need for a
  sandbox and makes every change auditable. Arbitrary transformations are out
  of scope.
- **Risk is measured, not named.** An op needs approval when its dry-run
  impact exceeds `RiskPolicy` limits (rows removed, columns removed, values
  turned null). A lossless cast is safe; the same cast on dirty text is not.
  Consequence: `standardize_missing` always needs approval under the default
  policy, because it turns markers into nulls. That is deliberate: the model
  is claiming those strings mean "missing", and a person should confirm it.
- **Only empty cells are null on load** (`csv_na_values`). pandas' default list
  would turn 'N/A' into null before the agent sees it.
- **Versions** (resolved by `uv sync` on 2026-10-04, pinned in `uv.lock`):
  Python 3.14.7, langgraph 1.2.12, langgraph-checkpoint-sqlite 3.1.1,
  langchain-ollama 1.1.0, langchain-core 1.6.6, ollama 0.6.3, pandas 3.0.6,
  pydantic 2.13.5, pydantic-settings 2.15.0, numpy 2.5.3.
- **Models:** `qwen3.5:9b-mlx` for development iterations, `qwen3.8:27b-mlx`
  for the final demo and evaluation. Set with `TRIAGE_MODEL__NAME`.

## Out of scope

- Model-written code and sandboxing.
- Web UI (Streamlit is a stretch goal after M8).
- LangSmith or any hosted tracing; it needs an account and sends data off the
  machine. Local logs only.
- Multi-agent graphs and deployment.

## Open questions

- Whether thinking with a larger token budget improves plan quality enough to
  justify the latency. Measure in M7 rather than guess.
- Whether to store intermediate frames as Parquet (needs `pyarrow`) or CSV.
