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
| M2 | 3–4 | Graph with planner node and risk routing, in-memory checkpointer | Done |
| M3 | 5 | Human approval via `interrupt()`, CLI | Done |
| M4 | 6–7 | `SqliteSaver`, crash-and-resume demo | Done |
| M5 | 8 | Validation node and bounded replan loop | Done |
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

### M2: graph and planner (done)

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

**Graph result (2026-10-05): built in `triage/graph.py`.** Differences from
the list above, on purpose:

- `validate` is not a node yet. It is all M5 behaviour (invariants, replan),
  so `route` goes to `finish` when the ops run out. `apply` always returns to
  `route`, which also covers a plan whose last op is skipped.
- Ops over the risk policy are **held**: logged with their impact and not
  applied. M3 changes that one branch in `route` to go to `approve`. A hold
  can make later ops risky too: on seed 0, holding `standardize_missing` on
  `amount` leaves markers that the following cast would null, so it is held.
- `apply` writes `runs/<thread_id>/step_<n>_<hash>.pkl`, the hash taken over
  its input path and the op. A re-run reuses the file; an M6 fork that picks
  a different op gets a different file, not a stale one.
- Checkpoints use `JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)`,
  so only the state's own pydantic types are revived.

Evidence: `uv run pytest -q` gives 46 passed, 2 deselected (6 new in
`tests/test_graph.py`, fake planner); the graph tests also pass with
`-W error` and with `LANGGRAPH_STRICT_MSGPACK=true`; ruff clean. A live run on
seed 0 with `qwen3.5:9b-mlx` (one-off script, not a test) planned 5 ops in
14.5 s: 1 applied (`strip_whitespace`), 4 held (two `standardize_missing`,
`filter_rows`, `dedupe`), status `done`.

### M3: human approval (done)

- An `approve` node calls `interrupt()` with the op, its reason, and its
  measured `Impact`. The resume value is `approve`, `reject`, or `edit` with a
  replacement op validated against `Op`.
- Keep side effects out of `approve`. On resume LangGraph re-runs the node
  from the top, so anything before `interrupt()` runs twice. Ops are applied
  only in `apply`.
- CLI: `uv run python -m triage.cli run <csv> --thread <id>` streams progress,
  shows each pending approval, and resumes with `Command(resume=...)`.

**Result (2026-10-06): built.** `approve` in `triage/graph.py`,
`triage/cli.py`. Differences from the list above, on purpose:

- An **edit goes back to `route`, not to `apply`**. The person has not seen
  the replacement op's impact, so `route` measures it: applied if safe, asked
  again if still over the policy, skipped if it cannot apply.
- `interrupt(..., response_schema=ApprovalDecision)` (langgraph 1.2) validates
  the resume value with pydantic and returns the model; the CLI sends it as
  plain JSON so it survives any checkpointer.
- `approve` re-measures impact before `interrupt()` (a read, safe to repeat)
  rather than carrying it in state from `route`.
- The CLI escapes control characters in everything it prints (column names,
  values and model reasons come from the file), and thread ids must be plain
  names because they become a directory under `runs_dir`.
- Only `run` exists. With `InMemorySaver` a stopped run cannot be resumed, so
  `resume` waits for M4 (built there).

Evidence: `uv run pytest -q` gives 60 passed, 2 deselected (10 in
`tests/test_graph.py`, 10 in `tests/test_cli.py`, fake planner, scripted
answers); those two files also pass with `LANGGRAPH_STRICT_MSGPACK=true` and
`-W error`; ruff clean. Live (`TRIAGE_MODEL__NAME=qwen3.5:9b-mlx`, seed 0,
`yes a` piped to `triage.cli run`): 5 ops planned in 41.0 s, 1 applied
without asking, 4 approved and applied, status `done`.

### M4: durable runs (done)

- Swap in `SqliteSaver` at `Settings.checkpoint_db`.
- Demo script: start a run, kill the process while it waits for approval,
  restart with the same `thread_id`, and finish.
- The DataFrame is not stored in graph state. State holds file paths;
  `apply` writes each intermediate frame to `runs/<thread_id>/` (built in
  M2; see above). This keeps checkpoints small and makes `apply` idempotent.

**Result (2026-10-05): built.** `open_checkpointer` in `triage/graph.py`,
`resume` in `triage/cli.py`, demo in `triage/crash_demo.py`. Differences
from the list above, on purpose:

- **`cli run` refuses a thread that already has checkpoints.** Checked
  against langgraph 1.2.12: new input on an interrupted thread starts again
  at `load`, drops the pending approval, and appends a second `loaded`/
  `planned` pair to the audit log. `cli resume --thread <id>` is the only way
  to continue a thread.
- `resume` covers two crash points: a pending `interrupt()` (asks, then sends
  `Command(resume=...)`) and a node that died mid-run (streams `None`, so
  LangGraph re-runs only that node; mid-`plan` that means one more model
  call).
- **`apply` now keys its output file on the input file's bytes, not its
  path.** With durable thread ids, a reused id (say, after deleting the
  database) rewrites `step_000.pkl` from another CSV under the same name, and
  the path-keyed hash read the old run's frames. Re-runs on the same data
  still reuse files: pickling the same frame gives the same bytes.
- The demo kills with SIGKILL, not Ctrl-C, so no handler or cleanup runs.
  The CLI prints the approval prompt only after the interrupt checkpoint is
  written, so the kill always lands after it.
- Not handled: two processes resuming the same thread at once, and a
  resume under different `Settings` (risk policy, `runs_dir`) than the run
  started with. `route` uses the resuming process's policy.

Evidence: `uv run pytest -q` gives 65 passed, 2 deselected (5 new in
`tests/test_durable.py`): a real subprocess SIGKILLed at the approval prompt
and resumed by a second process (`tests/cli_child.py`, fake planner);
a crash inside `apply` resumed through a fresh connection, with that node
alone re-run; `run` refusing an existing thread; `resume` of unknown and
finished threads; a reused thread id on another CSV giving the same output
as a fresh run (fails with the old path-keyed hash). The graph, CLI and
durable tests also pass with `LANGGRAPH_STRICT_MSGPACK=true` and `-W error`;
ruff clean. Live (`TRIAGE_MODEL__NAME=qwen3.5:9b-mlx uv run python -m
triage.crash_demo --seed 0`, 47.5 s): 5 ops planned in 44.4 s, process
SIGKILLed at approval #1, the second process resumed at `approve`, approved
4 ops, status `done`, exit 0; the final audit log has one `planned` entry.
Separately, Ctrl-C during planning exits 130 and `resume` re-runs `plan`,
then finishes with status `done`.

### M5: validation and replanning (done)

- `validate` re-profiles the output and checks invariants: rows removed within
  a configured fraction, no column with more nulls than before unless an
  approved op caused it, and the plan's target dtypes reached.
- On failure, route back to `plan` with the failure text, up to
  `max_plan_retries`; then finish with status `failed`.

**Result (2026-10-05): built.** `validate` node in `triage/graph.py`, checks
in `triage/validate.py`, `ValidationPolicy` in `triage/config.py`.
Differences from the list above, on purpose:

- **A replan starts again from the loaded file (`step_000`)**, not from the
  failed attempt's output. Two of the invariants (rows removed, unexplained
  nulls) are caused by ops already applied, and adding ops cannot undo them.
  Its cost, asking the person again, is mostly removed by reusing decisions
  (below).
- **A person's approve/reject is reused within a run** for the same op on
  the same input bytes (`op_key`: file bytes plus the op without `reason`,
  which the model rewrites on every attempt). Same bytes and parameters give
  the same measured impact, so it is the same question. Reused entries say
  so in the audit log; edits are not reused (the replacement is routed on
  its own). `RiskPolicy.reuse_decisions=false` turns it off. Step files use
  the same key, so a reworded op also reuses its file. A person who wants to
  change an answer needs a new run, or a fork once M6 exists.
- **An op skipped with `OpError` fails validation.** That is the failure the
  M2 gate saw (op order), and this loop is what fixes it.
- Rows removed is cumulative (`max_rows_removed_fraction`, default 0.25). A
  person approves one op at a time and never sees the total.
- The null check compares each column's new nulls with the
  `values_nulled` of applied ops on that column. `route` already gates each
  op, so this only fires if an op nulls values outside its own column: a
  cross-check on the audit log, not a quality check. It reads frames
  directly; no re-profile is needed.
- A cast is checked only if it was applied (not rejected by a person, not
  skipped); the failure is a later op undoing it, e.g. a text constant
  imputed into a datetime column turns it into `object`.
- Feedback to the planner: the previous plan, the failures, and the ops a
  person rejected (with "do not re-propose"), fenced as untrusted data like
  the profile.
- **Planner parse and transport failures share the `max_plan_retries`
  budget** and are retried with the error as feedback, instead of ending the
  run.
- **Planner errors are summarised, not quoted.** LangChain's parse error
  embeds the whole completion (about 4 KB on seed 3 below), which then went to
  the audit log, the terminal and the retry prompt. `plan_once` now uses the
  wrapped pydantic/JSON error (`__cause__`), adds a "cut off at the token
  limit, keep reasons short" note when Ollama reports `done_reason=length`,
  and caps the text at `ModelConfig.max_error_chars` (500). The full reply
  stays in `PlanAttempt.raw_content`.
- Out of retries, the run finishes `failed` without `cleaned.csv`. The last
  frame stays in `runs/<thread>/`.
- Executor fix found on the way: `impute(constant)` with a text value on a
  `Float64`/`Int64` column raised a raw `TypeError`, which `route` does not
  catch, so the run crashed instead of skipping the op. It now raises
  `OpError`.

Evidence: `uv run pytest -q` gives 82 passed, 2 deselected. New tests:
`tests/test_validate.py` (5, one per check), 8 in `tests/test_graph.py`
(replan restarts from the loaded file with feedback, rejected ops named in
feedback, retries exhausted, parse failure retried, decision reuse for a
reworded op but not for other bytes or with reuse off, reused rejection),
2 in `tests/test_planner.py` (the seed-3 cut-off reply through LangChain's
real `PydanticOutputParser`, the cap), 1 in `tests/test_cli.py`, 1 in
`tests/test_executor.py`. Putting `reason` back into `op_key` fails both
reuse tests. Graph, CLI, durable, validate and planner tests also pass with `LANGGRAPH_STRICT_MSGPACK=true`
and `-W error`; ruff clean. Live (`qwen3.5:9b-mlx`, seeds 0–4, `yes a`
piped to `triage.cli run`): 5/5 `done`. Seeds 0, 1, 2 and 4 validated on
the first plan. Seed 3's first call ran into `num_predict` mid-`reason`
and returned truncated JSON (`plan_failed`); the retry validated. No
validation-driven replan happened live: the op-order errors from the M2
gate did not recur on these seeds, so that path is covered by the
fake-planner tests only. (These runs predate the error summary and decision
reuse.) A rerun of seed 3 after both did not reproduce the cut-off: the same
prompt at temperature 0 gave a different 8-op plan, which validated first
time (36 s), so MLX output is not run-to-run deterministic.

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
- **Intermediate frames are pickles** (`triage.io.save_frame`), resolved in
  M2. CSV loses dtypes between steps, and Parquet needs `pyarrow` and rejects
  the mixed object columns `impute(constant)` can produce. Pickle loads run
  code, so these files have the same trust as the checkpoint database: only
  ever read ones this process wrote under `runs_dir`.
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
