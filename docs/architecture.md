# Architecture

Status markers: **(built)** exists and is tested; **(planned)** does not exist yet.

## Graph (planned, M2–M5)

```mermaid
flowchart TD
    START --> load --> profile --> plan
    plan --> route
    route -- safe --> apply
    route -- over risk policy --> approve
    route -- op cannot apply --> route
    approve -- approve / edit --> apply
    approve -- reject --> route
    apply -- more ops --> route
    apply -- plan done --> validate
    validate -- passes --> finish --> END
    validate -- fails, retries left --> plan
    validate -- fails, no retries --> finish
```

`approve` pauses the run with `interrupt()`. The run resumes when the CLI
sends `Command(resume=...)` on the same `thread_id`. Checkpoints are written
by `SqliteSaver` after every step, so a killed process resumes where it stopped.

## State (planned)

| Field | Type | Notes |
|---|---|---|
| `input_path` | `str` | Source CSV |
| `current_path` | `str` | Latest intermediate frame on disk |
| `profile` | `DatasetProfile` | From `triage.profile` |
| `plan` | `CleaningPlan` | From the planner |
| `op_index` | `int` | Next op to route |
| `audit` | `list[AuditEntry]` | `operator.add` reducer, so nodes append |
| `retries` | `int` | Replans used |
| `last_error` | `str \| None` | Fed back to the planner |

Frames live on disk, not in state, which keeps checkpoints small.

## Modules

| Module | Role | Status |
|---|---|---|
| `triage/ops.py` | Seven op models as a pydantic discriminated union; `CleaningPlan` | built |
| `triage/executor.py` | `apply_op`: pure, keeps the row index, raises `OpError` | built |
| `triage/impact.py` | `measure`, `assess` (dry run), `needs_approval` | built |
| `triage/profile.py` | Deterministic profile with per-column fault hints | built |
| `triage/io.py` | `load_csv` with explicit null markers | built |
| `triage/faults.py` | Synthetic dirty fixtures, manifests, outcome checks | built |
| `triage/config.py` | `Settings` via pydantic-settings, `TRIAGE_` env prefix | built |
| `triage/graph.py` | State, nodes, edges, compile | planned |
| `triage/planner.py` | Prompt and `ChatOllama` structured output, single attempt | built |
| `triage/gate.py` | M2 compatibility gate: parse and apply rate per model | built |
| `triage/cli.py` | `run`, `resume`, `history`, `fork` | planned |
| `triage/evaluate.py` | Fixture evaluation runner | planned |

## Ops

| Op | Effect | Usually needs approval? |
|---|---|---|
| `drop_column` | Remove a column | Yes (column removed) |
| `cast_type` | Parse to int, float, string, datetime or bool; failures become null | Only if values fail to parse |
| `impute` | Fill nulls by mean, median, mode or a constant | No |
| `dedupe` | Drop duplicate rows | Yes, if any exist |
| `filter_rows` | Keep rows matching a comparison | Yes, if rows are removed |
| `standardize_missing` | Turn marker strings into nulls | Yes (values nulled) |
| `strip_whitespace` | Trim string values | No |

The last column is what the default `RiskPolicy` produces. Approval is decided
from measured impact, so the same op can go either way depending on the data.

## Trust boundary

Profile sample values come from the file and are untrusted text. The planner
prompt presents them as data. Ops are validated against the schema before
anything runs, and unknown ops or extra fields are rejected, so text in a
data file cannot add an operation the schema doesn't define.

The fixture manifests are answer keys. They stay out of the planner's input.
