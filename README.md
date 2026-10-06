# dataset-triage-agent

A LangGraph agent that cleans a messy CSV with a human in the loop. It
profiles the data, proposes a cleaning plan as typed operations, applies safe
ones automatically, and pauses for approval on anything destructive. Runs
survive a crash and can be replayed from earlier checkpoints. Everything runs
locally on Ollama, on synthetic data.

## Status

Early. The deterministic parts are built and tested: op schema, executor,
impact measurement, profiler, and synthetic fault fixtures (milestone M1).
The planner, the graph, and human approval are built: ops whose measured
impact exceeds the risk policy pause the run for a person to approve, reject,
or edit, via `uv run python -m triage.cli run <csv> --thread <id>`. Runs are
checkpointed to SQLite, so a stopped or killed run continues with
`uv run python -m triage.cli resume --thread <id>`; `uv run python -m
triage.crash_demo` shows it. Each output is validated, and a failing run is
replanned up to a limit. Replay and evaluation are planned, not built. See
[`docs/implementation-plan.md`](docs/implementation-plan.md).

## Design in one paragraph

The model never writes code. It outputs a `CleaningPlan` of seven typed ops
(`drop_column`, `cast_type`, `impute`, `dedupe`, `filter_rows`,
`standardize_missing`, `strip_whitespace`), validated by pydantic. A
deterministic executor applies them. Before applying, each op is dry-run and
its impact measured (rows removed, columns removed, values turned null). Ops
over the configured risk policy wait for a person to approve, edit, or reject.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, for model runs (from M2),
a local [Ollama](https://ollama.com/) with the configured model pulled.

```
uv sync
uv run pytest -q                       # offline checks
uv run ruff check .
uv run python -m triage.faults --out fixtures --seeds 0 1 2
```

## Docs

- [`docs/implementation-plan.md`](docs/implementation-plan.md): milestones, decisions, versions
- [`docs/architecture.md`](docs/architecture.md): graph, state, modules, ops
- [`docs/evaluation-plan.md`](docs/evaluation-plan.md): fixtures, scoring, metrics

## Related

[`ds-research-agent`](../ds-research-agent) builds its agent loop by hand
rather than on a framework. This project uses LangGraph because its
checkpointing, interrupts, and routing match the problem. The final README
section (M8) compares the two choices.
