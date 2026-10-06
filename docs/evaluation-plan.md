# Evaluation plan (M7)

Built in `triage/evaluate.py`; results are in
[`implementation-plan.md`](implementation-plan.md#m7-evaluation).

## Data

Synthetic fixtures from `triage/faults.py`: a seeded clean "orders" table with
eight fault kinds injected (missing-value markers, whitespace padding, numbers
stored as text, impossible negative values, missing values, mixed date formats,
a constant column, duplicate rows). Each fixture has a manifest recording what
was injected. Generate with:

```
uv run python -m triage.faults --out fixtures --seeds 0 1 2 3 4 5 6 7 8 9
```

Fixtures are generated, not committed. Manifests never go into the planner's
input.

## Scoring

Outcome checks in `triage.faults.CHECKS` judge the cleaned output, not the
ops chosen. Any plan that removes a fault gets credit for it. A check also
fails if it "fixes" a fault by dropping the column (except the constant-column
fault, where dropping is the fix).

## Approval policy during evaluation

Evaluation runs need a scripted approver. Report two conditions:

1. **Approve all**: measures what the planner proposes, unfiltered.
2. **Rule-based approver**: rejects ops that remove more than a set fraction of
   rows or drop a column with more than one distinct value. Measures how the
   graph behaves when a person pushes back.

   As built, it also rejects `filter_rows(operator='not_null')`
   (`EvaluationConfig.reject_null_filters`). Measured on seeds 0–9, every
   single-fault removal is 7–9% of rows: the wanted ones (dedupe, the
   negative-quantity filter) and the unwanted ones (dropping rows whose region
   or rating is missing) alike. A size threshold cannot tell them apart, so
   without this rule the condition would differ from approve-all only on
   dropped columns. The size limit (`reject_rows_removed_fraction`, 0.10)
   stays, and catches a filter that removes more than one fault's worth.

## Metrics, per condition

- Faults fixed / faults injected, over all fixtures. A run that fails, times
  out, or crashes counts as zero faults fixed and stays in the denominator.
- Fully cleaned fixtures / all fixtures.
- Approvals requested per run, and rejections.
- Replans per run; runs that hit the retry limit.
- Collateral damage: rows removed beyond the injected duplicates and
  impossible values, compared with the manifest. As built: the distinct rows
  of the loaded frame, other than those with an injected impossible value,
  that have no copy left in the output. Rows are matched by content, so
  `dedupe(keep='last')` is not counted.
- Wall-clock time per run, and model calls per run.

Each run's row in `results.jsonl` also keeps the final plan and the audit
log, so a low score can be traced to the plan that caused it.

Report each metric for both models (`qwen3.5:9b-mlx`, `qwen3.8:27b-mlx`).
With 10 fixtures, report counts rather than implying precision a sample this
small doesn't have.

## Running

```
uv run python -m triage.evaluate --models qwen3.5:9b-mlx qwen3.8:27b-mlx \
    --seeds 0 1 2 3 4 5 6 7 8 9
uv run python -m triage.evaluate --summarize runs/eval/<stamp>/results.jsonl
```

Results go to `runs/eval/<UTC timestamp>/`: `results.jsonl` (one row per
run), `summary.md`, the fixtures, and each run's frames. Runs use an
in-memory checkpointer; durability is not under test here. There is no
per-run wall-clock limit: each model call is bounded by
`ModelConfig.timeout_s`, and a timed-out call counts as a failed plan
attempt against `max_plan_retries`.

## Separation of checks

- **Offline, default** (`uv run pytest -m "not live"`): executor, impact,
  profiler, fault checks, graph routing with a fake model.
- **Live** (`uv run pytest -m live`, and the evaluation runner): calls local
  Ollama. Slow, free, and kept out of the default check.
