# Evaluation plan (planned, M7)

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

## Metrics, per condition

- Faults fixed / faults injected, over all fixtures. A run that fails, times
  out, or crashes counts as zero faults fixed and stays in the denominator.
- Fully cleaned fixtures / all fixtures.
- Approvals requested per run, and rejections.
- Replans per run; runs that hit the retry limit.
- Collateral damage: rows removed beyond the injected duplicates and
  impossible values, compared with the manifest.
- Wall-clock time per run, and model calls per run.

Report each metric for both models (`qwen3.5:9b-mlx`, `qwen3.8:27b-mlx`).
With 10 fixtures, report counts rather than implying precision a sample this
small doesn't have.

## Separation of checks

- **Offline, default** (`uv run pytest -m "not live"`): executor, impact,
  profiler, fault checks, graph routing with a fake model.
- **Live** (`uv run pytest -m live`, and the evaluation runner): calls local
  Ollama. Slow, free, and kept out of the default check.
