"""M2 compatibility gate: does structured output give valid plans per model?

For each model and fixture seed, calls the planner once (no retry) and
records whether the output parsed into a ``CleaningPlan``, whether every op
then applied without an ``OpError``, and how long the call took. Failures and
timeouts stay in the denominator.

    python -m triage.gate --models qwen3.5:9b-mlx qwen3.8:27b-mlx --seeds 0 1 2

Fault coverage is reported for information only; scoring belongs to M7. The
manifest is read here for that, after planning, and never reaches the model.
"""

import argparse
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel

from triage.config import Settings
from triage.executor import OpError, apply_op
from triage.faults import check_all, make_dirty, write_fixture
from triage.io import load_csv
from triage.planner import make_planner, plan_once
from triage.profile import profile


class GateRow(BaseModel):
    model: str
    seed: int
    parsed: bool
    applied: bool
    n_ops: int
    faults_fixed: int
    faults_total: int
    seconds: float
    error: str | None
    plan: dict | None
    raw_content: str | None


def run_one(settings: Settings, model: str, seed: int, workdir: Path) -> GateRow:
    config = settings.model.model_copy(update={"name": model})
    df = load_csv(write_fixture(workdir, seed), settings.csv_na_values)
    attempt = plan_once(make_planner(config), profile(df, settings.profiler))
    _, manifest = make_dirty(seed)
    row = GateRow(model=model, seed=seed, parsed=attempt.plan is not None, applied=False,
                  n_ops=0, faults_fixed=0, faults_total=len(manifest.faults),
                  seconds=round(attempt.seconds, 1), error=attempt.error,
                  plan=None, raw_content=attempt.raw_content)
    if attempt.plan is None:
        return row
    row.plan, row.n_ops = attempt.plan.model_dump(), len(attempt.plan.ops)
    try:
        for op in attempt.plan.ops:
            df = apply_op(df, op)
    except OpError as exc:
        row.error = f"OpError: {exc}"
        return row
    row.applied = True
    row.faults_fixed = sum(check_all(df, manifest).values())
    return row


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Structured-output compatibility gate.")
    parser.add_argument("--models", nargs="+", default=[settings.model.name])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path("runs/gate.jsonl"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[GateRow] = []
    with tempfile.TemporaryDirectory() as tmp, args.out.open("w") as log:
        for model in args.models:
            for seed in args.seeds:
                row = run_one(settings, model, seed, Path(tmp))
                rows.append(row)
                log.write(row.model_dump_json() + "\n")
                log.flush()
                print(f"{model} seed={seed} parsed={row.parsed} applied={row.applied} "
                      f"ops={row.n_ops} fixed={row.faults_fixed}/{row.faults_total} "
                      f"{row.seconds}s {row.error or ''}", flush=True)

    print("\nmodel | parsed | applied | mean s")
    for model in args.models:
        mine = [r for r in rows if r.model == model]
        n = len(mine)
        print(f"{model} | {sum(r.parsed for r in mine)}/{n} | {sum(r.applied for r in mine)}/{n} | "
              f"{sum(r.seconds for r in mine) / n:.1f}")
    print(json.dumps({"log": str(args.out)}))


if __name__ == "__main__":
    main()
