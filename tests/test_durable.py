import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.cli_child import PLANNER
from triage import cli, graph
from triage.config import Settings
from triage.crash_demo import kill_at_first_approval, run_to_end
from triage.faults import write_fixture
from triage.graph import build_graph, open_checkpointer, run_config

ROOT = Path(__file__).parents[1]


def _settings(tmp_path):
    return Settings(runs_dir=tmp_path / "runs", checkpoint_db=tmp_path / "c.sqlite")


def _approve(_request):
    return cli.ApprovalDecision(action="approve")


def _audit(settings, thread="t"):
    """Read a thread's audit log through a fresh connection, as a new process would."""
    with open_checkpointer(settings.checkpoint_db) as saver:
        values = build_graph(settings, PLANNER, saver).get_state(run_config(thread)).values
    return [(e.action, e.op_index) for e in values["audit"]], values


def test_run_killed_at_approval_finishes_in_a_new_process(tmp_path, monkeypatch):
    env = {**os.environ, "TRIAGE_CHECKPOINT_DB": str(tmp_path / "c.sqlite"),
           "TRIAGE_RUNS_DIR": str(tmp_path / "runs")}
    child = [sys.executable, "-u", "-m", "tests.cli_child"]
    csv = write_fixture(tmp_path / "in", 0)
    first, second = [], []

    monkeypatch.chdir(ROOT)  # so ``-m tests.cli_child`` resolves
    assert kill_at_first_approval([*child, "run", str(csv), "--thread", "t"], first.append, 60,
                                  env=env)
    assert run_to_end([*child, "resume", "--thread", "t"], "a\n", second.append, 60, env=env) == 0

    assert any("Approval needed for #1" in line for line in second)
    actions, values = _audit(_settings(tmp_path))
    assert actions == [("loaded", None), ("planned", None), ("applied", 0), ("approved", 1),
                       ("applied", 1), ("finished", None)]
    assert "channel" not in pd.read_csv(values["output_path"]).columns


def test_crash_mid_node_reruns_only_that_node(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    csv = write_fixture(tmp_path / "in", 0)
    real_apply = graph.apply_op
    calls = []

    def crash_once(df, op):
        calls.append(op.op)
        if len(calls) == 2:
            raise RuntimeError("simulated crash inside apply")
        return real_apply(df, op)

    monkeypatch.setattr(graph, "apply_op", crash_once)
    with open_checkpointer(settings.checkpoint_db) as saver, pytest.raises(RuntimeError):
        cli.run(csv, "t", build_graph(settings, PLANNER, saver), _approve, lambda _: None)

    with open_checkpointer(settings.checkpoint_db) as saver:
        lines = []
        assert cli.resume("t", build_graph(settings, PLANNER, saver), _approve, lines.append) == 0
    assert "resuming at apply" in lines
    actions, _ = _audit(settings)
    assert actions == [("loaded", None), ("planned", None), ("applied", 0), ("approved", 1),
                       ("applied", 1), ("finished", None)]
    assert calls == ["strip_whitespace", "drop_column", "drop_column"]


def test_run_refuses_an_existing_thread(tmp_path):
    settings = _settings(tmp_path)
    csv = write_fixture(tmp_path / "in", 0)
    with open_checkpointer(settings.checkpoint_db) as saver:
        g = build_graph(settings, PLANNER, saver)
        assert cli.run(csv, "t", g, _approve, lambda _: None) == 0
        lines = []
        assert cli.run(csv, "t", g, _approve, lines.append) == 2
    assert "already exists" in lines[0]
    assert [a for a, _ in _audit(settings)[0]].count("loaded") == 1


def test_resume_of_unknown_or_finished_thread(tmp_path):
    settings = _settings(tmp_path)
    with open_checkpointer(settings.checkpoint_db) as saver:
        g = build_graph(settings, PLANNER, saver)
        lines = []
        assert cli.resume("nope", g, _approve, lines.append) == 2
        assert cli.run(write_fixture(tmp_path / "in", 0), "t", g, _approve, lambda _: None) == 0
        lines = []
        assert cli.resume("t", g, _approve, lines.append) == 0
    assert lines[-1].startswith("done:") and len(_audit(settings)[0]) == 6


def test_reused_thread_id_does_not_reuse_frames_from_another_input(tmp_path):
    # Same thread id and runs_dir, a fresh checkpoint store, a different CSV:
    # ``step_000.pkl`` keeps its name, so a path-keyed apply would read the
    # first run's stale output.
    def cleaned(seed, runs_dir, db):
        settings = Settings(runs_dir=runs_dir, checkpoint_db=db)
        with open_checkpointer(db) as saver:
            g = build_graph(settings, PLANNER, saver)
            assert cli.run(write_fixture(tmp_path / "in", seed), "t", g, _approve,
                           lambda _: None) == 0
            return pd.read_csv(g.get_state(run_config("t")).values["output_path"])

    cleaned(0, tmp_path / "runs", tmp_path / "a.sqlite")
    reused = cleaned(1, tmp_path / "runs", tmp_path / "b.sqlite")
    fresh = cleaned(1, tmp_path / "fresh", tmp_path / "c.sqlite")
    pd.testing.assert_frame_equal(reused, fresh)
