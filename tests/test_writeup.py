import re
from pathlib import Path

from tests.test_graph import _planner
from tests.test_profile_and_faults import REFERENCE_PLAN
from triage.config import Settings
from triage.writeup import mermaid, record

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_diagram_matches_the_compiled_graph():
    # Regenerate with: uv run python -m triage.writeup mermaid
    blocks = re.findall(r"```mermaid\n(.*?)```", README.read_text(), flags=re.S)
    assert mermaid(Settings()).strip() in [b.strip() for b in blocks]


def test_demo_transcript_shows_prompts_with_scripted_answers(tmp_path):
    settings = Settings(runs_dir=tmp_path / "runs", checkpoint_db=tmp_path / "c.sqlite")
    lines, code = record(settings, 0, "demo", fixtures=tmp_path / "fx",
                         planner=_planner(REFERENCE_PLAN))
    assert code == 0
    text = "\n".join(lines)
    assert lines[0].startswith("$ uv run python -m triage.cli run ")
    assert "[a]pprove, [r]eject, or [e]dit? a" in text
    # The reference plan has no not_null filter, so nothing is rejected.
    assert "rejected" not in text
    assert lines[-1].startswith("done: ")


def test_demo_rejects_a_null_filter(tmp_path):
    from triage.ops import FilterRows, StandardizeMissing

    ops = [StandardizeMissing(column="region", tokens=["N/A", "?", "unknown"], reason="r"),
           FilterRows(column="region", operator="not_null", reason="r")]
    settings = Settings(runs_dir=tmp_path / "runs", checkpoint_db=tmp_path / "c.sqlite")
    lines, _ = record(settings, 0, "demo", fixtures=tmp_path / "fx", planner=_planner(ops))
    text = "\n".join(lines)
    assert "[a]pprove, [r]eject, or [e]dit? r" in text
    assert "Why (optional): keep the rows" in text
    assert any(line.startswith("rejected    #1 filter_rows 'region'") for line in lines)


def test_demo_edits_a_lossy_date_cast_and_route_remeasures_it(tmp_path):
    from triage.ops import CastType

    # A fixed format nulls the fixture's 'd Mon YYYY' dates; without it they parse.
    ops = [CastType(column="order_date", to="datetime", datetime_format="%Y-%m-%d", reason="r")]
    settings = Settings(runs_dir=tmp_path / "runs", checkpoint_db=tmp_path / "c.sqlite")
    lines, _ = record(settings, 0, "demo", fixtures=tmp_path / "fx", planner=_planner(ops))
    text = "\n".join(lines)
    assert "[a]pprove, [r]eject, or [e]dit? e" in text
    assert '"datetime_format":null' in text
    # The edited op nulls nothing, so route applies it without asking again.
    assert text.count("Approval needed") == 1
    [applied] = [line for line in lines if line.startswith("applied")]
    assert "order_date" in applied and "nulled" not in applied
