import re
from pathlib import Path

from tests.test_graph import _planner
from tests.test_profile_and_faults import REFERENCE_PLAN
from triage.config import Settings
from triage.graph import build_graph, open_checkpointer, run_config
from triage.writeup import _mermaid_text, mermaid, record, run_path

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


def _path(settings, thread="demo"):
    with open_checkpointer(settings.checkpoint_db) as saver:
        return run_path(build_graph(settings, _planner([]), saver), thread)


def test_path_draws_each_node_with_route_decisions_on_the_edges(tmp_path):
    settings = Settings(runs_dir=tmp_path / "runs", checkpoint_db=tmp_path / "c.sqlite")
    record(settings, 0, "demo", fixtures=tmp_path / "fx", planner=_planner(REFERENCE_PLAN))
    lines = _path(settings).splitlines()
    boxes = [line.split("<b>")[1].split("</b>")[0] for line in lines if "<b>" in line]
    assert boxes[:3] == ["load", "profile", "plan"]
    assert boxes[-2:] == ["validate", "finish"]
    # Each route that only decided is an edge label, not a box.
    assert "route" not in boxes
    assert any("-->|within limits|" in line for line in lines)
    assert any("-->|over limits|" in line for line in lines)
    assert any("-->|all ops routed|" in line for line in lines)
    approvals = [line for line in lines if "<b>approve</b>" in line]
    assert approvals and all(line.endswith("<br/>approved\"]:::approved") for line in approvals)
    assert lines[-3:-1] == ['  done(["end: done"])', f"  s{len(boxes)} --> done"]


def test_path_shows_a_rejection_with_its_note_and_an_edit_field_by_field(tmp_path):
    from triage.ops import CastType, FilterRows, StandardizeMissing

    ops = [StandardizeMissing(column="region", tokens=["N/A", "?", "unknown"], reason="r"),
           FilterRows(column="region", operator="not_null", reason="r"),
           CastType(column="order_date", to="datetime", datetime_format="%Y-%m-%d", reason="r")]
    settings = Settings(runs_dir=tmp_path / "runs", checkpoint_db=tmp_path / "c.sqlite")
    record(settings, 0, "demo", fixtures=tmp_path / "fx", planner=_planner(ops))
    text = _path(settings)
    assert ("rejected: keep the rows; a missing value is not a reason to delete them\"]:::rejected"
            in text)
    assert "edited: datetime_format: '%Y-%m-%d' → None\"]:::edited" in text
    assert "classDef rejected" in text and "classDef edited" in text


def test_path_of_a_stopped_run_ends_at_the_pending_approval(tmp_path):
    from triage.faults import write_fixture

    settings = Settings(runs_dir=tmp_path / "runs", checkpoint_db=tmp_path / "c.sqlite")
    csv = write_fixture(tmp_path / "fx", 0)
    with open_checkpointer(settings.checkpoint_db) as saver:
        build_graph(settings, _planner(REFERENCE_PLAN), saver).invoke(
            {"input_path": str(csv)}, run_config("demo"))
    lines = _path(settings).splitlines()
    pending = next(line for line in lines if "stopped before approve" in line)
    assert pending.endswith(":::pending") and "#35;1 " in pending
    assert not any("done([" in line for line in lines)


def test_path_of_an_unknown_thread_is_none(tmp_path):
    assert _path(Settings(runs_dir=tmp_path, checkpoint_db=tmp_path / "c.sqlite"), "nope") is None


def test_mermaid_text_escapes_untrusted_label_text():
    text = _mermaid_text('a"b#1;<i>&`c`|d\x1b[2J', 90)
    assert text == "a#quot;b#35;1;#lt;i#gt;#amp;#96;c#96;#124;d\\x1b[2J"
    assert _mermaid_text("x" * 10, 5) == "xxxx…"
