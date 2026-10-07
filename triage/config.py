"""Validated configuration. Every tunable lives here, not in module constants.

Values come from defaults, then environment variables prefixed ``TRIAGE_``
with ``__`` between nested names, e.g. ``TRIAGE_MODEL__NAME=qwen3.5:9b-mlx``.
"""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    name: str = "qwen3.8:27b-mlx"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    # Passed to Ollama's ``think`` parameter; None leaves the model default.
    # Off by default: with thinking on, qwen3.5:9b-mlx spent a 3000-token
    # budget reasoning and never emitted the plan (M2 gate, 2026-10-05).
    think: bool | str | None = False
    # Caps thinking plus output tokens, so a runaway generation stops with
    # done_reason=length instead of running until the timeout.
    num_predict: int | None = 4096
    timeout_s: float = 600.0
    # Longest planner error kept for the audit log and fed back on a retry.
    # The parser's own message embeds the whole completion (4 KB when a reply
    # was cut off), which floods the terminal and the retry prompt.
    max_error_chars: int = Field(default=500, ge=50)


class RiskPolicy(BaseModel):
    """An op needs human approval when its measured impact exceeds any limit.

    Risk comes from what the op would actually do to this data (see
    ``triage.impact``), not from the op's name: a cast that parses every value
    is safe, the same cast that turns 40 values into nulls is not.
    """

    max_rows_removed: int = Field(default=0, ge=0)
    max_columns_removed: int = Field(default=0, ge=0)
    max_values_nulled: int = Field(default=0, ge=0)
    # Within a run, reuse a person's approve/reject for the same op (ignoring
    # its reason) on the same input bytes, e.g. when a replan proposes it
    # again. Same bytes and parameters give the same impact, so the person
    # would be answering an identical question.
    reuse_decisions: bool = True


class ValidationPolicy(BaseModel):
    """Checks on a finished run's output; a failure sends the run back to the
    planner, up to ``Settings.max_plan_retries`` times."""

    # Cumulative, over all ops, which a person approving one op at a time does
    # not see. The fixtures lose about 15% of rows to dedupe plus the
    # impossible-quantity filter.
    max_rows_removed_fraction: float = Field(default=0.25, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    """The scripted approvers in ``triage.evaluate``. The rule-based one
    stands in for a person who pushes back on large deletions."""

    # Of the rows in the frame the op sees. On seeds 0-9 every single-fault
    # removal is 7-9% of rows, the wanted ones (dedupe, the negative-quantity
    # filter) and the unwanted ones (dropping rows with a missing region or
    # rating) alike, so this only catches a filter that removes more than one
    # fault's worth.
    reject_rows_removed_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    # Reject ``filter_rows(operator='not_null')``: deleting whole rows because
    # one value is missing, where a person would rather impute or keep the
    # null. The size rule above cannot tell these from wanted removals.
    reject_null_filters: bool = True


class ProfilerConfig(BaseModel):
    sample_values: int = Field(default=5, ge=0)
    max_sample_chars: int = Field(default=40, ge=1)
    # Strings treated as probable missing-value markers when profiling.
    # Compared after stripping whitespace and lowercasing.
    missing_tokens: list[str] = ["", "n/a", "na", "null", "none", "-", "?", "unknown"]


class TraceConfig(BaseModel):
    """The Markdown traces (``triage.trace``) that ``triage.cli`` and
    ``triage.evaluate`` append to under ``runs_dir/<thread>/``."""

    enabled: bool = True
    file_name: str = "trace.md"
    # Each model call's rendered prompt, delta and reply, without the steps.
    prompts_file_name: str = "trace_prompts.md"
    # Ask Ollama for the exact prompt text it feeds the model. Costs one extra
    # request per model call, which renders without generating.
    render_prompt: bool = True
    render_timeout_s: float = 10.0
    # A second file with the libraries' own debug output, to compare with
    # the trace: LangChain's set_debug handler and LangGraph's debug stream.
    # It logs whole graph states at every step, so it runs to megabytes.
    debug_enabled: bool = True
    debug_file_name: str = "trace_debug.md"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGE_", env_nested_delimiter="__")

    model: ModelConfig = ModelConfig()
    risk: RiskPolicy = RiskPolicy()
    validation: ValidationPolicy = ValidationPolicy()
    profiler: ProfilerConfig = ProfilerConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    trace: TraceConfig = TraceConfig()
    # Cell values read as null on load. pandas' default list would hide markers
    # like 'N/A' before the agent ever sees them.
    csv_na_values: list[str] = [""]
    # Extra planner calls after a plan fails to parse or fails validation.
    max_plan_retries: int = Field(default=2, ge=0)
    checkpoint_db: Path = Path("runs/checkpoints.sqlite")
    # Each run writes its intermediate frames and output to runs_dir/<thread_id>.
    runs_dir: Path = Path("runs")
