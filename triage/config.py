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


class RiskPolicy(BaseModel):
    """An op needs human approval when its measured impact exceeds any limit.

    Risk comes from what the op would actually do to this data (see
    ``triage.impact``), not from the op's name: a cast that parses every value
    is safe, the same cast that turns 40 values into nulls is not.
    """

    max_rows_removed: int = Field(default=0, ge=0)
    max_columns_removed: int = Field(default=0, ge=0)
    max_values_nulled: int = Field(default=0, ge=0)


class ProfilerConfig(BaseModel):
    sample_values: int = Field(default=5, ge=0)
    max_sample_chars: int = Field(default=40, ge=1)
    # Strings treated as probable missing-value markers when profiling.
    # Compared after stripping whitespace and lowercasing.
    missing_tokens: list[str] = ["", "n/a", "na", "null", "none", "-", "?", "unknown"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGE_", env_nested_delimiter="__")

    model: ModelConfig = ModelConfig()
    risk: RiskPolicy = RiskPolicy()
    profiler: ProfilerConfig = ProfilerConfig()
    # Cell values read as null on load. pandas' default list would hide markers
    # like 'N/A' before the agent ever sees them.
    csv_na_values: list[str] = [""]
    max_plan_retries: int = Field(default=2, ge=0)
    checkpoint_db: Path = Path("runs/checkpoints.sqlite")
