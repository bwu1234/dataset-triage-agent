"""Typed cleaning operations: the only thing the planner model may output.

The model never writes code. It returns a ``CleaningPlan`` of these ops and the
deterministic executor in ``triage.executor`` applies them.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: str = Field(description="One sentence on why this op is needed, citing the profile.")


class DropColumn(_Op):
    op: Literal["drop_column"] = "drop_column"
    column: str


class CastType(_Op):
    """Values that fail to parse become null, so a lossy cast shows up in its impact."""

    op: Literal["cast_type"] = "cast_type"
    column: str
    to: Literal["int", "float", "string", "datetime", "bool"]
    datetime_format: str | None = Field(
        default=None, description="strftime format for to='datetime'; omit to accept mixed formats."
    )


class Impute(_Op):
    op: Literal["impute"] = "impute"
    column: str
    strategy: Literal["mean", "median", "mode", "constant"]
    value: str | float | None = Field(default=None, description="Required for strategy='constant'.")

    @model_validator(mode="after")
    def _value_matches_strategy(self) -> "Impute":
        if (self.strategy == "constant") != (self.value is not None):
            raise ValueError("value is required for strategy='constant' and not allowed otherwise")
        return self


class Dedupe(_Op):
    op: Literal["dedupe"] = "dedupe"
    subset: list[str] | None = Field(default=None, description="Columns to compare; omit for all.")
    keep: Literal["first", "last"] = "first"


class FilterRows(_Op):
    """Keep only rows where ``column <operator> value`` holds. Nulls fail comparisons."""

    op: Literal["filter_rows"] = "filter_rows"
    column: str
    operator: Literal["==", "!=", "<", "<=", ">", ">=", "not_null"]
    value: str | float | None = None

    @model_validator(mode="after")
    def _value_matches_operator(self) -> "FilterRows":
        if (self.operator == "not_null") != (self.value is None):
            raise ValueError("value is required for comparisons and not allowed for 'not_null'")
        return self


class StandardizeMissing(_Op):
    """Replace missing-value markers such as 'N/A' with real nulls."""

    op: Literal["standardize_missing"] = "standardize_missing"
    column: str
    tokens: list[str] = Field(min_length=1, description="Matched after strip, case-insensitive.")


class StripWhitespace(_Op):
    op: Literal["strip_whitespace"] = "strip_whitespace"
    column: str


AnyOp = DropColumn | CastType | Impute | Dedupe | FilterRows | StandardizeMissing | StripWhitespace

# A discriminated union: pydantic picks the model from the ``op`` field, which
# gives the planner a precise JSON schema and clear validation errors.
Op = Annotated[AnyOp, Field(discriminator="op")]


class CleaningPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ops: list[Op] = Field(description="Ops in the order they should be applied.")
