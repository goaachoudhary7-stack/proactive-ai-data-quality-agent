"""Result models for a schema-independent file profile.

Counts and ratios are stored explicitly. A ratio is null when its denominator
is zero so an empty file cannot produce a divide-by-zero result.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROCESSING_NOTE = (
    "This file is processed by the Streamlit server running this dashboard. "
    "When you run the dashboard on your computer, processing happens on that computer. "
    "If the dashboard is deployed later, processing would happen on the hosting server. "
    "The uploaded rows are not sent to Gemini, CrewAI, or another external model. "
    "An optional explanation sends only the summary you approve on this page. "
    "It is not saved into the project. "
    "Quality checks read the uploaded table. "
    "Approved cleaning changes a separate working copy and leaves the original upload in memory."
)

MissingRole = Literal["observation", "user_rule_violation", "user_rule_clear"]
KeyStatus = Literal["violation", "clear", "not_comparable"]
OutlierLabel = Literal["informational", "skipped"]


class Share(BaseModel):
    """A count divided by a named denominator."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    denominator: int = Field(ge=0)
    ratio: float | None
    unit: str


class MixedTypeFinding(BaseModel):
    """Possible mixed types. Category counts do not include cell values."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["informational"] = "informational"
    non_missing_count: int = Field(ge=0)
    categories: dict[str, int]
    detail: str


class OutlierFinding(BaseModel):
    """Possible numerical outliers, or the reason the check was skipped."""

    model_config = ConfigDict(extra="forbid")

    label: OutlierLabel
    non_missing_count: int = Field(ge=0)
    outlier_count: int | None = None
    denominator: int | None = None
    lower_fence: float | None = None
    upper_fence: float | None = None
    detail: str


class ColumnProfile(BaseModel):
    """One uploaded column, described without copying its cell values."""

    model_config = ConfigDict(extra="forbid")

    name: str
    inferred_type: str
    missing: Share
    missing_role: MissingRole
    mixed_type: MixedTypeFinding | None = None
    outlier: OutlierFinding | None = None


class ExactDuplicates(BaseModel):
    """Full-row duplicates. Both counts use every column and all data rows."""

    model_config = ConfigDict(extra="forbid")

    rows_in_duplicate_groups: Share
    extra_copies: Share
    detail: str


class KeyDuplicates(BaseModel):
    """Duplicates of a key the user selected. Separate from full-row duplicates."""

    model_config = ConfigDict(extra="forbid")

    key_columns: list[str]
    incomplete_key_rows: int = Field(ge=0)
    rows_in_duplicate_groups: Share
    extra_copies: Share
    status: KeyStatus
    detail: str


class AnalysisReport(BaseModel):
    """Deterministic profile of one CSV or one Excel worksheet."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    file_format: Literal["csv", "xlsx"]
    sheet_name: str | None
    data_rows: int = Field(ge=0)
    column_count: int = Field(ge=0)
    column_names: list[str]
    columns: list[ColumnProfile]
    exact_duplicates: ExactDuplicates
    key_duplicates: KeyDuplicates | None
    required_columns: list[str]
    processing_note: str


class CandidateKey(BaseModel):
    """A suggested single-column key. This is not a confirmed business rule."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    column: str
    data_rows: int = Field(ge=0)
    distinct_non_missing: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    detail: str


class CandidateRelationship(BaseModel):
    """A suggested link between two datasets. This is not a confirmed foreign key."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["suggestion"] = "suggestion"
    parent_dataset: str
    parent_column: str
    child_dataset: str
    child_column: str
    cardinality: Literal["one-to-one", "one-to-many"]
    child_non_missing: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    orphan_count: int = Field(ge=0)
    overlap_ratio: float
    detail: str


class DatasetResult(BaseModel):
    """One CSV or one Excel worksheet, or the error that stopped its analysis."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    source_name: str
    file_format: Literal["csv", "xlsx"] | None = None
    sheet_name: str | None = None
    error: str | None = None
    report: AnalysisReport | None = None


class BatchReport(BaseModel):
    """Independent dataset profiles plus cross-dataset suggestions."""

    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetResult]
    candidate_keys: list[CandidateKey]
    relationships: list[CandidateRelationship]
    limitations: list[str]
    processing_note: str
    file_byte_limit: int
    total_byte_limit: int


OperationName = Literal[
    "drop_exact_duplicates",
    "trim_whitespace",
    "whitespace_to_missing",
    "drop_empty_rows",
    "fill_constant",
    "fill_median",
    "change_case",
    "convert_type",
    "rename_columns",
]


class OperationSpec(BaseModel):
    """One cleaning request. The fill constant stays in memory for replay."""

    model_config = ConfigDict(extra="forbid")

    operation: OperationName
    dataset_id: str
    columns: list[str] = Field(default_factory=list)
    fill_value: str | None = None
    case: Literal["upper", "lower", "title"] | None = None
    target_type: Literal["numeric", "date", "text"] | None = None
    date_format: str | None = None
    renames: dict[str, str] = Field(default_factory=dict)


class PreviewResult(BaseModel):
    """Counts for a proposed operation. The working copy is unchanged."""

    model_config = ConfigDict(extra="forbid")

    applicable: bool
    block_reason: str | None = None
    dataset_id: str
    operation: str
    columns: list[str] = Field(default_factory=list)
    rows_before: int = Field(default=0, ge=0)
    rows_after: int = Field(default=0, ge=0)
    rows_removed: int = Field(default=0, ge=0)
    cells_changed: int = Field(default=0, ge=0)
    conversion_failures: int = Field(default=0, ge=0)
    columns_skipped: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MetricSnapshot(BaseModel):
    """Quality counts for one dataset. Cell values are not included."""

    model_config = ConfigDict(extra="forbid")

    data_rows: int = Field(ge=0)
    rows_in_duplicate_groups: int = Field(ge=0)
    extra_duplicate_copies: int = Field(ge=0)
    missing_cells: int = Field(ge=0)
    missing_by_column: dict[str, int]


class RelationshipSnapshot(BaseModel):
    """A suggested relationship reduced to names and counts."""

    model_config = ConfigDict(extra="forbid")

    parent_dataset: str
    parent_column: str
    child_dataset: str
    child_column: str
    cardinality: Literal["one-to-one", "one-to-many"]
    orphan_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    overlap_ratio: float


class AuditEntry(BaseModel):
    """One approved operation. Fill constants and cell values are omitted."""

    model_config = ConfigDict(extra="forbid")

    action_order: int = Field(ge=1)
    timestamp: str
    dataset_id: str
    source_name: str
    sheet_name: str | None = None
    operation: str
    columns: list[str]
    parameter_summary: str
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
    rows_removed: int = Field(ge=0)
    cells_changed: int = Field(ge=0)
    conversion_failures: int = Field(ge=0)
    columns_skipped: list[str]
    key_warning: str
    metrics_before: MetricSnapshot
    metrics_after: MetricSnapshot
    relationships_before: list[RelationshipSnapshot]
    relationships_after: list[RelationshipSnapshot]
