"""Counts and finding rows for an uploaded batch.

These rows are not banking findings. They have no severity, score, or
business-impact field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from universal_dq.models import BatchReport
from universal_dq.recommendations import Recommendation, build_recommendations

EMPTY_UPLOAD_MESSAGE = (
    "No uploaded file is available in this session. "
    "Open Analyze a file and choose a CSV or Excel file. "
    "These pages stay empty until a file is uploaded."
)
LOAD_FAILURE_MESSAGE = (
    "The uploaded files did not produce an analyzed dataset. "
    "Open Analyze a file to review the file error."
)

FindingKind = Literal["observation", "user_rule", "inferred_mismatch", "recommendation"]
KIND_LABELS = {
    "observation": "Observation",
    "user_rule": "User-selected rule",
    "inferred_mismatch": "Inferred mismatch",
    "recommendation": "Recommendation",
}


@dataclass(frozen=True)
class UploadOverview:
    """Measured totals for one version of the upload."""

    datasets_analyzed: int
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    extra_copies: int
    candidate_keys: int
    relationship_suggestions: int
    unmatched_references: int


@dataclass(frozen=True)
class UploadFinding:
    """One uploaded-data row. Evidence is a count, not a cell value."""

    dataset_id: str
    column: str | None
    kind: FindingKind
    issue: str
    evidence: str
    affected_count: int
    denominator: int


def upload_overview(batch: BatchReport) -> UploadOverview:
    """Sum confirmed counts. Keys and relationships stay suggestions."""
    reports = [item.report for item in batch.datasets if item.report is not None]
    return UploadOverview(
        datasets_analyzed=len(reports),
        rows=sum(report.data_rows for report in reports),
        columns=sum(report.column_count for report in reports),
        missing_values=sum(column.missing.count for report in reports for column in report.columns),
        duplicate_rows=sum(report.exact_duplicates.rows_in_duplicate_groups.count for report in reports),
        extra_copies=sum(report.exact_duplicates.extra_copies.count for report in reports),
        candidate_keys=len(batch.candidate_keys),
        relationship_suggestions=len(batch.relationships),
        unmatched_references=sum(item.orphan_count for item in batch.relationships),
    )


def upload_findings(batch: BatchReport, frames: dict) -> list[UploadFinding]:
    """List observations, selected-rule violations, orphan suggestions, and recommendations."""
    rows: list[UploadFinding] = []
    for item in batch.datasets:
        report = item.report
        if report is None:
            continue
        duplicates = report.exact_duplicates
        if duplicates.extra_copies.count or duplicates.rows_in_duplicate_groups.count:
            rows.append(
                UploadFinding(
                    dataset_id=item.dataset_id,
                    column=None,
                    kind="observation",
                    issue="Exact duplicates",
                    evidence=(
                        f"{duplicates.extra_copies.count} extra exact copies. "
                        f"{duplicates.rows_in_duplicate_groups.count} data rows sit in an exact duplicate group. "
                        f"The denominator is {report.data_rows} data rows."
                    ),
                    affected_count=duplicates.extra_copies.count,
                    denominator=report.data_rows,
                )
            )
        for column in report.columns:
            if column.missing.count and column.missing_role == "observation":
                rows.append(
                    UploadFinding(
                        dataset_id=item.dataset_id,
                        column=column.name,
                        kind="observation",
                        issue="Missing values",
                        evidence=(
                            f"{column.missing.count} of {report.data_rows} data rows are missing. "
                            "Empty cells and whitespace-only text are missing."
                        ),
                        affected_count=column.missing.count,
                        denominator=report.data_rows,
                    )
                )
            if column.missing_role == "user_rule_violation":
                rows.append(
                    UploadFinding(
                        dataset_id=item.dataset_id,
                        column=column.name,
                        kind="user_rule",
                        issue="Required column",
                        evidence=(
                            f"{column.missing.count} of {report.data_rows} data rows are missing "
                            "in a column you marked required."
                        ),
                        affected_count=column.missing.count,
                        denominator=report.data_rows,
                    )
                )
            if column.mixed_type is not None:
                listed = ", ".join(
                    f"{kind} {count}" for kind, count in sorted(column.mixed_type.categories.items())
                )
                rows.append(
                    UploadFinding(
                        dataset_id=item.dataset_id,
                        column=column.name,
                        kind="observation",
                        issue="Mixed types",
                        evidence=(
                            f"Possible mixed types among {column.mixed_type.non_missing_count} "
                            f"non-missing values: {listed}."
                        ),
                        affected_count=column.mixed_type.non_missing_count,
                        denominator=column.mixed_type.non_missing_count,
                    )
                )
            outlier = column.outlier
            if outlier is not None and outlier.outlier_count:
                rows.append(
                    UploadFinding(
                        dataset_id=item.dataset_id,
                        column=column.name,
                        kind="observation",
                        issue="Numerical outliers",
                        evidence=(
                            f"{outlier.outlier_count} of {outlier.denominator} non-missing numeric values "
                            "fall outside the interquartile-range fences."
                        ),
                        affected_count=int(outlier.outlier_count),
                        denominator=int(outlier.denominator or 0),
                    )
                )
        key = report.key_duplicates
        if key is not None and key.status == "violation":
            rows.append(
                UploadFinding(
                    dataset_id=item.dataset_id,
                    column=", ".join(key.key_columns),
                    kind="user_rule",
                    issue="Unique key",
                    evidence=(
                        f"{key.extra_copies.count} extra copies of a key you selected. "
                        f"{key.incomplete_key_rows} incomplete keys are excluded from that count."
                    ),
                    affected_count=key.extra_copies.count,
                    denominator=key.extra_copies.denominator or 0,
                )
            )
    for link in batch.relationships:
        if link.orphan_count <= 0:
            continue
        rows.append(
            UploadFinding(
                dataset_id=link.child_dataset,
                column=link.child_column,
                kind="inferred_mismatch",
                issue="Unmatched references",
                evidence=(
                    "Suggestion, not a confirmed foreign key. "
                    f"{link.orphan_count} non-missing values in {link.child_dataset} · {link.child_column} "
                    f"do not occur in {link.parent_dataset} · {link.parent_column}."
                ),
                affected_count=link.orphan_count,
                denominator=link.child_non_missing,
            )
        )
    for item in build_recommendations(batch, frames):
        rows.append(_recommendation_row(item))
    return rows


def _recommendation_row(item: Recommendation) -> UploadFinding:
    column = item.column or ""
    return UploadFinding(
        dataset_id=item.dataset_id,
        column=column or None,
        kind="recommendation",
        issue=item.issue_type.replace("_", " "),
        evidence=item.evidence,
        affected_count=item.affected_count,
        denominator=item.denominator,
    )
