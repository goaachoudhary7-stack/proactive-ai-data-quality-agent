"""Load and filter saved findings and investigation reports.

This module does not import Streamlit or CrewAI. It only reads JSON files.
It never writes datasets, findings, or investigation reports.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai_agents.findings_loader import (  # noqa: E402
    DEFAULT_FINDINGS_PATH,
    PROJECT_ROOT,
    FindingsFileError,
    load_findings_report,
)
from ai_agents.investigation_schema import InvestigationReport  # noqa: E402
from ai_agents.schemas import Finding, FindingsReport  # noqa: E402

DEFAULT_INVESTIGATIONS_DIR = PROJECT_ROOT / "outputs" / "ai_investigations"
MISSING_INVESTIGATION_MESSAGE = "AI investigation not yet generated"
SYNTHETIC_DATA_LABEL = (
    "Synthetic demonstration data for a fictional commercial bank. "
    "These records were generated for this project and do not describe real customers."
)

SEVERITY_ORDER = ("INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass(frozen=True)
class InvestigationView:
    """One finding's investigation file, after validation."""

    rule_id: str
    path: Path
    report: InvestigationReport | None
    status: str
    message: str


@dataclass(frozen=True)
class DashboardLoad:
    """Findings plus one investigation view per finding."""

    report: FindingsReport | None
    findings_error: str | None
    investigations: tuple[InvestigationView, ...]


def load_dashboard(
    findings_path: Path | None = None,
    investigations_dir: Path | None = None,
) -> DashboardLoad:
    """Read the findings file and every matching investigation file.

    A missing or malformed findings file is returned as an error message.
    A bad investigation file does not discard the other findings.
    """
    path = DEFAULT_FINDINGS_PATH if findings_path is None else findings_path
    directory = (
        DEFAULT_INVESTIGATIONS_DIR if investigations_dir is None else investigations_dir
    )
    try:
        report = load_findings_report(path)
    except FindingsFileError as exc:
        return DashboardLoad(None, str(exc), ())
    except ValidationError as exc:
        return DashboardLoad(
            None,
            "The findings file does not match the findings schema: " + str(exc),
            (),
        )
    except OSError as exc:
        return DashboardLoad(None, f"The findings file could not be read: {exc}", ())

    views = tuple(
        load_investigation_view(finding, directory) for finding in report.findings
    )
    return DashboardLoad(report, None, views)


def load_investigation_view(finding: Finding, directory: Path) -> InvestigationView:
    """Validate one investigation file against its original finding."""
    path = directory / f"{finding.rule_id}_investigation.json"
    if not directory.exists() or not path.is_file():
        return InvestigationView(
            finding.rule_id,
            path,
            None,
            "missing",
            MISSING_INVESTIGATION_MESSAGE,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = InvestigationReport.model_validate(payload)
    except json.JSONDecodeError as exc:
        return _rejected(
            finding.rule_id,
            path,
            f"This investigation file is not valid JSON: {exc.msg}",
        )
    except ValidationError as exc:
        return _rejected(
            finding.rule_id,
            path,
            "This investigation file does not match the investigation schema: "
            + _short_error(exc),
        )
    except OSError as exc:
        return _rejected(
            finding.rule_id,
            path,
            f"This investigation file could not be read: {exc}",
        )

    reason = mismatch_reason(report, finding)
    if reason is not None:
        return _rejected(finding.rule_id, path, reason)
    return InvestigationView(finding.rule_id, path, report, "available", "")


def mismatch_reason(report: InvestigationReport, finding: Finding) -> str | None:
    """Return an error when confirmed evidence differs from the original finding."""
    if report.rule_id != finding.rule_id:
        return (
            "This investigation was rejected because its rule id "
            f"({report.rule_id}) does not match the original finding "
            f"({finding.rule_id})."
        )
    if report.observed_evidence.model_dump() != finding.model_dump():
        return (
            "This investigation was rejected because its confirmed evidence "
            "does not match the original finding."
        )
    return None


def filter_findings(
    findings: list[Finding] | tuple[Finding, ...],
    *,
    datasets: list[str] | None = None,
    severities: list[str] | None = None,
    dimensions: list[str] | None = None,
    statuses: list[str] | None = None,
) -> list[Finding]:
    """Keep findings that match every active filter.

    ``None`` means that filter is not applied. An empty list matches nothing.
    """
    selected = list(findings)
    if datasets is not None:
        allowed = set(datasets)
        selected = [item for item in selected if item.dataset in allowed]
    if severities is not None:
        allowed = set(severities)
        selected = [item for item in selected if item.severity in allowed]
    if dimensions is not None:
        allowed = set(dimensions)
        selected = [item for item in selected if item.dimension in allowed]
    if statuses is not None:
        allowed = set(statuses)
        selected = [item for item in selected if item.status in allowed]
    return selected


def count_by_severity(findings: list[Finding] | tuple[Finding, ...]) -> list[tuple[str, int]]:
    """Count checks by severity. The values are counts, not a quality score."""
    counts = {name: 0 for name in SEVERITY_ORDER}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return [(name, counts[name]) for name in SEVERITY_ORDER if counts[name]]


def count_by_dimension(findings: list[Finding] | tuple[Finding, ...]) -> list[tuple[str, int]]:
    """Count checks by data-quality dimension."""
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.dimension] = counts.get(finding.dimension, 0) + 1
    return sorted(counts.items())


def available_report_count(views: tuple[InvestigationView, ...] | list[InvestigationView]) -> int:
    """Count investigation reports that validated and matched the original finding."""
    return sum(1 for view in views if view.status == "available" and view.report is not None)


def metric_explanation(finding: Finding) -> str:
    """Explain the stored metric without turning every number into a percentage.

    A percentage is shown only when the metric is the share of records checked
    (records affected divided by records checked). Day counts and volume
    deviations stay in the units stored by the detection engine.
    """
    affected = f"{finding.records_affected:,} of {finding.records_checked:,} records checked"
    if _is_share_of_records(finding):
        return (
            f"{affected} "
            f"({finding.metric_value * 100:.2f}%). "
            f"Threshold {finding.threshold * 100:.2f}%."
        )
    return (
        f"{affected}. "
        f"Metric {finding.metric_value:g}. "
        f"Threshold {finding.threshold:g}."
    )


def _is_share_of_records(finding: Finding) -> bool:
    if finding.records_checked <= 0:
        return False
    share = finding.records_affected / finding.records_checked
    return abs(finding.metric_value - share) <= 1e-12


def _rejected(rule_id: str, path: Path, message: str) -> InvestigationView:
    return InvestigationView(rule_id, path, None, "rejected", message)


def _short_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    detail = first.get("msg", "invalid value")
    if location:
        return f"{location}: {detail}"
    return str(detail)
