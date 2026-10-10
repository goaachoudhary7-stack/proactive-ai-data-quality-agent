"""Load outputs/dq_findings.json and check that it matches the report schema.

File problems (missing path, unreadable text, broken JSON) raise
FindingsFileError. A file that is valid JSON but missing a required field
raises pydantic.ValidationError, which names the field.

This module reads the engine report only. It never opens the ground-truth
manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import Finding, FindingsReport

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FINDINGS_PATH = PROJECT_ROOT / "outputs" / "dq_findings.json"


class FindingsFileError(Exception):
    """The findings file cannot be read as JSON."""


def load_findings_report(path: Path | str | None = None) -> FindingsReport:
    """Read a findings report and validate it.

    Args:
        path: JSON file to read. Defaults to outputs/dq_findings.json.

    Returns:
        A FindingsReport with summary counts and a list of findings.

    Raises:
        FindingsFileError: The path is missing, is not a file, or is not JSON.
        ValidationError: The JSON object is missing a required field or has
            a value of the wrong type.
    """
    report_path = DEFAULT_FINDINGS_PATH if path is None else Path(path)

    if not report_path.exists():
        raise FindingsFileError(
            f"Findings file not found: {report_path}. "
            "Run the Data Quality Engine first: python src/dq_engine/engine.py"
        )
    if not report_path.is_file():
        raise FindingsFileError(f"Findings path is not a file: {report_path}.")

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FindingsFileError(
            f"{report_path} is not valid JSON: {exc.msg} "
            f"at line {exc.lineno}, column {exc.colno}."
        ) from exc

    return FindingsReport.model_validate(payload)


def failed_findings(report: FindingsReport) -> list[Finding]:
    """Return findings whose status is FAIL.

    PASS and WARNING rows stay in the full report. This helper is the list an
    investigator would look at first.
    """
    return [item for item in report.findings if item.status == "FAIL"]


class FindingSelectionError(Exception):
    """The report has no single finding for the requested rule id."""


def select_finding(report: FindingsReport, rule_id: str) -> Finding:
    """Return the one finding with this rule id.

    Args:
        report: A validated findings report.
        rule_id: Check id such as "DQ004".

    Raises:
        FindingSelectionError: The id is missing, or more than one row uses it.
    """
    matches = [item for item in report.findings if item.rule_id == rule_id]
    if not matches:
        available = ", ".join(item.rule_id for item in report.findings) or "(none)"
        raise FindingSelectionError(
            f"No finding with rule_id {rule_id!r}. Available rule ids: {available}."
        )
    if len(matches) > 1:
        raise FindingSelectionError(
            f"Expected one finding for rule_id {rule_id!r}, found {len(matches)}."
        )
    return matches[0]
