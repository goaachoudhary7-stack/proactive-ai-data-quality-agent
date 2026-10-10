"""Tests for the read-only findings dashboard data layer.

These tests read outputs/dq_findings.json or temporary copies of it.
They do not read injected_issues.json, call a model, or start Streamlit.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ai_agents.findings_loader import load_findings_report  # noqa: E402
from ai_agents.investigation_schema import InvestigationReport  # noqa: E402
from ai_agents.schemas import Finding  # noqa: E402
from dashboard.data import (  # noqa: E402
    MISSING_INVESTIGATION_MESSAGE,
    available_report_count,
    count_by_dimension,
    count_by_severity,
    filter_findings,
    load_dashboard,
    load_investigation_view,
    metric_explanation,
)

FINDINGS_PATH = ROOT / "outputs" / "dq_findings.json"
DASHBOARD_DIR = SRC / "dashboard"


def _valid_report(finding: Finding) -> InvestigationReport:
    return InvestigationReport(
        rule_id=finding.rule_id,
        issue_summary="Summary of the failed check.",
        observed_evidence=finding,
        business_impact="Campaign reporting can misstate performance.",
        possible_root_causes=[
            {
                "hypothesis": "A source system omitted the campaign identifier.",
                "verification_step": "Compare the source extract with the warehouse row.",
            }
        ],
        recommended_investigation_steps=["Review the extract for this campaign type."],
        recommended_remediation=["Correct the source mapping and reload the rows."],
        confidence_level="low",
        limitations="The hypothesis is not a confirmed cause.",
    )


def _write_report(directory: Path, report: InvestigationReport) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.rule_id}_investigation.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def test_dashboard_loads_all_seven_findings() -> None:
    loaded = load_dashboard(FINDINGS_PATH, ROOT / "outputs" / "does_not_exist")
    assert loaded.findings_error is None
    assert loaded.report is not None
    assert loaded.report.checks_run == 7
    assert loaded.report.passed == 0
    assert loaded.report.warnings == 0
    assert loaded.report.failed == 7
    assert [item.rule_id for item in loaded.report.findings] == [
        "DQ001",
        "DQ002",
        "DQ003",
        "DQ004",
        "DQ005",
        "DQ006",
        "DQ007",
    ]
    assert available_report_count(loaded.investigations) == 0
    assert all(
        view.message == MISSING_INVESTIGATION_MESSAGE for view in loaded.investigations
    )


def test_filter_findings_combines_dataset_and_dimension() -> None:
    findings = list(load_findings_report(FINDINGS_PATH).findings)
    selected = filter_findings(
        findings,
        datasets=["customers"],
        dimensions=["UNIQUENESS"],
    )
    assert [item.rule_id for item in selected] == ["DQ001"]
    assert filter_findings(findings, datasets=[]) == []
    assert len(filter_findings(findings)) == 7


def test_counts_are_check_counts() -> None:
    findings = list(load_findings_report(FINDINGS_PATH).findings)
    assert sum(count for _, count in count_by_severity(findings)) == 7
    assert sum(count for _, count in count_by_dimension(findings)) == 7


def test_metric_explanation_keeps_non_rate_units() -> None:
    findings = {item.rule_id: item for item in load_findings_report(FINDINGS_PATH).findings}
    attribution = findings["DQ004"]
    freshness = findings["DQ006"]
    volume = findings["DQ007"]
    assert "of 39,401 records checked" in metric_explanation(attribution)
    assert "%" in metric_explanation(attribution)
    freshness_text = metric_explanation(freshness)
    assert "Metric 7" in freshness_text
    assert "%" not in freshness_text
    volume_text = metric_explanation(volume)
    assert "Metric -0.75333" in volume_text
    assert "%" not in volume_text


def test_available_report_must_match_the_original_finding(tmp_path: Path) -> None:
    finding = next(
        item
        for item in load_findings_report(FINDINGS_PATH).findings
        if item.rule_id == "DQ004"
    )
    _write_report(tmp_path, _valid_report(finding))
    view = load_investigation_view(finding, tmp_path)
    assert view.status == "available"
    assert view.report is not None
    assert view.report.observed_evidence.records_affected == 6736
    assert view.report.observed_evidence.metric_value == finding.metric_value


def test_missing_investigation_uses_the_required_message(tmp_path: Path) -> None:
    finding = load_findings_report(FINDINGS_PATH).findings[1]
    view = load_investigation_view(finding, tmp_path)
    assert view.status == "missing"
    assert view.report is None
    assert view.message == "AI investigation not yet generated"


def test_malformed_investigation_does_not_drop_other_findings(tmp_path: Path) -> None:
    findings = list(load_findings_report(FINDINGS_PATH).findings)
    valid = _valid_report(findings[0])
    _write_report(tmp_path, valid)
    broken = tmp_path / "DQ002_investigation.json"
    broken.write_text("{", encoding="utf-8")
    loaded = load_dashboard(FINDINGS_PATH, tmp_path)
    assert loaded.findings_error is None
    assert loaded.report is not None
    by_rule = {view.rule_id: view for view in loaded.investigations}
    assert by_rule["DQ001"].status == "available"
    assert by_rule["DQ002"].status == "rejected"
    assert by_rule["DQ002"].report is None
    assert "not valid JSON" in by_rule["DQ002"].message
    assert by_rule["DQ004"].message == MISSING_INVESTIGATION_MESSAGE
    assert available_report_count(loaded.investigations) == 1


def test_schema_invalid_investigation_is_rejected(tmp_path: Path) -> None:
    finding = load_findings_report(FINDINGS_PATH).findings[0]
    path = tmp_path / f"{finding.rule_id}_investigation.json"
    path.write_text(json.dumps({"rule_id": finding.rule_id}), encoding="utf-8")
    view = load_investigation_view(finding, tmp_path)
    assert view.status == "rejected"
    assert view.report is None
    assert "schema" in view.message


def test_mismatched_evidence_is_not_treated_as_verified(tmp_path: Path) -> None:
    finding = next(
        item
        for item in load_findings_report(FINDINGS_PATH).findings
        if item.rule_id == "DQ004"
    )
    changed = finding.model_copy(update={"metric_value": 0.0, "records_affected": 1})
    report = _valid_report(finding).model_copy(update={"observed_evidence": changed})
    _write_report(tmp_path, report)
    view = load_investigation_view(finding, tmp_path)
    assert view.status == "rejected"
    assert view.report is None
    assert "confirmed evidence" in view.message


def test_mismatched_rule_id_is_rejected(tmp_path: Path) -> None:
    findings = list(load_findings_report(FINDINGS_PATH).findings)
    other = _valid_report(findings[3])
    destination = tmp_path / "DQ001_investigation.json"
    destination.write_text(other.model_dump_json(indent=2), encoding="utf-8")
    view = load_investigation_view(findings[0], tmp_path)
    assert view.status == "rejected"
    assert view.report is None
    assert "rule id" in view.message


def test_malformed_findings_file_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "dq_findings.json"
    path.write_text("{", encoding="utf-8")
    loaded = load_dashboard(path, tmp_path)
    assert loaded.report is None
    assert loaded.findings_error is not None
    assert loaded.investigations == ()


def test_missing_findings_file_does_not_raise(tmp_path: Path) -> None:
    loaded = load_dashboard(tmp_path / "missing.json", tmp_path)
    assert loaded.report is None
    assert loaded.findings_error is not None
    assert "not found" in loaded.findings_error


def test_dashboard_source_does_not_read_secrets_or_the_manifest() -> None:
    forbidden = ("injected_issues", "GEMINI_API_KEY", "OPENAI_API_KEY", "getenv", "environ")
    for path in DASHBOARD_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for word in forbidden:
                    assert word not in node.value
