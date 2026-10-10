"""Tests for the Phase 3A findings loader.

These tests read outputs/dq_findings.json or temporary copies of it.
They do not read injected_issues.json.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ai_agents.findings_loader import (  # noqa: E402
    FindingsFileError,
    failed_findings,
    load_findings_report,
)
from ai_agents.schemas import Finding, FindingsReport  # noqa: E402

FINDINGS_PATH = ROOT / "outputs" / "dq_findings.json"
AI_AGENTS_DIR = SRC / "ai_agents"
EXPECTED_RULES = ["DQ001", "DQ002", "DQ003", "DQ004", "DQ005", "DQ006", "DQ007"]


def test_valid_report_loads() -> None:
    report = load_findings_report()

    assert isinstance(report, FindingsReport)
    assert report.checks_run == 7
    assert report.passed == 0
    assert report.warnings == 0
    assert report.failed == 7
    assert isinstance(report.findings[0], Finding)
    assert isinstance(report.findings[0].evidence, dict)
    assert report.findings[0].evidence


def test_seven_findings_are_recognized() -> None:
    report = load_findings_report()

    assert len(report.findings) == 7
    assert [item.rule_id for item in report.findings] == EXPECTED_RULES
    assert {item.dataset for item in report.findings} == {
        "customers",
        "interactions",
        "opportunities",
    }


def test_missing_file_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "dq_findings.json"

    with pytest.raises(FindingsFileError, match="not found"):
        load_findings_report(missing)


def test_malformed_json_raises_clear_error(tmp_path: Path) -> None:
    broken = tmp_path / "dq_findings.json"
    broken.write_text('{"checks_run": ', encoding="utf-8")

    with pytest.raises(FindingsFileError, match="not valid JSON"):
        load_findings_report(broken)


def test_missing_required_fields_are_rejected(tmp_path: Path) -> None:
    original = json.loads(FINDINGS_PATH.read_text(encoding="utf-8"))

    missing_summary = json.loads(json.dumps(original))
    del missing_summary["checks_run"]
    summary_path = tmp_path / "missing_summary.json"
    summary_path.write_text(json.dumps(missing_summary), encoding="utf-8")
    with pytest.raises(ValidationError) as summary_error:
        load_findings_report(summary_path)
    assert "checks_run" in str(summary_error.value)

    missing_finding = json.loads(json.dumps(original))
    del missing_finding["findings"][0]["rule_id"]
    finding_path = tmp_path / "missing_finding.json"
    finding_path.write_text(json.dumps(missing_finding), encoding="utf-8")
    with pytest.raises(ValidationError) as finding_error:
        load_findings_report(finding_path)
    assert "rule_id" in str(finding_error.value)


def test_failed_findings_keeps_only_fail_status() -> None:
    report = load_findings_report()
    current_failures = failed_findings(report)

    assert len(current_failures) == 7
    assert all(item.status == "FAIL" for item in current_failures)

    first = report.findings[0]
    mixed = FindingsReport(
        checks_run=3,
        passed=1,
        warnings=1,
        failed=1,
        findings=[
            first.model_copy(update={"rule_id": "DQ001", "status": "FAIL"}),
            first.model_copy(
                update={"rule_id": "DQ002", "status": "WARNING", "severity": "LOW"}
            ),
            first.model_copy(
                update={"rule_id": "DQ003", "status": "PASS", "severity": "INFO"}
            ),
        ],
    )

    assert [item.rule_id for item in failed_findings(mixed)] == ["DQ001"]


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Ids of module, function, and class docstring nodes."""
    skipped: set[int] = set()
    containers = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, containers):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        value = getattr(first, "value", None)
        if (
            isinstance(first, ast.Expr)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            skipped.add(id(first))
            skipped.add(id(value))
    return skipped


def _executable_manifest_references(source: str) -> list[str]:
    """Return executable names and strings that mention the ground-truth manifest."""
    tree = ast.parse(source)
    skipped = _docstring_node_ids(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        if id(node) in skipped:
            continue
        fragments: list[str] = []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            fragments.append(node.value)
        elif isinstance(node, ast.Name):
            fragments.append(node.id)
        elif isinstance(node, ast.Attribute):
            fragments.append(node.attr)
        elif isinstance(node, ast.alias):
            fragments.append(node.name)
            if node.asname:
                fragments.append(node.asname)
        elif isinstance(node, ast.ImportFrom) and node.module:
            fragments.append(node.module)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            fragments.append(node.name)
        elif isinstance(node, ast.arg):
            fragments.append(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            fragments.append(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            fragments.extend(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            fragments.append(node.name)
        for fragment in fragments:
            if "injected_issues" in fragment:
                hits.append(fragment)
    return hits


def test_loader_does_not_depend_on_injected_issues_json() -> None:
    paths = sorted(AI_AGENTS_DIR.glob("*.py"))
    assert paths, "expected ai_agents modules to scan"
    for path in paths:
        references = _executable_manifest_references(path.read_text(encoding="utf-8"))
        assert references == [], (
            f"{path.name} references injected_issues in executable code: {references}"
        )
