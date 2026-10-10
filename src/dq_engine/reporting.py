"""Console summary and JSON export for DQ findings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import DQFinding


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def findings_payload(findings: list[DQFinding]) -> dict[str, Any]:
    return {
        "checks_run": len(findings),
        "passed": sum(1 for item in findings if item.status == "PASS"),
        "warnings": sum(1 for item in findings if item.status == "WARNING"),
        "failed": sum(1 for item in findings if item.status == "FAIL"),
        "findings": [_json_safe(item.to_dict()) for item in findings],
    }


def write_findings(findings: list[DQFinding], output_path: Path) -> None:
    """Write structured findings. This file is engine output, not ground truth."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = findings_payload(findings)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def format_console_summary(findings: list[DQFinding]) -> str:
    lines = [
        "=================================================",
        "PROACTIVE DATA QUALITY ENGINE",
        "=================================================",
        "",
    ]
    for item in findings:
        lines.extend(
            [
                f"{item.rule_id} | {item.check_name}",
                f"Status: {item.status}",
                f"Affected: {item.records_affected:,} of {item.records_checked:,} checked",
                f"Metric: {item.metric_value}",
                f"Threshold: {item.threshold}",
                "",
            ]
        )

    passed = sum(1 for item in findings if item.status == "PASS")
    warnings = sum(1 for item in findings if item.status == "WARNING")
    failed = sum(1 for item in findings if item.status == "FAIL")
    lines.extend(
        [
            "=================================================",
            "SUMMARY",
            "=================================================",
            "",
            f"Checks run: {len(findings)}",
            f"Passed: {passed}",
            f"Warnings: {warnings}",
            f"Failed: {failed}",
        ]
    )
    return "\n".join(lines)


def print_console_summary(findings: list[DQFinding]) -> None:
    print(format_console_summary(findings))
