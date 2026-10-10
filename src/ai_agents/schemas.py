"""Pydantic models for the Data Quality Engine report.

The JSON file has two levels:

1. A summary: how many checks ran, and how many passed, warned, or failed.
2. A list of findings: one object per check.

Evidence is a dictionary because each rule stores different keys. For example,
DQ001 stores duplicate IDs, while DQ006 stores freshness dates. Those keys are
kept as written; this model does not invent a single evidence layout.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Status = Literal["PASS", "WARNING", "FAIL"]
Severity = Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


class Finding(BaseModel):
    """One check result from outputs/dq_findings.json."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    check_name: str
    dataset: str
    field: str
    dimension: str
    status: Status
    severity: Severity
    records_checked: int
    records_affected: int
    metric_value: float
    threshold: float
    description: str
    evidence: dict[str, Any]


class FindingsReport(BaseModel):
    """The full report written by the Data Quality Engine."""

    model_config = ConfigDict(extra="forbid")

    checks_run: int
    passed: int
    warnings: int
    failed: int
    findings: list[Finding]
