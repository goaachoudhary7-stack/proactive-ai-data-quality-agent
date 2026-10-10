"""Structured, JSON-serializable result of one data-quality check."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DQFinding:
    """One deterministic quality-check result."""

    rule_id: str
    check_name: str
    dataset: str
    field: str
    dimension: str
    status: str
    severity: str
    records_checked: int
    records_affected: int
    metric_value: float
    threshold: float
    description: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict that json.dump can serialize."""
        return asdict(self)
