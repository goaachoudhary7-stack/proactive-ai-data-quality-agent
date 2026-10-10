"""Deterministic Data Quality Engine.

These checks establish facts from the corrupted CSV files using pandas rules.
They do not read injected_issues.json and they do not use AI.
"""

from .models import DQFinding

__all__ = ["DQFinding"]
