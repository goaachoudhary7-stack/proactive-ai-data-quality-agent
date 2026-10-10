"""Tests for the Phase 2 deterministic Data Quality Engine.

Assertions recalculate metrics from the corrupted CSVs. They do not read
injected_issues.json and they do not hard-code injected counts or dates.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dq_engine.checks import run_all_checks  # noqa: E402
from dq_engine.engine import load_config, load_corrupted_datasets, run_engine  # noqa: E402
from dq_engine.models import DQFinding  # noqa: E402
from dq_engine.reporting import findings_payload, write_findings  # noqa: E402

CORRUPTED_DIR = ROOT / "data" / "corrupted"
CONFIG_PATH = ROOT / "config" / "data_config.yaml"
DQ_ENGINE_DIR = SRC / "dq_engine"
OUTPUT_PATH = ROOT / "outputs" / "dq_findings.json"

FINDING_FIELDS = {
    "rule_id",
    "check_name",
    "dataset",
    "field",
    "dimension",
    "status",
    "severity",
    "records_checked",
    "records_affected",
    "metric_value",
    "threshold",
    "description",
    "evidence",
}

EXPECTED_RULES = ["DQ001", "DQ002", "DQ003", "DQ004", "DQ005", "DQ006", "DQ007"]


def _missing(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return series.isna() | text.str.strip().eq("") | text.str.lower().eq("nan")


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def datasets() -> dict[str, pd.DataFrame]:
    return load_corrupted_datasets(CORRUPTED_DIR)


@pytest.fixture(scope="module")
def findings(datasets: dict[str, pd.DataFrame], config: dict) -> list[DQFinding]:
    return run_all_checks(datasets, config)


def _by_id(findings: list[DQFinding], rule_id: str) -> DQFinding:
    matches = [item for item in findings if item.rule_id == rule_id]
    assert len(matches) == 1
    return matches[0]


def test_all_seven_checks_execute(findings: list[DQFinding]) -> None:
    assert [item.rule_id for item in findings] == EXPECTED_RULES


def test_findings_follow_schema(findings: list[DQFinding]) -> None:
    for item in findings:
        assert isinstance(item, DQFinding)
        payload = item.to_dict()
        assert set(payload) == FINDING_FIELDS
        assert item.status in {"PASS", "WARNING", "FAIL"}
        assert item.severity in {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert isinstance(item.records_checked, int)
        assert isinstance(item.records_affected, int)
        assert isinstance(item.metric_value, float)
        assert isinstance(item.threshold, float)
        assert isinstance(item.evidence, dict)


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Ids of module, function, and class docstring nodes.

    Comments never appear in the AST. The first string expression in a body
    is the docstring, so both that expression and its constant are skipped.
    """
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


def test_engine_does_not_depend_on_injected_issues_json() -> None:
    paths = sorted(DQ_ENGINE_DIR.glob("*.py"))
    assert paths, "expected dq_engine modules to scan"
    for path in paths:
        references = _executable_manifest_references(path.read_text(encoding="utf-8"))
        assert references == [], (
            f"{path.name} references injected_issues in executable code: {references}"
        )


def test_dq001_detects_duplicate_customer_ids(
    findings: list[DQFinding], datasets: dict[str, pd.DataFrame]
) -> None:
    customers = datasets["customers"]
    total_rows = len(customers)
    unique_ids = customers["customer_id"].nunique(dropna=True)
    excess = total_rows - unique_ids
    finding = _by_id(findings, "DQ001")

    assert excess > 0
    assert finding.records_checked == total_rows
    assert finding.records_affected == excess
    assert finding.evidence["unique_customer_ids"] == unique_ids
    assert finding.evidence["duplicate_excess_records"] == excess
    assert finding.metric_value == pytest.approx(excess / total_rows)
    assert finding.status == "FAIL"


def test_dq002_detects_missing_customer_ids(
    findings: list[DQFinding], datasets: dict[str, pd.DataFrame]
) -> None:
    interactions = datasets["interactions"]
    missing = int(_missing(interactions["customer_id"]).sum())
    finding = _by_id(findings, "DQ002")

    assert missing > 0
    assert finding.records_checked == len(interactions)
    assert finding.records_affected == missing
    assert finding.metric_value == pytest.approx(missing / len(interactions))
    assert finding.status == "FAIL"


def test_dq003_detects_broken_references(
    findings: list[DQFinding], datasets: dict[str, pd.DataFrame]
) -> None:
    interactions = datasets["interactions"]
    customers = datasets["customers"]
    has_id = ~_missing(interactions["customer_id"])
    valid_ids = set(customers["customer_id"].dropna().astype(str))
    checked = interactions.loc[has_id]
    broken = int((~checked["customer_id"].astype(str).isin(valid_ids)).sum())
    finding = _by_id(findings, "DQ003")

    assert broken > 0
    assert finding.records_checked == len(checked)
    assert finding.records_affected == broken
    assert finding.metric_value == pytest.approx(broken / len(checked))
    assert finding.status == "FAIL"


def test_dq004_distinguishes_eligible_campaign_interactions(
    findings: list[DQFinding], datasets: dict[str, pd.DataFrame], config: dict
) -> None:
    interactions = datasets["interactions"]
    types = config["dq_engine"]["campaign_driven_interaction_types"]
    eligible = interactions["interaction_type"].isin(types)
    missing_eligible = int(_missing(interactions.loc[eligible, "campaign_id"]).sum())
    missing_all = int(_missing(interactions["campaign_id"]).sum())
    finding = _by_id(findings, "DQ004")

    assert finding.records_checked == int(eligible.sum())
    assert finding.records_checked < len(interactions)
    assert finding.records_affected == missing_eligible
    assert missing_all > missing_eligible
    assert finding.evidence["non_eligible_null_campaign_id"] == missing_all - missing_eligible
    assert finding.status == "FAIL"


def test_dq005_detects_recent_vs_baseline_null_rate_difference(
    findings: list[DQFinding], datasets: dict[str, pd.DataFrame], config: dict
) -> None:
    customers = datasets["customers"]
    last_updated = pd.to_datetime(customers["last_updated"], errors="coerce")
    window_days = config["dq_thresholds"]["industry_recent_window_days"]
    window_start = last_updated.max() - pd.Timedelta(days=window_days - 1)
    recent = customers.loc[last_updated >= window_start]
    baseline = customers.loc[last_updated < window_start]
    recent_rate = int(_missing(recent["industry"]).sum()) / len(recent)
    baseline_rate = int(_missing(baseline["industry"]).sum()) / len(baseline)
    finding = _by_id(findings, "DQ005")

    assert recent_rate > baseline_rate
    assert finding.metric_value == pytest.approx(recent_rate - baseline_rate)
    assert finding.evidence["recent_null_rate"] == pytest.approx(recent_rate)
    assert finding.evidence["baseline_null_rate"] == pytest.approx(baseline_rate)
    assert finding.status == "FAIL"


def test_dq006_uses_other_datasets_not_system_time(
    findings: list[DQFinding], datasets: dict[str, pd.DataFrame]
) -> None:
    engine_source = (DQ_ENGINE_DIR / "checks.py").read_text(encoding="utf-8")
    assert "datetime.now" not in engine_source
    assert "Timestamp.now" not in engine_source
    assert "pd.Timestamp.today" not in engine_source

    opp_max = pd.to_datetime(datasets["opportunities"]["last_updated"], errors="coerce").max()
    reference = max(
        pd.to_datetime(datasets["customers"]["last_updated"], errors="coerce").max(),
        pd.to_datetime(datasets["campaigns"]["last_updated"], errors="coerce").max(),
        pd.to_datetime(datasets["interactions"]["last_updated"], errors="coerce").max(),
    )
    lag_days = float((reference - opp_max).days)
    finding = _by_id(findings, "DQ006")

    assert lag_days > 0
    assert finding.metric_value == pytest.approx(lag_days)
    assert finding.evidence["opportunity_max_last_updated"] == str(pd.Timestamp(opp_max).date())
    assert finding.evidence["reference_timestamp"] == str(pd.Timestamp(reference).date())
    assert finding.status == "FAIL"


def test_dq007_identifies_abnormal_daily_volume_drop(
    findings: list[DQFinding], datasets: dict[str, pd.DataFrame]
) -> None:
    dates = pd.to_datetime(datasets["interactions"]["interaction_date"], errors="coerce")
    daily = dates.groupby(dates.dt.normalize()).size()
    finding = _by_id(findings, "DQ007")
    evidence = finding.evidence
    affected = pd.Timestamp(evidence["affected_date"])

    assert affected in daily.index
    assert evidence["actual_volume"] == int(daily.loc[affected])
    assert evidence["actual_volume"] < evidence["baseline_volume"]
    assert evidence["percentage_deviation"] < 0
    assert finding.status == "FAIL"


def test_findings_json_serializes_and_parses(
    findings: list[DQFinding], tmp_path: Path
) -> None:
    temp_file = tmp_path / "dq_findings.json"
    write_findings(findings, temp_file)
    loaded = json.loads(temp_file.read_text(encoding="utf-8"))
    assert loaded["checks_run"] == 7
    assert len(loaded["findings"]) == 7
    assert findings_payload(findings)["failed"] == loaded["failed"]

    run_engine(CORRUPTED_DIR, CONFIG_PATH, OUTPUT_PATH)
    production = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    assert production["checks_run"] == 7
    json.loads(json.dumps(production))


def test_config_contains_phase2_thresholds() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "duplicate_customer_rate",
        "missing_customer_id_rate",
        "broken_reference_rate",
        "campaign_attribution_missing_rate",
        "industry_recent_window_days",
        "industry_null_rate_difference",
        "opportunity_freshness_days",
        "interaction_volume_drop_percentage",
    }
    assert required.issubset(config["dq_thresholds"])
