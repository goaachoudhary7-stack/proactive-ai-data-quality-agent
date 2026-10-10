"""Tests for the Phase 3B DQ004 investigator.

The model is mocked. These tests do not call a paid API and do not read the
corruption manifest.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ai_agents.findings_loader import (  # noqa: E402
    FindingSelectionError,
    load_findings_report,
    select_finding,
)
from ai_agents.investigation_schema import (  # noqa: E402
    InvestigationReport,
    RootCauseHypothesis,
)
from ai_agents.investigator import (  # noqa: E402
    AGENT_ROLE,
    InvestigationOutputError,
    InvestigatorConfigError,
    agent_runtime_limits,
    assemble_report,
    build_task_description,
    crew_runtime_limits,
    ensure_report_matches_finding,
    investigate_finding,
    parse_narrative,
    resolve_llm_settings,
    run_investigation,
    run_investigations,
    write_investigation_report,
)
from ai_agents import run_investigation as cli  # noqa: E402
from ai_agents import investigator  # noqa: E402

FINDINGS_PATH = ROOT / "outputs" / "dq_findings.json"
AI_AGENTS_DIR = SRC / "ai_agents"


def _dq004():
    return select_finding(load_findings_report(), "DQ004")


def _narrative() -> dict:
    return {
        "issue_summary": (
            "Campaign-driven interactions are missing campaign attribution "
            "above the configured threshold."
        ),
        "business_impact": (
            "Campaign response reporting in this synthetic portfolio can "
            "under-count attributed email and event engagement."
        ),
        "possible_root_causes": [
            {
                "hypothesis": (
                    "The interaction extract dropped campaign_id on some "
                    "email and event rows."
                ),
                "verification_step": (
                    "Group the eligible interaction types and compare the "
                    "missing campaign_id rate for each type."
                ),
            }
        ],
        "recommended_investigation_steps": [
            "Recount missing campaign_id values for Email Open, Email Click, and Event Attendance."
        ],
        "recommended_remediation": [
            "Correct the attribution feed for those interaction types, then rerun the data-quality check."
        ],
        "confidence_level": "low",
        "limitations": (
            "Root causes are hypotheses, not confirmed facts. "
            "This report uses one validated synthetic finding and did not access a live banking system."
        ),
    }


def test_select_finding_by_rule_id() -> None:
    report = load_findings_report()

    selected = select_finding(report, "DQ004")

    assert selected.rule_id == "DQ004"
    assert selected.check_name == "Campaign Attribution Completeness"
    assert selected.records_checked == 39401
    assert selected.records_affected == 6736


def test_unknown_rule_id_is_rejected() -> None:
    report = load_findings_report()

    with pytest.raises(FindingSelectionError, match="DQ999"):
        select_finding(report, "DQ999")


def test_dq004_is_passed_to_the_investigator() -> None:
    finding = _dq004()
    seen = []

    def source(selected):
        seen.append(selected)
        return _narrative()

    report = investigate_finding(finding, source)

    assert seen == [finding]
    assert report.rule_id == "DQ004"
    assert isinstance(report, InvestigationReport)


def test_investigator_output_matches_schema(tmp_path: Path) -> None:
    destination = tmp_path / "DQ004_investigation.json"

    written = run_investigation(
        "DQ004",
        output_path=destination,
        narrative_source=lambda _finding: _narrative(),
    )

    payload = json.loads(written.read_text(encoding="utf-8"))
    restored = InvestigationReport.model_validate(payload)
    assert restored.rule_id == "DQ004"
    assert restored.confidence_level == "low"
    assert isinstance(restored.possible_root_causes[0], RootCauseHypothesis)
    assert restored.possible_root_causes[0].verification_step


def test_observed_evidence_is_copied_from_the_finding() -> None:
    finding = _dq004()
    source = finding.model_copy(deep=True)
    original_rate = source.evidence["missing_attribution_rate"]

    report = assemble_report(source, parse_narrative(_narrative()))
    source.evidence["missing_attribution_rate"] = 0
    source.metric_value = 0

    evidence = report.observed_evidence
    assert evidence.model_dump() == finding.model_dump()
    assert evidence.rule_id == "DQ004"
    assert evidence.records_checked == finding.records_checked
    assert evidence.records_affected == finding.records_affected
    assert evidence.metric_value == finding.metric_value
    assert evidence.threshold == finding.threshold
    assert evidence.evidence["missing_attribution_rate"] == original_rate


def test_missing_required_output_fields_are_rejected(tmp_path: Path) -> None:
    incomplete = _narrative()
    del incomplete["business_impact"]
    destination = tmp_path / "DQ004_investigation.json"

    with pytest.raises(InvestigationOutputError, match="business_impact"):
        run_investigation(
            "DQ004",
            output_path=destination,
            narrative_source=lambda _finding: incomplete,
        )

    assert not destination.exists()


def test_hypothesis_without_verification_step_is_rejected(tmp_path: Path) -> None:
    incomplete = _narrative()
    del incomplete["possible_root_causes"][0]["verification_step"]
    destination = tmp_path / "DQ004_investigation.json"

    with pytest.raises(InvestigationOutputError, match="verification_step"):
        run_investigation(
            "DQ004",
            output_path=destination,
            narrative_source=lambda _finding: incomplete,
        )

    assert not destination.exists()


def test_mismatched_rule_id_is_rejected() -> None:
    finding = _dq004()
    report = assemble_report(finding, parse_narrative(_narrative()))
    payload = report.model_dump()
    payload["rule_id"] = "DQ001"

    with pytest.raises(ValidationError, match="does not match"):
        InvestigationReport.model_validate(payload)


def test_original_findings_file_is_unchanged(tmp_path: Path) -> None:
    before = FINDINGS_PATH.read_bytes()

    run_investigation(
        "DQ004",
        findings_path=FINDINGS_PATH,
        output_path=tmp_path / "DQ004_investigation.json",
        narrative_source=lambda _finding: _narrative(),
    )

    assert FINDINGS_PATH.read_bytes() == before


def test_refuses_to_overwrite_findings_file() -> None:
    before = FINDINGS_PATH.read_bytes()

    with pytest.raises(InvestigationOutputError, match="findings file"):
        run_investigation(
            "DQ004",
            findings_path=FINDINGS_PATH,
            output_path=FINDINGS_PATH,
            narrative_source=lambda _finding: _narrative(),
        )

    assert FINDINGS_PATH.read_bytes() == before


def _docstring_node_ids(tree: ast.AST) -> set[int]:
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


def test_investigator_does_not_reference_corruption_manifest() -> None:
    for path in sorted(AI_AGENTS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skipped = _docstring_node_ids(tree)
        for node in ast.walk(tree):
            if id(node) in skipped:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "injected_issues" not in node.value, path.name
        source = path.read_text(encoding="utf-8")
        assert ".csv" not in source, path.name


def test_prompt_contains_finding_metrics_and_not_csv_files() -> None:
    finding = _dq004()
    text = build_task_description(finding)

    assert str(finding.records_checked) in text
    assert str(finding.records_affected) in text
    assert finding.rule_id in text
    assert ".csv" not in text
    assert "data/clean" not in text
    assert "data/corrupted" not in text
    assert "injected_issues" not in text


def test_unrecovered_api_failure_does_not_write_or_overwrite(tmp_path: Path) -> None:
    """A raised model error must not create a report or replace a valid one.

    CrewAI may retry a 503 and then return a narrative. This test is the case
    where that retry is already exhausted and the call still fails.
    """
    destination = tmp_path / "DQ004_investigation.json"

    def fail(_finding):
        raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(RuntimeError, match="503 UNAVAILABLE"):
        run_investigation("DQ004", output_path=destination, narrative_source=fail)
    assert not destination.exists()

    written = run_investigation(
        "DQ004",
        output_path=destination,
        narrative_source=lambda _finding: _narrative(),
    )
    before = written.read_bytes()
    InvestigationReport.model_validate_json(before)

    with pytest.raises(RuntimeError, match="503 UNAVAILABLE"):
        run_investigation("DQ004", output_path=destination, narrative_source=fail)

    assert destination.read_bytes() == before


def test_mocked_narrative_does_not_call_the_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("The model client was called.")

    monkeypatch.setattr(investigator, "execute_crew", fail_if_called)

    written = run_investigation(
        "DQ004",
        output_path=tmp_path / "DQ004_investigation.json",
        narrative_source=lambda _finding: _narrative(),
    )

    assert written.is_file()


def test_agent_has_no_tools_and_cannot_execute_code() -> None:
    limits = agent_runtime_limits()
    crew_limits = crew_runtime_limits()

    assert limits["role"] == AGENT_ROLE
    assert limits["tools"] == []
    assert limits["allow_delegation"] is False
    assert limits["allow_code_execution"] is False
    assert limits["memory"] is False
    assert crew_limits["name"] == "Data Quality Investigator"
    assert crew_limits["memory"] is False
    assert crew_limits["tracing"] is False


def test_missing_model_raises_clear_error() -> None:
    with pytest.raises(InvestigatorConfigError, match="DQ_INVESTIGATOR_MODEL"):
        resolve_llm_settings({})


def test_missing_api_key_raises_clear_error() -> None:
    with pytest.raises(InvestigatorConfigError, match="OPENAI_API_KEY"):
        resolve_llm_settings({"DQ_INVESTIGATOR_MODEL": "openai/gpt-4o"})


def test_blank_api_key_raises_clear_error() -> None:
    with pytest.raises(InvestigatorConfigError, match="OPENAI_API_KEY"):
        resolve_llm_settings(
            {"DQ_INVESTIGATOR_MODEL": "openai/gpt-4o", "OPENAI_API_KEY": "   "}
        )


def test_missing_credentials_do_not_call_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DQ_INVESTIGATOR_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(investigator, "_load_dotenv", lambda: None)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("The model client was called.")

    monkeypatch.setattr(investigator, "execute_crew", fail_if_called)
    destination = tmp_path / "DQ004_investigation.json"

    with pytest.raises(InvestigatorConfigError, match="DQ_INVESTIGATOR_MODEL"):
        run_investigation(output_path=destination)

    assert not destination.exists()


def test_cli_runs_dq004_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen = {}

    def fake_run(rule_id: str = "DQ001") -> Path:
        seen["rule_id"] = rule_id
        return tmp_path / "DQ004_investigation.json"

    monkeypatch.setattr(cli, "run_investigation", fake_run)

    cli.main([])

    assert seen["rule_id"] == "DQ004"
    assert "DQ004_investigation.json" in capsys.readouterr().out


def test_module_entrypoint_imports_without_running() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.ai_agents.run_investigation as cli; raise SystemExit(0 if cli.DQ004 == 'DQ004' else 1)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_cli_rule_id_selects_that_finding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = {}

    def fake_run(rule_id: str = "DQ004") -> Path:
        seen["rule_id"] = rule_id
        return Path("outputs/ai_investigations/DQ002_investigation.json")

    monkeypatch.setattr(cli, "run_investigation", fake_run)

    cli.main(["--rule-id", "DQ002"])

    assert seen["rule_id"] == "DQ002"
    assert "DQ002_investigation.json" in capsys.readouterr().out


def test_rule_id_report_keeps_that_findings_evidence(tmp_path: Path) -> None:
    finding = select_finding(load_findings_report(), "DQ002")
    destination = tmp_path / "DQ002_investigation.json"

    run_investigation(
        "DQ002",
        output_path=destination,
        narrative_source=lambda _finding: _narrative(),
    )

    restored = InvestigationReport.model_validate_json(destination.read_text(encoding="utf-8"))
    assert restored.rule_id == "DQ002"
    assert restored.observed_evidence.model_dump() == finding.model_dump()


def test_all_failed_runs_sequentially_and_skips_passes(tmp_path: Path) -> None:
    payload = json.loads(FINDINGS_PATH.read_text(encoding="utf-8"))
    for item in payload["findings"]:
        if item["rule_id"] == "DQ007":
            item["status"] = "PASS"
            item["severity"] = "INFO"
    payload["passed"] = 1
    payload["failed"] = 6
    findings_path = tmp_path / "findings.json"
    findings_path.write_text(json.dumps(payload), encoding="utf-8")
    seen: list[str] = []

    def source(finding):
        seen.append(finding.rule_id)
        return _narrative()

    attempts = run_investigations(
        all_failed=True,
        findings_path=findings_path,
        output_dir=tmp_path / "reports",
        narrative_source=source,
    )

    assert seen == ["DQ001", "DQ002", "DQ003", "DQ004", "DQ005", "DQ006"]
    assert [item.rule_id for item in attempts] == seen
    assert all(item.error is None for item in attempts)
    assert (tmp_path / "reports" / "DQ006_investigation.json").is_file()
    assert not (tmp_path / "reports" / "DQ007_investigation.json").exists()


def test_batch_continues_after_a_failure_and_cli_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    finding = select_finding(load_findings_report(), "DQ003")
    protected = output_dir / "DQ003_investigation.json"
    write_investigation_report(
        assemble_report(finding, parse_narrative(_narrative())),
        protected,
    )
    before = protected.read_bytes()

    def source(selected):
        if selected.rule_id == "DQ003":
            raise RuntimeError("503 UNAVAILABLE")
        return _narrative()

    attempts = run_investigations(
        all_failed=True,
        output_dir=output_dir,
        narrative_source=source,
    )

    by_id = {item.rule_id: item for item in attempts}
    assert by_id["DQ003"].error is not None
    assert "503 UNAVAILABLE" in by_id["DQ003"].error
    assert protected.read_bytes() == before
    assert by_id["DQ002"].error is None
    assert by_id["DQ004"].error is None
    assert (output_dir / "DQ004_investigation.json").is_file()

    monkeypatch.setattr(cli, "run_investigations", lambda **_kwargs: attempts)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--all-failed"])
    assert caught.value.code == 1
    summary = capsys.readouterr().out
    assert "DQ003: failed:" in summary
    assert "Investigated 7 findings: 6 written, 1 failed." in summary


def test_dry_run_lists_failures_without_credentials_or_api_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DQ_INVESTIGATOR_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(investigator, "_load_dotenv", lambda: None)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("The model client was called.")

    monkeypatch.setattr(investigator, "execute_crew", fail_if_called)
    monkeypatch.setattr(investigator, "resolve_llm_settings", fail_if_called)

    cli.main(["--all-failed", "--dry-run"])

    output = capsys.readouterr().out
    assert "Dry run. No model call will be made." in output
    assert "DQ001 FAIL customers -> outputs/ai_investigations/DQ001_investigation.json" in output
    assert "DQ004 FAIL interactions -> outputs/ai_investigations/DQ004_investigation.json" in output
    assert "DQ007 FAIL interactions -> outputs/ai_investigations/DQ007_investigation.json" in output
    assert "7 findings selected." in output


def test_dry_run_default_lists_only_dq004(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("The model client was called.")

    monkeypatch.setattr(investigator, "execute_crew", fail_if_called)
    monkeypatch.setattr(investigator, "resolve_llm_settings", fail_if_called)

    cli.main(["--dry-run"])

    output = capsys.readouterr().out
    assert "DQ004 FAIL interactions -> outputs/ai_investigations/DQ004_investigation.json" in output
    assert "1 finding selected." in output
    assert "DQ001" not in output


def test_rule_id_and_all_failed_cannot_be_combined() -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["--rule-id", "DQ001", "--all-failed"])

    assert caught.value.code == 2


def test_invalid_narrative_does_not_overwrite_a_valid_report(tmp_path: Path) -> None:
    destination = tmp_path / "DQ004_investigation.json"
    run_investigation(
        "DQ004",
        output_path=destination,
        narrative_source=lambda _finding: _narrative(),
    )
    before = destination.read_bytes()
    incomplete = _narrative()
    del incomplete["limitations"]

    with pytest.raises(InvestigationOutputError, match="limitations"):
        run_investigation(
            "DQ004",
            output_path=destination,
            narrative_source=lambda _finding: incomplete,
        )

    assert destination.read_bytes() == before


def test_mismatched_evidence_does_not_overwrite_a_valid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finding = _dq004()
    destination = tmp_path / "DQ004_investigation.json"
    valid = assemble_report(finding, parse_narrative(_narrative()))
    write_investigation_report(valid, destination)
    before = destination.read_bytes()
    tampered = valid.model_copy(
        update={"observed_evidence": finding.model_copy(update={"metric_value": 0.0})}
    )

    monkeypatch.setattr(investigator, "investigate_finding", lambda *_args, **_kwargs: tampered)

    with pytest.raises(InvestigationOutputError, match="does not match"):
        run_investigation(
            "DQ004",
            output_path=destination,
            narrative_source=lambda _finding: _narrative(),
        )

    assert destination.read_bytes() == before
    with pytest.raises(InvestigationOutputError, match="does not match"):
        ensure_report_matches_finding(tampered, finding)


def test_atomic_replace_keeps_the_previous_report_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "DQ004_investigation.json"
    report = assemble_report(_dq004(), parse_narrative(_narrative()))
    write_investigation_report(report, destination)
    before = destination.read_bytes()
    assert not destination.with_name(f"{destination.name}.tmp").exists()

    def fail_temporary_write(self: Path, *args, **kwargs):
        if self.name.endswith(".tmp"):
            raise OSError("disk full")
        raise AssertionError("the existing report was opened for a direct write")

    monkeypatch.setattr(Path, "write_text", fail_temporary_write)

    with pytest.raises(OSError, match="disk full"):
        write_investigation_report(report, destination)

    assert destination.read_bytes() == before
    assert not destination.with_name(f"{destination.name}.tmp").exists()
