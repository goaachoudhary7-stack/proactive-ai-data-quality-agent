"""One CrewAI investigator for a single validated data-quality finding.

The crew has one agent and one task. It receives the finding JSON only.
It does not read data files, and it does not receive tools, code execution,
or permission to delegate.

Confirmed numbers are copied from the finding in this module. The model
fills InvestigationNarrative, which has no metric fields.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .findings_loader import (
    DEFAULT_FINDINGS_PATH,
    PROJECT_ROOT,
    failed_findings,
    load_findings_report,
    select_finding,
)
from .investigation_schema import InvestigationNarrative, InvestigationReport
from .schemas import Finding, FindingsReport

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ai_investigations"
DQ004_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "DQ004_investigation.json"

AGENT_NAME = "Data Quality Investigator"
AGENT_ROLE = "Senior Data Quality Analyst specializing in commercial banking analytics."
AGENT_GOAL = (
    "Investigate validated data quality findings and produce clear, "
    "evidence-based reports for data engineering teams and business stakeholders."
)
AGENT_BACKSTORY = (
    "You are the Data Quality Investigator for a synthetic commercial-banking "
    "analytics portfolio. You explain one validated finding at a time. "
    "You treat root causes as hypotheses unless the finding itself proves them. "
    "You do not have access to live banking systems, source files, or production logs."
)

EXPECTED_OUTPUT = (
    "A structured narrative with an issue summary, business impact, "
    "possible root-cause hypotheses that each include a verification step, "
    "investigation steps, remediation actions, a confidence level of low, "
    "medium, or high, and limitations. Do not add rule ids, record counts, "
    "rates, or thresholds. Those confirmed values are stored from the finding."
)

PROVIDER_API_KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

NarrativeSource = Callable[[Finding], InvestigationNarrative | Mapping[str, Any]]


class InvestigatorConfigError(Exception):
    """The model name or API key is missing or blank."""


class InvestigationOutputError(Exception):
    """The model output is missing or does not match the investigation schema."""


@dataclass(frozen=True)
class LlmSettings:
    """Which model to call, and which environment variable holds the key.

    The key value is not stored here.
    """

    model: str
    api_key_env: str


def agent_runtime_limits() -> dict[str, Any]:
    """Limits applied to the single investigator agent.

    CrewAI identifies an agent by role, so the display name is stored on the
    crew. An empty tool list means the agent cannot read files or run commands.
    """
    return {
        "role": AGENT_ROLE,
        "goal": AGENT_GOAL,
        "backstory": AGENT_BACKSTORY,
        "tools": [],
        "allow_delegation": False,
        "allow_code_execution": False,
        "memory": False,
        "verbose": False,
        "max_iter": 3,
    }


def crew_runtime_limits() -> dict[str, Any]:
    """Limits for the one-task crew. Memory and tracing stay off."""
    return {
        "name": AGENT_NAME,
        "verbose": False,
        "memory": False,
        "cache": False,
        "tracing": False,
    }


def _load_dotenv() -> None:
    """Load a gitignored .env file when python-dotenv is installed.

    Existing environment variables win, so a shell setting is not overwritten.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_llm_settings(environ: Mapping[str, str] | None = None) -> LlmSettings:
    """Read the model id and check that the matching API key is present.

    Args:
        environ: Mapping to read. When omitted, the process environment is used
            after loading .env.

    Raises:
        InvestigatorConfigError: The model or key is missing, blank, or the
            provider is not one this project configures. No model call is made.
    """
    if environ is None:
        _load_dotenv()
        environ = os.environ

    model = environ.get("DQ_INVESTIGATOR_MODEL", "").strip()
    if not model:
        raise InvestigatorConfigError(
            "DQ_INVESTIGATOR_MODEL is missing or blank. "
            "Set it to a CrewAI model id such as openai/gpt-4o. "
            "No investigation was started."
        )
    if "/" not in model:
        raise InvestigatorConfigError(
            "DQ_INVESTIGATOR_MODEL must look like provider/model-id, "
            f"for example openai/gpt-4o. Got {model!r}. "
            "No investigation was started."
        )

    provider = model.split("/", 1)[0].strip().lower()
    key_names = PROVIDER_API_KEY_ENV.get(provider)
    if key_names is None:
        known = ", ".join(sorted(PROVIDER_API_KEY_ENV))
        raise InvestigatorConfigError(
            f"Unsupported model provider {provider!r}. "
            f"Use a model id starting with one of: {known}. "
            "No investigation was started."
        )
    key_env = next(
        (name for name in key_names if environ.get(name, "").strip()),
        None,
    )
    if key_env is None:
        needed = " or ".join(key_names)
        raise InvestigatorConfigError(
            f"{needed} is missing or blank. "
            "Set that environment variable to your API key. "
            "The key is not stored in the project. No investigation was started."
        )
    return LlmSettings(model=model, api_key_env=key_env)


def build_task_description(finding: Finding) -> str:
    """Build the task text from one finding.

    The JSON is the only evidence the model sees. Dataset files are not included.
    """
    confirmed = json.dumps(finding.model_dump(mode="json"), indent=2)
    return (
        "Investigate one validated data-quality finding from a synthetic, "
        "fictional commercial-banking analytics portfolio.\n\n"
        "Rules you must follow:\n"
        "- Use only the confirmed finding JSON below.\n"
        "- Do not invent or change the rule id, record counts, metric, threshold, "
        "or evidence values. If you mention a number, it must match this JSON exactly.\n"
        "- Do not put replacement metrics in your structured answer. The program "
        "copies the confirmed finding into the saved report.\n"
        "- Explain the observed problem in plain language for a data engineer "
        "and a business stakeholder.\n"
        "- Describe possible business impact. Label it as an implication.\n"
        "- Separate confirmed observations from possible root causes.\n"
        "- Every root cause must be a hypothesis, not a confirmed fact, and must "
        "include a verification step someone could perform.\n"
        "- Recommend specific investigation steps and practical remediation actions.\n"
        "- State limitations and uncertainty. You cannot see source pipelines. "
        "You do not have access to live banking systems or data files.\n"
        "- Do not claim a root cause is confirmed.\n\n"
        "Confirmed finding JSON:\n"
        f"{confirmed}\n"
    )


def parse_narrative(payload: InvestigationNarrative | Mapping[str, Any]) -> InvestigationNarrative:
    """Validate model output. Raise InvestigationOutputError instead of saving it."""
    if isinstance(payload, InvestigationNarrative):
        return payload
    try:
        return InvestigationNarrative.model_validate(payload)
    except ValidationError as exc:
        fields = ", ".join(
            ".".join(str(part) for part in error["loc"]) or "<root>"
            for error in exc.errors()
        )
        raise InvestigationOutputError(
            "The investigation output failed validation and was not saved. "
            f"Invalid fields: {fields}."
        ) from exc


def narrative_from_crew_result(result: Any) -> InvestigationNarrative:
    """Read a Crew kickoff result and validate the structured narrative."""
    parsed = getattr(result, "pydantic", None)
    if parsed is not None:
        return parse_narrative(parsed)
    json_payload = getattr(result, "json_dict", None)
    if json_payload is not None:
        return parse_narrative(json_payload)
    raise InvestigationOutputError(
        "The investigation output failed validation and was not saved. "
        "The model did not return the structured investigation report."
    )


def assemble_report(finding: Finding, narrative: InvestigationNarrative) -> InvestigationReport:
    """Combine the untouched finding with the validated narrative.

    rule_id and observed_evidence always come from finding, not from the model.
    """
    confirmed = finding.model_copy(deep=True)
    return InvestigationReport(
        rule_id=confirmed.rule_id,
        issue_summary=narrative.issue_summary,
        observed_evidence=confirmed,
        business_impact=narrative.business_impact,
        possible_root_causes=narrative.possible_root_causes,
        recommended_investigation_steps=list(narrative.recommended_investigation_steps),
        recommended_remediation=list(narrative.recommended_remediation),
        confidence_level=narrative.confidence_level,
        limitations=narrative.limitations,
    )


def investigate_finding(finding: Finding, narrative_source: NarrativeSource) -> InvestigationReport:
    """Turn one validated finding into a validated investigation report."""
    narrative = parse_narrative(narrative_source(finding))
    return assemble_report(finding, narrative)


def execute_crew(finding: Finding, settings: LlmSettings) -> InvestigationNarrative:
    """Run the one-agent crew. The API key stays in the environment."""
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

    from crewai import Agent, Crew, LLM, Task

    llm = LLM(model=settings.model)
    agent = Agent(llm=llm, **agent_runtime_limits())
    task = Task(
        description=build_task_description(finding),
        expected_output=EXPECTED_OUTPUT,
        agent=agent,
        output_pydantic=InvestigationNarrative,
        tools=[],
    )
    crew = Crew(agents=[agent], tasks=[task], **crew_runtime_limits())
    return narrative_from_crew_result(crew.kickoff())


def default_output_path(rule_id: str) -> Path:
    """Path for a saved investigation report."""
    return DEFAULT_OUTPUT_DIR / f"{rule_id}_investigation.json"


def ensure_report_matches_finding(report: InvestigationReport, finding: Finding) -> None:
    """Reject a report whose confirmed evidence differs from the finding.

    Raises:
        InvestigationOutputError: The rule id or observed evidence was changed.
            The caller must not write a file after this error.
    """
    if report.rule_id != finding.rule_id or report.observed_evidence.model_dump() != finding.model_dump():
        raise InvestigationOutputError(
            "The investigation output failed validation and was not saved. "
            "The report evidence does not match the validated finding."
        )


def write_investigation_report(report: InvestigationReport, path: Path) -> None:
    """Replace a report only after the new file has been written completely.

    A failed attempt raises before this function is called, so an existing
    report stays in place. The findings file is never the destination.
    """
    destination = path.resolve()
    if destination == DEFAULT_FINDINGS_PATH.resolve():
        raise InvestigationOutputError(
            "Refusing to write the investigation over the findings file. "
            "The findings file was not changed."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.model_dump(mode="json"), indent=2) + "\n"
    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def run_investigation(
    rule_id: str = "DQ004",
    findings_path: Path | str | None = None,
    output_path: Path | str | None = None,
    narrative_source: NarrativeSource | None = None,
) -> Path:
    """Load one finding, investigate it, and save a validated report.

    Args:
        rule_id: Finding to investigate. The command-line entry point passes DQ004.
        findings_path: Findings JSON. Defaults to outputs/dq_findings.json.
        output_path: Where to write the report.
        narrative_source: Replacement for the crew, used by tests. When omitted,
            the configured model is called.

    Returns:
        The path of the written report.

    Raises:
        InvestigatorConfigError: The model or API key is not configured.
        InvestigationOutputError: The model output is invalid. Nothing is written.
    """
    finding = select_finding(load_findings_report(findings_path), rule_id)
    if narrative_source is None:
        settings = resolve_llm_settings()

        def narrative_source(selected: Finding) -> InvestigationNarrative:
            return execute_crew(selected, settings)

    report = investigate_finding(finding, narrative_source)
    ensure_report_matches_finding(report, finding)
    destination = Path(output_path) if output_path is not None else default_output_path(rule_id)
    write_investigation_report(report, destination)
    return destination


@dataclass(frozen=True)
class InvestigationAttempt:
    """One selected finding and the outcome of investigating it."""

    rule_id: str
    status: str
    dataset: str
    output_path: Path
    error: str | None = None


def findings_to_investigate(
    report: FindingsReport,
    *,
    rule_id: str | None = None,
    all_failed: bool = False,
) -> list[Finding]:
    """Choose DQ004, one rule id, or every failed finding.

    Args:
        report: A validated findings report.
        rule_id: One rule to investigate. Ignored when all_failed is true.
        all_failed: When true, return findings whose status is FAIL, in file order.
    """
    if all_failed:
        return failed_findings(report)
    return [select_finding(report, "DQ004" if rule_id is None else rule_id)]


def run_investigations(
    rule_id: str | None = None,
    all_failed: bool = False,
    dry_run: bool = False,
    findings_path: Path | str | None = None,
    output_dir: Path | None = None,
    narrative_source: NarrativeSource | None = None,
) -> list[InvestigationAttempt]:
    """Investigate the selected findings one at a time with the same investigator.

    A dry run only lists the findings. It does not read API credentials and it
    does not call the model. When a finding fails, its report file is left
    unchanged and the remaining findings still run.
    """
    loaded = load_findings_report(findings_path)
    selected = findings_to_investigate(loaded, rule_id=rule_id, all_failed=all_failed)
    directory = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    planned = [
        InvestigationAttempt(
            rule_id=item.rule_id,
            status=item.status,
            dataset=item.dataset,
            output_path=directory / f"{item.rule_id}_investigation.json",
        )
        for item in selected
    ]
    if dry_run:
        return planned

    source = narrative_source
    if source is None:
        settings = resolve_llm_settings()

        def source(selected_finding: Finding) -> InvestigationNarrative:
            return execute_crew(selected_finding, settings)

    results: list[InvestigationAttempt] = []
    for attempt in planned:
        try:
            written = run_investigation(
                attempt.rule_id,
                findings_path=findings_path,
                output_path=attempt.output_path,
                narrative_source=source,
            )
        except Exception as exc:
            results.append(
                InvestigationAttempt(
                    rule_id=attempt.rule_id,
                    status=attempt.status,
                    dataset=attempt.dataset,
                    output_path=attempt.output_path,
                    error=str(exc),
                )
            )
            continue
        results.append(
            InvestigationAttempt(
                rule_id=attempt.rule_id,
                status=attempt.status,
                dataset=attempt.dataset,
                output_path=written,
                error=None,
            )
        )
    return results
