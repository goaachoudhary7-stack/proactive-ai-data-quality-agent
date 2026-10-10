"""Streamlit dashboard for saved banking results and uploaded CSV or Excel files.

Launch from the project root:

    .\\.venv\\Scripts\\python.exe -m streamlit run src/dashboard/app.py

The sidebar offers Overview and Analyze a file. Banking Overview reads saved
findings and investigation JSON. The upload page analyzes files in memory.
An optional explanation runs only after you approve the summary shown on the
page. The page does not write datasets, findings, reports, or uploaded files.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st

from dashboard.session_store import (
    PAGE_WIDGET_KEY,
    SOURCE_LABELS,
    SOURCE_WIDGET_KEY,
    UPLOADED_SOURCE_LABEL,
    VERSION_LABELS,
    VERSION_WIDGET_KEY,
    apply_page_request,
    apply_source_request,
    fallback_hidden_page,
    remember_source,
    remember_version,
    restore_source_widget,
    restore_version_widget,
)
from dashboard.upload import render_upload
from dashboard.upload_pages import render_uploaded_pages
from dashboard.data import (
    SYNTHETIC_DATA_LABEL,
    DashboardLoad,
    InvestigationView,
    available_report_count,
    count_by_dimension,
    count_by_severity,
    filter_findings,
    load_dashboard,
    metric_explanation,
)
from ai_agents.schemas import Finding

TABLE_COLUMNS = [
    ("rule_id", "Rule ID"),
    ("check_name", "Check"),
    ("dataset", "Dataset"),
    ("field", "Field"),
    ("dimension", "Dimension"),
    ("severity", "Severity"),
    ("status", "Status"),
    ("records_affected", "Records affected"),
    ("records_checked", "Records checked"),
    ("metric_value", "Metric"),
    ("threshold", "Threshold"),
]


def main() -> None:
    st.set_page_config(
        page_title="Data quality",
        layout="wide",
    )
    apply_source_request(st.session_state)
    apply_page_request(st.session_state)
    fallback_hidden_page(st.session_state)
    restore_source_widget(st.session_state)
    source = st.sidebar.radio("Active data source", list(SOURCE_LABELS), key=SOURCE_WIDGET_KEY)
    remember_source(st.session_state, str(source))
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Analyze a file"],
        key=PAGE_WIDGET_KEY,
    )
    if page == "Analyze a file":
        render_upload()
        return

    if source == UPLOADED_SOURCE_LABEL:
        restore_version_widget(st.session_state)
        version = st.radio(
            "Uploaded data",
            list(VERSION_LABELS),
            horizontal=True,
            key=VERSION_WIDGET_KEY,
        )
        remember_version(st.session_state, str(version))
        render_uploaded_pages(page, str(version))
        return

    loaded = load_dashboard()

    if loaded.findings_error is not None or loaded.report is None:
        st.markdown("### Data quality overview")
        st.error(loaded.findings_error or "The findings file could not be loaded.")
        return

    render_overview(loaded)


def render_overview(loaded: DashboardLoad) -> None:
    report = loaded.report
    assert report is not None
    available = available_report_count(loaded.investigations)
    st.markdown("### Data quality overview")
    st.caption(
        "Read-only view of saved check results and investigation reports. "
        "No files are uploaded or overwritten from this screen."
    )
    st.info(
        "Synthetic demonstration data. These records describe a fictional commercial bank. "
        "The seven failed checks were deliberately introduced into this demonstration dataset. "
        "They do not describe real customers."
    )
    _render_kpis(report.checks_run, report.passed, report.warnings, report.failed, available, len(report.findings))
    st.caption(
        "A validated investigation matches the investigation schema and its confirmed "
        "evidence matches the original finding."
    )
    _render_business_risk(report.findings, report.passed, report.warnings, report.failed)

    st.subheader("Checks by severity")
    st.caption("Each bar is a count of checks. This chart is not a data-quality score.")
    _bar_chart(count_by_severity(report.findings))

    st.subheader("Checks by dimension")
    st.caption("Each bar is a count of checks in that data-quality dimension.")
    _bar_chart(count_by_dimension(report.findings))

    st.download_button(
        "Download findings JSON",
        data=_findings_download(loaded),
        file_name="dq_findings.json",
        mime="application/json",
    )


def render_findings(loaded: DashboardLoad) -> None:
    report = loaded.report
    assert report is not None
    findings = list(report.findings)
    st.subheader("Findings")
    st.caption(SYNTHETIC_DATA_LABEL)

    datasets = sorted({item.dataset for item in findings})
    severities = sorted({item.severity for item in findings})
    dimensions = sorted({item.dimension for item in findings})
    statuses = sorted({item.status for item in findings})

    dataset_choice = st.sidebar.multiselect("Dataset", datasets, default=datasets)
    severity_choice = st.sidebar.multiselect("Severity", severities, default=severities)
    dimension_choice = st.sidebar.multiselect("Dimension", dimensions, default=dimensions)
    status_choice = st.sidebar.multiselect("Status", statuses, default=statuses)

    selected = filter_findings(
        findings,
        datasets=dataset_choice,
        severities=severity_choice,
        dimensions=dimension_choice,
        statuses=status_choice,
    )
    frame = findings_frame(selected)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered findings CSV",
        data=frame.to_csv(index=False),
        file_name="dq_findings_filtered.csv",
        mime="text/csv",
    )

    if not selected:
        st.info("No findings match the current filters.")
        return

    labels = {item.rule_id: f"{item.rule_id} — {item.check_name}" for item in selected}
    choice = st.selectbox("Finding detail", list(labels), format_func=labels.get)
    finding = next(item for item in selected if item.rule_id == choice)
    render_finding_detail(finding)


def render_finding_detail(finding: Finding) -> None:
    st.markdown(f"**{finding.check_name}**")
    st.write(finding.description)
    st.write(metric_explanation(finding))
    st.caption(
        "Records affected and records checked come from the detection engine. "
        "The metric is shown in the units stored for this check."
    )
    st.json(finding.evidence)


def render_investigation(loaded: DashboardLoad) -> None:
    report = loaded.report
    assert report is not None
    st.subheader("Investigation")
    st.caption(SYNTHETIC_DATA_LABEL)
    st.caption(
        "Confirmed evidence is copied from the detection engine. "
        "Hypotheses, impact, and recommendations are AI-generated and are not confirmed causes."
    )

    labels = {
        item.rule_id: f"{item.rule_id} — {item.check_name}" for item in report.findings
    }
    choice = st.selectbox("Finding", list(labels), format_func=labels.get)
    finding = next(item for item in report.findings if item.rule_id == choice)
    view = next(item for item in loaded.investigations if item.rule_id == choice)

    st.markdown("#### Confirmed evidence")
    st.write(metric_explanation(finding))
    st.write(finding.description)
    st.json(finding.evidence)

    st.markdown("#### AI investigation")
    if view.status != "available" or view.report is None:
        st.info(view.message)
        return

    narrative = view.report
    st.write(narrative.issue_summary)
    st.markdown("**Business impact**")
    st.write(narrative.business_impact)
    st.markdown("**Possible root causes**")
    st.caption("Each item is a hypothesis with a step that could verify it.")
    for cause in narrative.possible_root_causes:
        st.markdown(f"- **Hypothesis:** {cause.hypothesis}")
        st.markdown(f"  **Verification step:** {cause.verification_step}")
    st.markdown("**Recommended investigation steps**")
    for step in narrative.recommended_investigation_steps:
        st.markdown(f"- {step}")
    st.markdown("**Recommended remediation**")
    for step in narrative.recommended_remediation:
        st.markdown(f"- {step}")
    st.write(f"Confidence: {narrative.confidence_level}")
    st.markdown("**Limitations**")
    st.write(narrative.limitations)
    st.download_button(
        "Download investigation JSON",
        data=view.path.read_bytes(),
        file_name=view.path.name,
        mime="application/json",
    )


def _render_kpis(
    checks_run: int,
    passed: int,
    warnings: int,
    failed: int,
    available: int,
    finding_count: int,
) -> None:
    """Show the five saved counts. Failed and validated investigations use separate accents."""
    cards = (
        ("Checks run", str(checks_run), "neutral"),
        ("Passed", str(passed), "neutral"),
        ("Warnings", str(warnings), "neutral"),
        ("Failed checks", str(failed), "fail"),
        ("Validated AI investigations", f"{available} of {finding_count}", "ai"),
    )
    st.markdown(
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));"
        "gap:0.75rem;margin:0.25rem 0 0.4rem 0;'>"
        + "".join(_kpi_card(label, value, accent) for label, value, accent in cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, accent: str) -> str:
    borders = {
        "fail": "#fb7185",
        "ai": "#c4b5fd",
        "neutral": "#22d3ee",
    }
    border = borders[accent]
    return (
        "<div style='background:#141a2e;color:#e7eefc;border:1px solid #243049;border-left:4px solid "
        f"{border};border-radius:8px;padding:0.65rem 0.8rem;min-width:0;'>"
        f"<div style='font-size:0.8rem;color:#b7c3dc;'>{html.escape(label)}</div>"
        f"<div style='font-size:1.55rem;font-weight:650;line-height:1.25;color:#e7eefc;'>{html.escape(value)}</div>"
        "</div>"
    )


def _render_business_risk(
    findings: list[Finding] | tuple[Finding, ...],
    passed: int,
    warnings: int,
    failed: int,
) -> None:
    """Summarize saved failures. Counts and metrics come from the findings file."""
    failed_findings = [item for item in findings if item.status == "FAIL"]
    critical = [item for item in failed_findings if item.severity == "CRITICAL"]
    high = [item for item in failed_findings if item.severity == "HIGH"]
    st.subheader("Business risk")
    outcome = f"{failed} failed checks are in the saved results: {len(critical)} critical and {len(high)} high."
    if passed == 0 and warnings == 0:
        outcome += " No checks passed or warned."
    else:
        outcome += f" Passed: {passed}. Warnings: {warnings}."
    st.write(outcome)
    st.caption("Each row uses that check's stored records, metric, and threshold. These rows are not combined into a quality score.")

    for finding in critical:
        _critical_card(finding)

    if high:
        st.markdown("**High-severity findings**")
        rows = [
            {
                "Rule ID": item.rule_id,
                "Check": item.check_name,
                "Dataset": item.dataset,
                "Measured result": _measured_result(item),
            }
            for item in sorted(high, key=lambda item: item.rule_id)
        ]
        st.table(pd.DataFrame(rows))


def _critical_card(finding: Finding) -> None:
    first_sentence = finding.description.split(". ")[0].rstrip(".") + "."
    st.markdown(
        "<div style='background:#141a2e;color:#e7eefc;border:1px solid rgba(251,113,133,0.45);"
        "border-left:6px solid #fb7185;border-radius:8px;padding:0.8rem 1rem;margin:0.35rem 0 0.8rem 0;'>"
        "<div style='color:#fb7185;font-weight:700;letter-spacing:0.04em;'>CRITICAL</div>"
        f"<div style='font-size:1.05rem;font-weight:650;color:#e7eefc;'>{html.escape(finding.rule_id)} · "
        f"{html.escape(finding.check_name)}</div>"
        f"<div style='color:#b7c3dc;'>{html.escape(finding.dataset)} · {html.escape(finding.field)}</div>"
        f"<div style='margin-top:0.4rem;color:#e7eefc;'>{html.escape(metric_explanation(finding))}</div>"
        f"<div style='margin-top:0.3rem;color:#d5deef;'>{html.escape(first_sentence)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _measured_result(finding: Finding) -> str:
    """Explain a finding in the units stored for that check."""
    if finding.rule_id == "DQ005":
        return _null_rate_gap_text(finding)
    if finding.rule_id == "DQ006":
        return _freshness_lag_text(finding)
    if finding.rule_id == "DQ007":
        return _volume_deviation_text(finding)
    return metric_explanation(finding)


def _null_rate_gap_text(finding: Finding) -> str:
    evidence = finding.evidence
    gap = evidence.get("rate_difference", finding.metric_value)
    recent = evidence.get("recent_null_rate")
    baseline = evidence.get("baseline_null_rate")
    window = evidence.get("window_days")
    records = (
        f"{finding.records_affected:,} of {finding.records_checked:,} records checked"
    )
    detail = f"Industry null-rate gap: {_compact_number(gap)}. Threshold {finding.threshold:g}."
    if recent is not None and baseline is not None:
        detail += (
            f" Recent null rate {_compact_number(recent)} minus baseline {_compact_number(baseline)}"
        )
        if window is not None:
            detail += f" over {window:g} days"
        detail += "."
    return (
        f"{detail} {records}. "
        "This gap is the difference between two null rates, not a percentage of records checked."
    )


def _freshness_lag_text(finding: Finding) -> str:
    lag = finding.evidence.get("freshness_lag_days", finding.metric_value)
    return (
        f"Freshness lag: {_compact_number(lag)} days. "
        f"Threshold {finding.threshold:g} days. "
        f"{finding.records_affected:,} of {finding.records_checked:,} records checked. "
        "The metric is a day count, not a percentage of records."
    )


def _volume_deviation_text(finding: Finding) -> str:
    deviation = finding.evidence.get("percentage_deviation", finding.metric_value)
    return (
        f"Volume deviation from the trailing daily baseline: {_compact_number(deviation)}. "
        f"Threshold {finding.threshold:g}. "
        f"{finding.records_affected:,} of {finding.records_checked:,} records checked. "
        "This deviation is not the share of records checked."
    )


def _compact_number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:g}"


def findings_frame(findings: list[Finding]) -> pd.DataFrame:
    """Build the findings table from validated finding fields."""
    rows = [
        {label: getattr(finding, field) for field, label in TABLE_COLUMNS}
        for finding in findings
    ]
    return pd.DataFrame(rows, columns=[label for _, label in TABLE_COLUMNS])


def _bar_chart(rows: list[tuple[str, int]]) -> None:
    if not rows:
        st.info("No checks to chart.")
        return
    frame = pd.DataFrame(rows, columns=["Category", "Number of checks"])
    st.bar_chart(frame, x="Category", y="Number of checks")


def _findings_download(loaded: DashboardLoad) -> bytes:
    report = loaded.report
    assert report is not None
    return report.model_dump_json(indent=2).encode("utf-8")


if __name__ == "__main__":
    main()
