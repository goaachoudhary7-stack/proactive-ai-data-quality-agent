"""Streamlit workflow for CSV or Excel uploads.

The banking demonstration pages do not call this module. Uploaded bytes stay
in the session and are not written into the project. Moving between the five
steps does not replace the working copy.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from dashboard.cleaning_view import render_cleaning, render_cleaning_downloads
from dashboard.recommendations_view import recommendation_workbook, render_recommendations
from dashboard.session_store import (
    KEY_COLUMNS_KEY,
    PAYLOAD_KEY,
    REQUIRED_KEY,
    STEP_LABELS,
    UPLOADER_GENERATION_KEY,
    WORKFLOW_WIDGET_KEY,
    StoredUpload,
    apply_step_request,
    clear_uploaded_files,
    remember_step,
    request_uploaded_source,
    restore_step_widget,
    should_select_uploaded_source,
    snapshot_uploads,
    step_id,
)
from dashboard.upload_context import open_upload_session, profile_batches
from dashboard.workflow import (
    render_inventory,
    render_relationship_diagram,
    render_step_heading,
    section_label,
)
from universal_dq.batch import LoadedDataset
from universal_dq.cleaning import CleaningSession
from universal_dq.loader import FileLoadError, inspect_upload
from universal_dq.models import PROCESSING_NOTE, AnalysisReport, BatchReport
from universal_dq.report import format_share, report_to_csv, report_to_json, report_to_xlsx


def render_upload() -> None:
    """Show the five-step analyzer."""
    st.markdown("### Analyze a file")
    st.caption("Uploaded rows stay in this session and are not sent to an external model.")
    with st.expander("Privacy & processing details"):
        st.write(PROCESSING_NOTE)
        st.write(
            "These checks are separate from the banking demonstration and its saved AI investigations. "
            "Missing values, possible mixed types, and possible numerical outliers are observations. "
            "A missing value becomes a rule result only when you mark that column as required. "
            "You can select more than one file. Each selected worksheet is a separate dataset."
        )
        st.write(
            "A successful upload selects My uploaded files. "
            "The banking demonstration stays available under Active data source. "
            "Uploaded files are not written to the project folder. "
            "Changing the uploaded files or the selected worksheets starts a new working copy."
        )

    apply_step_request(st.session_state)
    restore_step_widget(st.session_state)
    label = st.radio(
        "Workflow",
        list(STEP_LABELS),
        horizontal=True,
        label_visibility="collapsed",
        key=WORKFLOW_WIDGET_KEY,
    )
    remember_step(st.session_state, str(label))
    current = step_id(str(label))
    render_step_heading(current)

    if current == "upload":
        _render_uploader()

    opened = open_upload_session(st.session_state)
    if opened is None:
        if current != "upload":
            st.info("Upload a file to continue.")
        else:
            st.caption("Choose one or more .csv or .xlsx files. The first row of each dataset is the header.")
        return
    if should_select_uploaded_source(
        bytes_changed=opened.bytes_changed,
        dataset_count=len(opened.session.order),
    ):
        request_uploaded_source(st.session_state)
        st.rerun()

    session = opened.session
    errors = opened.errors
    fingerprint = opened.fingerprint
    required, key_columns = _rule_selection(session, errors, show_widgets=current == "analyze")
    original_batch, current_batch = profile_batches(st.session_state, opened, required, key_columns)
    _show_load_errors(original_batch)
    selected = render_inventory(current_batch) if current == "analyze" else None

    if current == "upload":
        _render_upload_summary(original_batch)
    elif current == "analyze":
        _render_analyze(original_batch, selected, single=_single(session, errors))
    elif current == "insights":
        _render_insights(current_batch, session, fingerprint)
    elif current == "clean":
        if session.order:
            render_cleaning(original_batch, current_batch)
        else:
            st.caption("No dataset is available to clean.")
    else:
        _render_exports(original_batch, current_batch, session, fingerprint, single=_single(session, errors))


def _render_uploader() -> None:
    if st.session_state.pop("dq_remove_now", None):
        st.session_state.pop("dq-remove-confirm", None)
        clear_uploaded_files(st.session_state)
        st.rerun()
    generation = int(st.session_state.get(UPLOADER_GENERATION_KEY, 0))
    uploaded = st.file_uploader(
        "CSV or Excel files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key=f"dq-uploader-{generation}",
    )
    if uploaded:
        incoming = [(item.name, item.getvalue()) for item in uploaded]
        existing = list(st.session_state.get(PAYLOAD_KEY) or [])
        st.session_state[PAYLOAD_KEY] = snapshot_uploads(existing, incoming)
    payload = list(st.session_state.get(PAYLOAD_KEY) or [])
    if not payload:
        return
    st.caption("Choosing files above replaces the files kept for this session. Worksheet choices stay with those files.")
    updated: list[StoredUpload] = []
    for index, item in enumerate(payload):
        updated.append(_sheet_choice(index, item))
    st.session_state[PAYLOAD_KEY] = updated
    st.checkbox("Remove the uploaded files from this session", key="dq-remove-confirm")
    if st.button("Remove uploaded files"):
        if not st.session_state.get("dq-remove-confirm"):
            st.error("Check the confirmation box before removing uploaded files.")
        else:
            st.session_state["dq_remove_now"] = True
            st.rerun()


def _sheet_choice(index: int, item: StoredUpload) -> StoredUpload:
    try:
        inspection = inspect_upload(item.content, item.name)
    except FileLoadError:
        return item
    if inspection.file_format != "xlsx":
        return item
    options = list(inspection.sheet_names)
    widget_key = f"worksheets-{index}-{_short_hash(item.content)}"
    if widget_key not in st.session_state:
        chosen = list(item.sheets) if item.sheets else ([options[0]] if options else [])
        st.session_state[widget_key] = [name for name in chosen if name in options]
    selected = st.multiselect(f"Worksheets in {item.name}", options, key=widget_key)
    return replace(item, sheets=tuple(selected))


def _rule_selection(
    session: CleaningSession,
    errors: list[LoadedDataset],
    *,
    show_widgets: bool,
) -> tuple[list[str] | None, list[str] | None]:
    if not _single(session, errors):
        return None, None
    names = [str(name) for name in session.copies[session.order[0]].original.columns]
    required = [name for name in list(st.session_state.get(REQUIRED_KEY) or []) if name in names]
    key_columns = [name for name in list(st.session_state.get(KEY_COLUMNS_KEY) or []) if name in names]
    if not show_widgets:
        return required or None, key_columns or None
    held_required = st.session_state.get("dq-required-columns", required)
    held_keys = st.session_state.get("dq-key-columns", key_columns)
    st.session_state["dq-required-columns"] = [name for name in held_required if name in names]
    st.session_state["dq-key-columns"] = [name for name in held_keys if name in names]
    required = st.multiselect("Required columns (optional)", names, key="dq-required-columns")
    key_columns = st.multiselect(
        "Unique-key columns (optional)",
        names,
        key="dq-key-columns",
        help="Duplicate keys are calculated separately from exact full-row duplicates.",
    )
    st.session_state[REQUIRED_KEY] = required
    st.session_state[KEY_COLUMNS_KEY] = key_columns
    return required or None, key_columns or None


def _render_upload_summary(batch: BatchReport) -> None:
    analyzed = [item for item in batch.datasets if item.report is not None]
    first, second, third = st.columns(3)
    first.metric("Files kept", len(batch.datasets))
    second.metric("Datasets analyzed", len(analyzed))
    third.metric("Data rows", sum(item.report.data_rows for item in analyzed if item.report is not None))
    with st.expander("Limitations"):
        for note in batch.limitations:
            st.write(note)


def _render_analyze(batch: BatchReport, selected: str | None, *, single: bool) -> None:
    item = next((entry for entry in batch.datasets if entry.dataset_id == selected), None)
    if item is None or item.report is None:
        st.caption("This dataset has no profile.")
        return
    section_label("confirmed", "Column profile")
    _render_summary(item.report)
    _render_columns(item.report)
    if single:
        _render_rules(item.report)


def _render_insights(batch: BatchReport, session: CleaningSession, fingerprint: str) -> None:
    _render_keys(batch)
    render_relationship_diagram(batch.relationships)
    if batch.relationships:
        st.table(
            pd.DataFrame(
                [
                    {
                        "Parent dataset": item.parent_dataset,
                        "Parent column": item.parent_column,
                        "Child dataset": item.child_dataset,
                        "Child column": item.child_column,
                        "Cardinality": item.cardinality,
                        "Matched": item.matched_count,
                        "Child non-missing": item.child_non_missing,
                        "Orphans": item.orphan_count,
                        "Evidence": item.detail,
                    }
                    for item in batch.relationships
                ]
            )
        )
    frames = {dataset_id: session.copies[dataset_id].working for dataset_id in session.order}
    render_recommendations(batch, frames, fingerprint=fingerprint, generation=session.generation)


def _render_keys(batch: BatchReport) -> None:
    section_label("inferred", "Candidate keys")
    st.caption("Suggestions from completeness and uniqueness. These are not confirmed keys.")
    if batch.candidate_keys:
        st.table(
            pd.DataFrame(
                [
                    {
                        "Dataset": item.dataset_id,
                        "Column": item.column,
                        "Distinct values": item.distinct_non_missing,
                        "Missing values": item.missing_count,
                        "Evidence": item.detail,
                    }
                    for item in batch.candidate_keys
                ]
            )
        )
    else:
        st.caption("No column is both complete and unique across at least 2 data rows.")


def _render_exports(
    original: BatchReport,
    current: BatchReport,
    session: CleaningSession,
    fingerprint: str,
    *,
    single: bool,
) -> None:
    st.subheader("Original reports")
    st.caption("These downloads describe the original upload.")
    analyzed = [item for item in original.datasets if item.report is not None]
    if single and analyzed:
        report = analyzed[0].report
        assert report is not None
        st.download_button(
            "Download data-quality report JSON",
            data=report_to_json(report),
            file_name="data_quality_report.json",
            mime="application/json",
        )
        st.download_button(
            "Download data-quality report CSV",
            data=report_to_csv(report),
            file_name="data_quality_report.csv",
            mime="text/csv",
        )
    st.download_button(
        "Download consolidated Excel report",
        data=report_to_xlsx(original),
        file_name="data_quality_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.subheader("Recommendation report")
    st.caption("This workbook is separate from the original report. It adds recommendation sheets.")
    frames = {dataset_id: session.copies[dataset_id].working for dataset_id in session.order}
    st.download_button(
        "Download Excel report with recommendations",
        data=recommendation_workbook(current, frames, fingerprint=fingerprint, generation=session.generation),
        file_name="data_quality_report_with_recommendations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="recommendations-xlsx",
    )
    render_cleaning_downloads(original, current)


def _show_load_errors(batch: BatchReport) -> None:
    for item in batch.datasets:
        if item.error:
            st.error(f"{item.dataset_id}: {item.error}")


def _single(session: CleaningSession, errors: list[LoadedDataset]) -> bool:
    return len(session.order) == 1 and not errors


def _short_hash(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()[:8]


def _render_summary(report: AnalysisReport) -> None:
    st.subheader("File profile")
    if report.sheet_name:
        st.caption(f"Worksheet: {report.sheet_name}")
    first, second, third, fourth = st.columns(4)
    first.metric("Data rows", report.data_rows)
    second.metric("Columns", report.column_count)
    third.metric(
        "Rows in exact duplicate groups",
        report.exact_duplicates.rows_in_duplicate_groups.count,
    )
    fourth.metric("Extra exact copies", report.exact_duplicates.extra_copies.count)
    st.write(format_share(report.exact_duplicates.rows_in_duplicate_groups))
    st.write("Extra copies beyond the first occurrence: " + format_share(report.exact_duplicates.extra_copies))
    st.caption(report.exact_duplicates.detail)


def _render_columns(report: AnalysisReport) -> None:
    st.subheader("Columns")
    rows = []
    for column in report.columns:
        mixed = column.mixed_type.detail if column.mixed_type is not None else ""
        outlier = column.outlier.detail if column.outlier is not None else ""
        rows.append(
            {
                "Column": column.name,
                "Inferred type": column.inferred_type,
                "Missing": format_share(column.missing),
                "Missing label": _missing_label(column.missing_role),
                "Mixed types": mixed,
                "Outliers": outlier,
            }
        )
    st.table(pd.DataFrame(rows))


def _render_rules(report: AnalysisReport) -> None:
    if report.required_columns:
        st.subheader("Required columns")
        st.caption("You selected these columns. Missing values here are rule results.")
        for column in report.columns:
            if column.name not in report.required_columns:
                continue
            st.write(f"**{column.name}:** {format_share(column.missing)} ({_missing_label(column.missing_role)})")
    st.subheader("Unique key")
    if report.key_duplicates is None:
        st.caption(
            "No unique key is selected. Exact full-row duplicates are still shown above. "
            "Choose one or more columns to add a separate key-duplicate check."
        )
        return
    key = report.key_duplicates
    st.write("Key columns: " + ", ".join(key.key_columns))
    st.write("Rows in duplicate key groups: " + format_share(key.rows_in_duplicate_groups))
    st.write("Extra key copies beyond the first occurrence: " + format_share(key.extra_copies))
    st.write(f"Incomplete keys excluded: {key.incomplete_key_rows:,}")
    st.caption(key.detail)


def _missing_label(role: str) -> str:
    if role == "user_rule_violation":
        return "Required column has missing values"
    if role == "user_rule_clear":
        return "Required column has no missing values"
    return "Observation"
