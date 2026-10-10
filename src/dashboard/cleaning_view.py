"""Cleaning controls for the upload page.

The working copies live in the Streamlit session. This module does not write
files or call an external model.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from universal_dq.cleaning import (
    CleaningSession,
    PendingPreview,
    apply_operation,
    preview_operation,
    reset,
    same_request,
    undo,
)
from universal_dq.cleaning import CleaningRejected
from universal_dq.cleaning_export import (
    cleaned_dataset_csv,
    cleaned_datasets_xlsx,
    cleaning_audit_xlsx,
)
from universal_dq.models import BatchReport, OperationSpec
from universal_dq.recommendations import PREFILL_KEY

SESSION_KEY = "dq_cleaning_session"
FINGERPRINT_KEY = "dq_cleaning_fingerprint"
ERRORS_KEY = "dq_cleaning_errors"
PENDING_KEY = "dq_cleaning_pending"

_OPERATIONS = (
    ("drop_exact_duplicates", "Remove exact duplicate rows"),
    ("trim_whitespace", "Trim leading and trailing whitespace"),
    ("whitespace_to_missing", "Convert whitespace-only text to missing"),
    ("drop_empty_rows", "Remove completely empty rows"),
    ("fill_constant", "Fill missing values with a constant"),
    ("fill_median", "Fill missing numeric values with the median"),
    ("change_case", "Standardize text case"),
    ("convert_type", "Convert column type"),
    ("rename_columns", "Rename columns"),
)
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
)
_ROW_OPERATIONS = {"drop_exact_duplicates", "drop_empty_rows"}


def render_cleaning(original: BatchReport, current: BatchReport) -> None:
    """Show preview, approval, undo, and reset. Downloads live on the export step."""
    session: CleaningSession | None = st.session_state.get(SESSION_KEY)
    if session is None or not session.order:
        return

    spec = _operation_form(session)
    if spec is None:
        return
    if st.button("Preview", key="clean-preview"):
        st.session_state[PENDING_KEY] = PendingPreview(
            spec=spec,
            preview=preview_operation(session, spec),
            generation=session.generation,
        )
    _render_pending(session, spec)
    _render_history(session)
    _render_comparison(original, current)


def render_cleaning_downloads(original: BatchReport, current: BatchReport) -> None:
    """Offer cleaned files and the audit workbook."""
    session: CleaningSession | None = st.session_state.get(SESSION_KEY)
    if session is None or not session.order:
        return
    _render_downloads(session, original, current)


def _operation_form(session: CleaningSession) -> OperationSpec | None:
    prefill = st.session_state.pop(PREFILL_KEY, None)
    if isinstance(prefill, dict) and prefill.get("dataset_id") in session.order:
        st.session_state["clean-dataset"] = prefill["dataset_id"]
        if prefill.get("operation"):
            st.session_state["clean-operation"] = prefill["operation"]
        columns = prefill.get("columns") or []
        if columns:
            st.session_state[f"clean-columns-{prefill['dataset_id']}"] = columns
    dataset_id = st.selectbox("Dataset", session.order, key="clean-dataset")
    labels = {key: label for key, label in _OPERATIONS}
    operation = st.selectbox(
        "Operation",
        [key for key, _label in _OPERATIONS],
        format_func=lambda key: labels[key],
        key="clean-operation",
    )
    item = session.copies[dataset_id]
    names = [str(name) for name in item.working.columns]
    columns: list[str] = []
    fill_value: str | None = None
    case: str | None = None
    target_type: str | None = None
    date_format: str | None = None
    renames: dict[str, str] = {}
    if operation not in _ROW_OPERATIONS:
        columns = st.multiselect("Columns", names, key=f"clean-columns-{dataset_id}")
    if operation == "fill_constant":
        fill_value = st.text_input("Constant for missing cells", key="clean-fill")
        st.caption("The audit workbook does not store this constant.")
    elif operation == "fill_median":
        st.caption("Median fill runs only when every non-missing value in the column is already numeric.")
    elif operation == "change_case":
        case = st.radio("Text case", ["upper", "lower", "title"], horizontal=True, key="clean-case")
    elif operation == "convert_type":
        target_type = st.radio(
            "Convert to",
            ["numeric", "date", "text"],
            horizontal=True,
            key="clean-type",
        )
        if target_type == "date":
            choice = st.selectbox("Date format", [*_DATE_FORMATS, "Other"], key="clean-date-choice")
            if choice == "Other":
                date_format = st.text_input("Explicit date format", value="%Y-%m-%d", key="clean-date-custom")
            else:
                date_format = str(choice)
            st.caption(
                "Dates are parsed only with this format. "
                "A value that does not match is left unchanged and counted as a conversion failure."
            )
    elif operation == "rename_columns":
        for name in columns:
            entered = st.text_input(f"New name for {name}", value=name, key=f"clean-rename-{dataset_id}-{name}")
            renames[name] = entered
    return OperationSpec(
        operation=operation,  # type: ignore[arg-type]
        dataset_id=dataset_id,
        columns=columns,
        fill_value=fill_value,
        case=case,  # type: ignore[arg-type]
        target_type=target_type,  # type: ignore[arg-type]
        date_format=date_format,
        renames=renames,
    )


def _render_pending(session: CleaningSession, spec: OperationSpec) -> None:
    pending: PendingPreview | None = st.session_state.get(PENDING_KEY)
    if pending is None:
        st.caption("Preview an operation to see how many rows and cells it would change.")
        return
    if not same_request(pending, spec, session.generation):
        st.caption("The form or the working copy has changed. Preview this operation again before applying it.")
        return
    preview = pending.preview
    if not preview.applicable:
        st.error(preview.block_reason or "This operation cannot be applied.")
        return
    st.write(
        f"Rows: {preview.rows_before} → {preview.rows_after}. "
        f"Rows removed: {preview.rows_removed}. "
        f"Cells changed: {preview.cells_changed}. "
        f"Conversion failures: {preview.conversion_failures}."
    )
    for note in preview.columns_skipped:
        st.warning(note)
    for note in preview.warnings:
        st.warning(note)
    approved = st.checkbox("I approve this operation", key="clean-approve")
    if st.button("Apply approved operation", key="clean-apply"):
        if not approved:
            st.error("Check the approval box before applying this operation.")
            return
        try:
            updated = apply_operation(session, spec, approved=True, now=datetime.now(timezone.utc))
        except CleaningRejected as exc:
            st.error(str(exc))
            return
        st.session_state[SESSION_KEY] = updated
        st.session_state[PENDING_KEY] = None
        st.rerun()


def _render_history(session: CleaningSession) -> None:
    undo_col, reset_col = st.columns(2)
    with undo_col:
        if st.button("Undo last action", key="clean-undo", disabled=not session.audit):
            st.session_state[SESSION_KEY] = undo(session)
            st.session_state[PENDING_KEY] = None
            st.rerun()
    with reset_col:
        confirmed = st.checkbox("Restore every dataset to the original upload", key="clean-reset-confirm")
        if st.button("Reset to original", key="clean-reset"):
            if not confirmed:
                st.error("Check the reset box before restoring the original upload.")
            else:
                st.session_state[SESSION_KEY] = reset(session)
                st.session_state[PENDING_KEY] = None
                st.rerun()
    if not session.audit:
        st.caption("No cleaning operation has been approved in this session.")
        return
    st.subheader("Cleaning audit")
    st.caption("Counts and settings only. Fill constants and cell values are not listed.")
    st.table(
        pd.DataFrame(
            [
                {
                    "Order": entry.action_order,
                    "Dataset": entry.dataset_id,
                    "Operation": entry.operation,
                    "Columns": ", ".join(entry.columns),
                    "Parameters": entry.parameter_summary,
                    "Rows removed": entry.rows_removed,
                    "Cells changed": entry.cells_changed,
                    "Conversion failures": entry.conversion_failures,
                    "Missing cells before": entry.metrics_before.missing_cells,
                    "Missing cells after": entry.metrics_after.missing_cells,
                    "Extra copies before": entry.metrics_before.extra_duplicate_copies,
                    "Extra copies after": entry.metrics_after.extra_duplicate_copies,
                }
                for entry in session.audit
            ]
        )
    )


def _render_comparison(original: BatchReport, current: BatchReport) -> None:
    st.subheader("Before and after")
    st.caption("Before is the original upload. After is the working copy. These counts do not list cell values.")
    original_items = {item.dataset_id: item for item in original.datasets if item.report is not None}
    current_items = {item.dataset_id: item for item in current.datasets if item.report is not None}
    rows = []
    for dataset_id in list(dict.fromkeys([*original_items, *current_items])):
        before = original_items[dataset_id].report
        after = current_items.get(dataset_id)
        after_report = after.report if after is not None else None
        if before is None or after_report is None:
            continue
        rows.append(
            {
                "Dataset": dataset_id,
                "Rows before": before.data_rows,
                "Rows after": after_report.data_rows,
                "Duplicate rows before": before.exact_duplicates.rows_in_duplicate_groups.count,
                "Duplicate rows after": after_report.exact_duplicates.rows_in_duplicate_groups.count,
                "Extra copies before": before.exact_duplicates.extra_copies.count,
                "Extra copies after": after_report.exact_duplicates.extra_copies.count,
                "Missing cells before": sum(column.missing.count for column in before.columns),
                "Missing cells after": sum(column.missing.count for column in after_report.columns),
            }
        )
    if rows:
        st.table(pd.DataFrame(rows))
    _render_relationships(original, current)


def _render_relationships(original: BatchReport, current: BatchReport) -> None:
    st.subheader("Relationship suggestions")
    st.caption("Suggestions are recalculated after each approved operation. Orphan references are not deleted or repaired.")

    def _key(item: object) -> tuple[str, str, str, str]:
        return (
            str(getattr(item, "parent_dataset")),
            str(getattr(item, "parent_column")),
            str(getattr(item, "child_dataset")),
            str(getattr(item, "child_column")),
        )

    before = {_key(item): item for item in original.relationships}
    after = {_key(item): item for item in current.relationships}
    keys = sorted(set(before) | set(after))
    if not keys:
        st.caption("No relationship suggestion meets the evidence bar for the original upload or the working copy.")
        return
    table = []
    for key in keys:
        left = before.get(key)
        right = after.get(key)
        if left is None:
            status = "suggestion added"
        elif right is None:
            status = "suggestion removed"
        elif left.orphan_count != right.orphan_count:
            status = "orphan count changed"
        else:
            status = "unchanged"
        table.append(
            {
                "Parent": f"{key[0]} · {key[1]}",
                "Child": f"{key[2]} · {key[3]}",
                "Orphans before": left.orphan_count if left is not None else "",
                "Orphans after": right.orphan_count if right is not None else "",
                "Overlap before": f"{left.overlap_ratio:.2f}" if left is not None else "",
                "Overlap after": f"{right.overlap_ratio:.2f}" if right is not None else "",
                "Status": status,
            }
        )
    st.table(pd.DataFrame(table))


def _render_downloads(session: CleaningSession, original: BatchReport, current: BatchReport) -> None:
    st.subheader("Download cleaned data")
    st.caption("Cleaned files contain the working copy. The audit workbook does not contain cell values or fill constants.")
    for index, dataset_id in enumerate(session.order, start=1):
        item = session.copies[dataset_id]
        st.download_button(
            f"Download cleaned CSV for {dataset_id}",
            data=cleaned_dataset_csv(item.working),
            file_name=f"cleaned_{index}.csv",
            mime="text/csv",
            key=f"clean-csv-{index}",
        )
    st.download_button(
        "Download cleaned datasets (Excel)",
        data=cleaned_datasets_xlsx(session),
        file_name="cleaned_datasets.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="clean-xlsx",
    )
    st.download_button(
        "Download cleaning audit workbook",
        data=cleaning_audit_xlsx(session, original, current),
        file_name="cleaning_audit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="clean-audit",
    )
