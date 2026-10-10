"""Overview, Findings, and Investigation for the uploaded files.

These pages read the session upload. They do not load the banking findings
and they do not call a model unless the explanation control is used.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.recommendations_view import render_recommendations
from dashboard.session_store import ORIGINAL_VERSION_LABEL
from dashboard.upload_context import frames_for, open_upload_session, profile_batches, stored_rule_columns
from dashboard.upload_summary import (
    EMPTY_UPLOAD_MESSAGE,
    KIND_LABELS,
    LOAD_FAILURE_MESSAGE,
    UploadFinding,
    upload_findings,
    upload_overview,
)
from dashboard.workflow import section_label
from universal_dq.models import BatchReport


def render_uploaded_pages(page: str, version: str) -> None:
    """Show one uploaded-data page for the selected version."""
    st.caption(f"Active version: {version}")
    st.caption("Uploaded files stay in this session. They are not written to the project folder.")
    opened = open_upload_session(st.session_state)
    if opened is None:
        st.info(EMPTY_UPLOAD_MESSAGE)
        return
    if not opened.session.order:
        for item in opened.errors:
            if item.error:
                st.error(f"{item.dataset_id}: {item.error}")
        st.info(LOAD_FAILURE_MESSAGE)
        return
    required, key_columns = stored_rule_columns(st.session_state, opened.session, opened.errors)
    original, current = profile_batches(st.session_state, opened, required, key_columns)
    use_original = version == ORIGINAL_VERSION_LABEL
    batch = original if use_original else current
    frames = frames_for(opened.session, original=use_original)
    if page == "Findings":
        _render_findings(upload_findings(batch, frames))
    elif page == "Investigation":
        _render_investigation(batch, frames, opened.fingerprint, opened.session.generation, use_original)
    else:
        _render_overview(batch, frames)


def _render_overview(batch: BatchReport, frames: dict) -> None:
    st.markdown("### Data quality overview")
    st.caption(
        "Counts from the uploaded files. Candidate keys and relationships are suggestions. "
        "These totals are not a quality score."
    )
    summary = upload_overview(batch)
    first, second, third = st.columns(3)
    first.metric("Datasets analyzed", summary.datasets_analyzed)
    second.metric("Data rows", summary.rows)
    third.metric("Columns", summary.columns)
    fourth, fifth, sixth = st.columns(3)
    fourth.metric("Missing values", summary.missing_values)
    fifth.metric("Rows in duplicate groups", summary.duplicate_rows)
    sixth.metric("Extra exact copies", summary.extra_copies)
    seventh, eighth, ninth = st.columns(3)
    seventh.metric("Candidate keys", summary.candidate_keys)
    eighth.metric("Relationship suggestions", summary.relationship_suggestions)
    ninth.metric("Unmatched references", summary.unmatched_references)
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
                    }
                    for item in batch.candidate_keys
                ]
            )
        )
    else:
        st.caption("No column is both complete and unique across at least 2 data rows.")
    section_label("inferred", "Relationship suggestions")
    st.caption("Each link is a suggestion. It is not a confirmed foreign-key constraint.")
    if batch.relationships:
        st.table(
            pd.DataFrame(
                [
                    {
                        "Parent": f"{item.parent_dataset} · {item.parent_column}",
                        "Child": f"{item.child_dataset} · {item.child_column}",
                        "Unmatched references": item.orphan_count,
                        "Cardinality": item.cardinality,
                    }
                    for item in batch.relationships
                ]
            )
        )
    else:
        st.caption("No relationship met the evidence bar.")
    _render_measured_issues(upload_findings(batch, frames))


def _render_measured_issues(rows: list[UploadFinding]) -> None:
    """List measured issues already produced for this version."""
    st.markdown("**Measured issues**")
    st.caption("No severity is assigned. These rows are not a quality score.")
    if not rows:
        st.caption("No measured issue met the evidence bar for this version.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Dataset": item.dataset_id,
                    "Kind": KIND_LABELS[item.kind],
                    "Issue": item.issue,
                    "Column": item.column or "",
                    "Affected": item.affected_count,
                }
                for item in rows
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def _render_findings(rows: list[UploadFinding]) -> None:
    st.subheader("Findings")
    st.caption(
        "Observations, rules you selected, inferred mismatches, and deterministic recommendations. "
        "No severity is assigned on this page."
    )
    if not rows:
        st.info("No measured issue met the evidence bar for this version.")
        return
    datasets = sorted({item.dataset_id for item in rows})
    kinds = [KIND_LABELS[item.kind] for item in rows]
    kind_order = list(dict.fromkeys(kinds))
    dataset_choice = st.sidebar.multiselect(
        "Dataset",
        datasets,
        default=datasets,
        key="upload-findings-dataset",
    )
    kind_choice = st.sidebar.multiselect(
        "Kind",
        kind_order,
        default=kind_order,
        key="upload-findings-kind",
    )
    selected = [
        item
        for item in rows
        if item.dataset_id in dataset_choice and KIND_LABELS[item.kind] in kind_choice
    ]
    frame = pd.DataFrame(
        [
            {
                "Dataset": item.dataset_id,
                "Kind": KIND_LABELS[item.kind],
                "Issue": item.issue,
                "Column": item.column or "",
                "Affected": item.affected_count,
                "Denominator": item.denominator,
                "Evidence": item.evidence,
            }
            for item in selected
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    if frame.empty:
        st.info("No findings match the current filters.")
        return
    labels = [
        f"{index + 1}. {item.issue} · {item.dataset_id}" + (f" · {item.column}" if item.column else "")
        for index, item in enumerate(selected)
    ]
    choice = st.selectbox("Finding detail", labels)
    detail = selected[labels.index(str(choice))]
    section_label("confirmed", detail.issue)
    st.write(detail.evidence)
    st.caption(f"Affected count {detail.affected_count}. Denominator {detail.denominator}.")
    if detail.kind == "inferred_mismatch":
        st.caption("This mismatch comes from a relationship suggestion. It is not a confirmed foreign key.")
    if detail.kind == "recommendation":
        st.caption("Hypotheses for this recommendation are on the Investigation page.")


def _render_investigation(
    batch: BatchReport,
    frames: dict,
    fingerprint: str,
    generation: int,
    use_original: bool,
) -> None:
    st.subheader("Investigation")
    st.caption(
        "Confirmed counts stay separate from hypotheses. "
        "An explanation is requested only after you approve the summary shown here."
    )
    render_recommendations(
        batch,
        frames,
        fingerprint=fingerprint,
        generation=generation,
        mutate_explanations=not use_original,
    )
