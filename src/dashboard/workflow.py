"""Shared layout for the five upload steps.

Labels distinguish confirmed measurements, inferred suggestions, and
hypotheses. The relationship diagram repeats suggestions already returned
by the inference engine.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from dashboard.session_store import SELECTED_DATASET_KEY
from universal_dq.models import BatchReport, CandidateRelationship

_KIND_COLOR = {
    "confirmed": "#6ee7b7",
    "inferred": "#fbbf24",
    "hypothesis": "#c4b5fd",
}
_KIND_TEXT = {
    "confirmed": "Confirmed",
    "inferred": "Inferred suggestion",
    "hypothesis": "Hypothesis",
}


def section_label(kind: str, title: str) -> None:
    """Show a short kind label above a section title."""
    color = _KIND_COLOR[kind]
    kind_text = _KIND_TEXT[kind]
    st.markdown(
        (
            f"<div style='border-left:4px solid {color};padding:2px 10px;margin:10px 0 4px 0'>"
            f"<span style='color:{color};font-size:0.8rem'>{kind_text}</span>"
            f"<div style='font-weight:600'>{html.escape(title)}</div></div>"
        ),
        unsafe_allow_html=True,
    )


def render_step_heading(step_id: str) -> None:
    """Show which of the five steps is open."""
    titles = {
        "upload": "Upload",
        "analyze": "Analyze",
        "insights": "Insights",
        "clean": "Clean",
        "export": "Export",
    }
    notes = {
        "upload": "Add CSV or Excel files and choose worksheets. Files stay in this session.",
        "analyze": "Confirmed counts from the original upload. No raw rows are listed here.",
        "insights": "Inferred suggestions and hypotheses sit apart from confirmed counts.",
        "clean": "Preview, then approve. Only the working copy changes, and orphan references stay as they are.",
        "export": "Original reports, the recommendation workbook, and cleaned files are separate downloads.",
    }
    st.subheader(titles[step_id])
    st.caption(notes[step_id])


def render_inventory(batch: BatchReport) -> str | None:
    """Show dataset names and counts, and return the selected dataset id."""
    st.markdown("**Dataset inventory**")
    st.caption("Row and column counts come from the working copy. Cell values are not shown.")
    if not batch.datasets:
        st.caption("No dataset is loaded.")
        return None
    rows = [
        {
            "Dataset": item.dataset_id,
            "File": item.source_name,
            "Worksheet": item.sheet_name or "",
            "Rows": item.report.data_rows if item.report is not None else 0,
            "Columns": item.report.column_count if item.report is not None else 0,
            "Status": "analyzed" if item.report is not None else "error",
        }
        for item in batch.datasets
    ]
    st.table(pd.DataFrame(rows))
    ids = [item.dataset_id for item in batch.datasets]
    widget_key = "dq-selected-dataset"
    remembered = st.session_state.get(SELECTED_DATASET_KEY)
    if widget_key not in st.session_state and remembered in ids:
        st.session_state[widget_key] = remembered
    if st.session_state.get(widget_key) not in ids:
        st.session_state[widget_key] = ids[0]
    selected = st.selectbox("Dataset", ids, key=widget_key)
    st.session_state[SELECTED_DATASET_KEY] = selected
    return str(selected)


def render_relationship_diagram(relationships: list[CandidateRelationship]) -> None:
    """Draw only relationships the inference engine already returned."""
    section_label("inferred", "Relationship suggestions")
    st.caption(
        "Each link is a suggestion from column-name similarity and value overlap. "
        "It is not a confirmed foreign-key constraint."
    )
    if not relationships:
        st.caption(
            "No relationship met the evidence bar. A suggestion needs a complete unique column, "
            "a similarly named column in another dataset, and at least 80% of the child values "
            "found in the parent column."
        )
        return
    blocks: list[str] = []
    for item in relationships:
        parent = html.escape(f"{item.parent_dataset} · {item.parent_column}")
        child = html.escape(f"{item.child_dataset} · {item.child_column}")
        detail = html.escape(f"Suggestion · {item.cardinality} · orphans {item.orphan_count}")
        blocks.append(
            "<div style='border:1px solid #8b7cf7;border-radius:8px;padding:8px 12px;margin:6px 0'>"
            f"{parent} &rarr; {child}<br><span style='color:#c4b5fd'>{detail}</span></div>"
        )
        st.markdown(
            f"Suggestion: {item.parent_dataset} · {item.parent_column} → "
            f"{item.child_dataset} · {item.child_column} "
            f"({item.cardinality}; orphans {item.orphan_count})"
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)
