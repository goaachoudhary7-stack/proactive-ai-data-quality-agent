"""AI SQL Studio, stage 1: catalog, descriptions, and consent.

The page shows the local file mapping and the exact summary. It does not
generate SQL, run SQL, or send the summary.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from dashboard.session_store import (
    ORIGINAL_VERSION_LABEL,
    VERSION_LABELS,
    VERSION_WIDGET_KEY,
    WORKING_VERSION_LABEL,
    remember_version,
    restore_version_widget,
)
from dashboard.upload_context import open_upload_session, profile_batches, stored_rule_columns
from dashboard.upload_summary import EMPTY_UPLOAD_MESSAGE, LOAD_FAILURE_MESSAGE
from studio.binding import drop_stale_query, make_binding
from studio.catalog import build_catalog, example_questions, types_from_batch
from studio.payload import (
    QUESTION_LIMIT,
    build_payload,
    consent_matches,
    payload_hash,
    payload_problems,
)
from studio.state import (
    CONSENT_KEY,
    QUESTION_MEMORY_KEY,
    QUESTION_REQUEST_KEY,
    QUESTION_WIDGET_KEY,
    column_description,
    dataset_description,
    description_map,
    prepare_question,
    set_column_description,
    set_dataset_description,
)

_VERSION_IDS = {
    ORIGINAL_VERSION_LABEL: "original",
    WORKING_VERSION_LABEL: "working",
}


def render_studio() -> None:
    """Show the dataset mapping, the summary, and the approval control."""
    st.markdown("### AI SQL Studio")
    st.caption(
        "This workspace uses the files uploaded in this session. "
        "The banking demonstration is separate and is not queried here."
    )
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

    restore_version_widget(st.session_state)
    version_label = st.radio("Uploaded data", list(VERSION_LABELS), key=VERSION_WIDGET_KEY)
    remember_version(st.session_state, str(version_label))
    version_id = _VERSION_IDS[str(version_label)]
    st.info(f"Active version: {version_label}")

    required, key_columns = stored_rule_columns(st.session_state, opened.session, opened.errors)
    original, current = profile_batches(st.session_state, opened, required, key_columns)
    batch = original if version_id == "original" else current
    for item in opened.errors:
        if item.error:
            st.error(f"{item.dataset_id}: {item.error}")

    catalog = build_catalog(
        opened.session,
        version=version_id,
        descriptions=_dataset_text(version_id),
        column_descriptions=_column_text(version_id),
        types=types_from_batch(batch),
    )
    _render_mapping(catalog)
    _render_description_inputs(catalog, version_id)
    catalog = build_catalog(
        opened.session,
        version=version_id,
        descriptions=_dataset_text(version_id),
        column_descriptions=_column_text(version_id),
        types=types_from_batch(batch),
    )
    _render_question(catalog)
    question = str(st.session_state.get(QUESTION_WIDGET_KEY, ""))
    payload = build_payload(catalog, question)
    binding = make_binding(
        catalog,
        fingerprint=opened.fingerprint,
        generation=opened.session.generation,
        payload=payload,
    )
    drop_stale_query(st.session_state, binding)
    _render_payload(payload)


def _dataset_text(version_id: str) -> dict[str, str]:
    datasets, _columns = description_map(st.session_state, version_id)
    return datasets


def _column_text(version_id: str) -> dict[str, dict[str, str]]:
    _datasets, columns = description_map(st.session_state, version_id)
    return columns


def _render_mapping(catalog) -> None:
    st.markdown("#### Dataset explorer")
    st.caption("SQL names stay on this page with the file, worksheet, and column they refer to.")
    rows: list[dict[str, object]] = []
    for dataset in catalog.datasets:
        if not dataset.columns:
            rows.append(_mapping_row(dataset, None))
            continue
        for column in dataset.columns:
            rows.append(_mapping_row(dataset, column))
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _mapping_row(dataset, column) -> dict[str, object]:
    return {
        "SQL table": dataset.table_ref,
        "SQL column": "" if column is None else column.column_ref,
        "File": dataset.source_name,
        "Worksheet": dataset.sheet_name or "",
        "Column": "" if column is None else column.column_name,
        "Type": "" if column is None else column.inferred_type,
    }


def _render_description_inputs(catalog, version_id: str) -> None:
    st.caption("Descriptions are optional. Only the text you enter here can be included in the summary.")
    for dataset in catalog.datasets:
        with st.expander(f"{dataset.table_ref} descriptions"):
            st.caption(f"File: {dataset.source_name}")
            if dataset.sheet_name:
                st.caption(f"Worksheet: {dataset.sheet_name}")
            dataset_key = f"studio-dataset-desc-{version_id}-{dataset.dataset_id}"
            if dataset_key not in st.session_state:
                st.session_state[dataset_key] = dataset_description(
                    st.session_state, version_id, dataset.dataset_id
                )
            entered = st.text_input("Dataset description", key=dataset_key, max_chars=120)
            set_dataset_description(st.session_state, version_id, dataset.dataset_id, entered)
            for column in dataset.columns:
                column_key = f"studio-column-desc-{version_id}-{dataset.dataset_id}-{column.column_name}"
                if column_key not in st.session_state:
                    st.session_state[column_key] = column_description(
                        st.session_state,
                        version_id,
                        dataset.dataset_id,
                        column.column_name,
                    )
                column_entered = st.text_input(
                    f"Description for {column.column_ref}",
                    key=column_key,
                    max_chars=120,
                    help=column.column_name,
                )
                set_column_description(
                    st.session_state,
                    version_id,
                    dataset.dataset_id,
                    column.column_name,
                    column_entered,
                )


def _render_question(catalog) -> None:
    st.markdown("#### Ask AI")
    prepare_question(st.session_state)
    if QUESTION_WIDGET_KEY not in st.session_state:
        st.session_state[QUESTION_WIDGET_KEY] = ""
    question = st.text_area("Question", key=QUESTION_WIDGET_KEY, max_chars=QUESTION_LIMIT)
    st.session_state[QUESTION_MEMORY_KEY] = question
    suggestions = example_questions(catalog)
    if suggestions:
        st.caption("Example questions fill the box. Choosing one does not send a summary.")
        for index, suggestion in enumerate(suggestions):
            if st.button(suggestion, key=f"studio-example-{index}"):
                st.session_state[QUESTION_REQUEST_KEY] = suggestion
                st.rerun()


def _render_payload(payload: dict) -> None:
    st.caption(
        "This is the exact summary that would be sent after you approve it. "
        "The question and any descriptions you write can contain sensitive information. "
        "Uploaded rows, file names, and column headers are left out of this summary."
    )
    st.code(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), language="json")
    problems = payload_problems(payload)
    for problem in problems:
        st.caption(problem)
    digest = payload_hash(payload)
    consent_key = f"dq-studio-consent-{digest}"
    if consent_key not in st.session_state:
        st.session_state[consent_key] = consent_matches(st.session_state.get(CONSENT_KEY), payload)
    agreed = st.checkbox(
        "I approve this exact summary",
        key=consent_key,
        disabled=bool(problems),
    )
    if problems or not agreed:
        st.session_state.pop(CONSENT_KEY, None)
        return
    st.session_state[CONSENT_KEY] = {"hash": digest}
    st.caption("This exact summary is approved. Nothing has been sent.")
