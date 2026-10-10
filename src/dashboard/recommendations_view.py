"""Recommendation panel for the upload page.

Deterministic recommendations render with the profile. An explanation is
requested only after the summary checkbox is checked.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ai_agents.upload_explainer import explain_recommendation, explanation_available
from dashboard.session_store import CLEAN_STEP_LABEL, PAGE_REQUEST_KEY, STEP_REQUEST_KEY
from universal_dq.models import BatchReport
from universal_dq.recommendations import (
    AI_STORE_KEY,
    PREFILL_KEY,
    Recommendation,
    StoredExplanation,
    build_recommendations,
    evidence_token,
    explanation_payload,
    explanations_current,
)
from universal_dq.report import report_to_xlsx


def recommendation_workbook(
    batch: BatchReport,
    frames: dict,
    *,
    fingerprint: str,
    generation: int,
) -> bytes:
    """Build the separate workbook that adds recommendation sheets."""
    recommendations, store = _prepare(
        batch,
        frames,
        fingerprint=fingerprint,
        generation=generation,
        mutate_explanations=True,
    )
    explanations = [item for item in store["items"].values() if isinstance(item, StoredExplanation)]
    return report_to_xlsx(batch, recommendations, explanations)


def render_recommendations(
    batch: BatchReport,
    frames: dict,
    *,
    fingerprint: str,
    generation: int,
    mutate_explanations: bool = True,
) -> None:
    """Show local recommendations and the optional explanation control."""
    recommendations, store = _prepare(
        batch,
        frames,
        fingerprint=fingerprint,
        generation=generation,
        mutate_explanations=mutate_explanations,
    )
    st.subheader("Recommendations")
    st.caption(
        "Confirmed counts come from this profile. "
        "Hypotheses and possible implications are separate from those counts. "
        "Opening Clean fills the form and does not apply a change."
    )
    if not recommendations:
        st.caption("No recommendation met the evidence bar.")
        return
    _render_name_map(recommendations)
    available = explanation_available()
    for item in recommendations:
        _render_card(item, store, available=available)


def _prepare(
    batch: BatchReport,
    frames: dict,
    *,
    fingerprint: str,
    generation: int,
    mutate_explanations: bool,
) -> tuple[list[Recommendation], dict]:
    recommendations = build_recommendations(batch, frames)
    store = explanation_store_for_view(
        st.session_state,
        fingerprint=fingerprint,
        generation=generation,
        token=evidence_token(recommendations),
        mutate=mutate_explanations,
    )
    return recommendations, store


def explanation_store_for_view(
    state: dict,
    *,
    fingerprint: str,
    generation: int,
    token: str,
    mutate: bool,
) -> dict:
    """Return explanations that match this version. Leave a mismatched store in place when asked."""
    current = state.get(AI_STORE_KEY)
    if explanations_current(current, fingerprint=fingerprint, generation=generation, token=token):
        return current
    if not mutate:
        return {"fingerprint": fingerprint, "generation": generation, "token": token, "items": {}}
    fresh = {"fingerprint": fingerprint, "generation": generation, "token": token, "items": {}}
    state[AI_STORE_KEY] = fresh
    return fresh


def _render_name_map(recommendations: list[Recommendation]) -> None:
    rows = []
    seen: set[tuple[str, str]] = set()
    for item in recommendations:
        key = (item.dataset_ref, item.dataset_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"Shown on this page": item.dataset_id, "Summary identifier": item.dataset_ref})
    st.caption("The summary identifiers below are what an approved explanation would use. File names stay on this page.")
    st.table(pd.DataFrame(rows))


def _render_card(item: Recommendation, store: dict, *, available: bool) -> None:
    stored = store["items"].get(item.recommendation_id)
    if not isinstance(stored, StoredExplanation):
        stored = None
    with st.container(border=True):
        _render_confirmed(item)
        _render_causes(item, stored)
        _render_next_checks(item)
        if stored is not None and stored.status == "ready":
            _render_original(stored)
        _render_explanation_request(item, store, available=available)


def _render_confirmed(item: Recommendation) -> None:
    subject = item.column or item.dataset_id
    title = f"{_issue_label(item.issue_type)} in {subject}"
    with st.container(gap="xxsmall"):
        st.markdown(f"##### {_markdown_text(title)}")
        st.caption(_markdown_text(item.dataset_id))
        st.badge("Local analysis", color="gray")
        share = _share_text(item.affected_count, item.denominator)
        boxes = st.columns(3 if share is not None else 2, gap="small", wrap=True)
        boxes[0].metric("Affected records", item.affected_count)
        boxes[1].metric(_denominator_label(item), item.denominator)
        if share is not None:
            boxes[2].metric("Affected percentage", share)
        if item.evidence:
            st.caption(_markdown_text(item.evidence))


def _render_causes(item: Recommendation, stored: StoredExplanation | None) -> None:
    with st.container(gap="xxsmall"):
        st.markdown("**Possible causes to investigate**")
        if item.why_it_matters:
            st.caption("Local note. Not a verified cause.")
            st.text(item.why_it_matters)
        if item.hypotheses:
            st.caption("Local hypotheses. Unverified.")
            for hypothesis in item.hypotheses:
                st.text(hypothesis)
        if stored is None:
            return
        if stored.status == "error":
            st.error(stored.error or "The explanation could not be completed.")
            return
        _render_ai_narrative(stored)


def _render_ai_narrative(stored: StoredExplanation) -> None:
    explanation = stored.explanation or ""
    implication = stored.possible_implication or ""
    hypotheses = [item for item in stored.hypotheses if item]
    if not explanation and not implication and not hypotheses:
        return
    st.caption("AI-generated and unverified.")
    if explanation:
        st.markdown("**AI-generated explanation**")
        st.text(explanation)
    if hypotheses:
        st.markdown("**AI-generated hypotheses**")
        for hypothesis in hypotheses:
            st.text(hypothesis)
    if implication:
        st.markdown("**AI-generated possible implication**")
        st.text(implication)


def _render_next_checks(item: Recommendation) -> None:
    steps = [step for step in item.verification_steps if step.strip()]
    options = [option for option in item.remediation if option.note.strip() or option.operation]
    if not steps and not options:
        return
    with st.container(gap="xxsmall"):
        st.markdown("**Recommended next checks**")
        st.caption("From the local recommendation.")
        for step in steps:
            st.text(step)
        if item.approval:
            st.caption(f"Approval required: {item.approval.replace('_', ' ')}.")
        for index, option in enumerate(item.remediation):
            if option.note.strip():
                st.text(option.note)
            if option.operation and st.button(option.label, key=f"prefill-{item.recommendation_id}-{index}"):
                st.session_state[PREFILL_KEY] = {
                    "dataset_id": item.dataset_id,
                    "operation": option.operation,
                    "columns": list(option.columns),
                }
                st.session_state[STEP_REQUEST_KEY] = CLEAN_STEP_LABEL
                st.session_state[PAGE_REQUEST_KEY] = "Analyze a file"
                st.rerun()


def _render_original(stored: StoredExplanation) -> None:
    parts = [stored.explanation or "", stored.possible_implication or "", *stored.hypotheses]
    if not any(part.strip() for part in parts):
        return
    with st.expander("Original AI explanation"):
        st.caption("AI-generated and unverified. The wording is unchanged.")
        if stored.explanation:
            st.text(stored.explanation)
        if stored.possible_implication:
            st.text(stored.possible_implication)
        for hypothesis in stored.hypotheses:
            if hypothesis:
                st.text(hypothesis)


def _issue_label(issue_type: str) -> str:
    words = issue_type.replace("_", " ")
    return words[:1].upper() + words[1:] if words else words


def _denominator_label(item: Recommendation) -> str:
    """Name the existing denominator. The count itself is not recalculated."""
    counts = item.evidence_counts
    if counts.get("child_non_missing") == item.denominator:
        return "Non-missing child values"
    if counts.get("numeric_values") == item.denominator:
        return "Numeric values"
    if counts.get("rows_with_complete_key") == item.denominator:
        return "Rows with a complete key"
    if counts.get("non_missing_count") == item.denominator and "data_rows" not in counts:
        return "Non-missing values"
    if counts.get("data_rows") == item.denominator:
        return "Data rows"
    return "Denominator"


def _markdown_text(value: str) -> str:
    """Keep a measured name as text inside Markdown. HTML stays disabled."""
    escaped = value.replace("\\", "\\\\")
    for char in ("*", "_", "`", "[", "]"):
        escaped = escaped.replace(char, "\\" + char)
    return escaped


def _share_text(affected: int, denominator: int) -> str | None:
    """Percent of the local denominator. Omitted when that denominator is zero."""
    if denominator <= 0:
        return None
    percent = (affected / denominator) * 100
    rounded = round(percent, 1)
    if float(rounded).is_integer():
        return f"{int(rounded)}%"
    return f"{rounded:.1f}%"


def _render_explanation_request(item: Recommendation, store: dict, *, available: bool) -> None:
    with st.expander("Explain with AI"):
        payload = explanation_payload(item)
        st.caption("This is the exact summary that would be sent after you approve it.")
        st.code(json.dumps(payload, indent=2, sort_keys=True), language="json")
        if not available:
            st.caption(
                "An explanation cannot be requested until an API key is configured. "
                "The offline recommendations are unchanged."
            )
        consent = st.checkbox(
            "I approve sending this summary",
            key=f"consent-{item.recommendation_id}",
            disabled=not available,
        )
        if st.button(
            "Explain this recommendation",
            key=f"explain-{item.recommendation_id}",
            disabled=not available,
        ):
            if not available:
                st.caption(
                    "An explanation cannot be requested until an API key is configured. "
                    "The offline recommendations are unchanged."
                )
                return
            if not consent:
                st.error("Check the approval box before sending the summary.")
            else:
                result = explain_recommendation(
                    payload,
                    consent=True,
                    recommendation_id=item.recommendation_id,
                    dataset_ref=item.dataset_ref,
                    column_ref=item.column_ref,
                )
                store["items"][item.recommendation_id] = result
                st.session_state[AI_STORE_KEY] = store
                st.rerun()
