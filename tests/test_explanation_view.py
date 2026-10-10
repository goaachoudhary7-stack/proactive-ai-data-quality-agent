"""Presentation tests for the uploaded recommendation card.

These tests render local recommendations and saved explanation text.
They do not call Gemini.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dashboard.recommendations_view import _denominator_label, _share_text  # noqa: E402
from dashboard.session_store import PREFILL_KEY  # noqa: E402
from universal_dq.batch import LoadedDataset, build_batch_report  # noqa: E402
from universal_dq.recommendations import (  # noqa: E402
    AI_STORE_KEY,
    Recommendation,
    RemediationOption,
    StoredExplanation,
    build_recommendations,
    evidence_token,
    explanation_payload,
)

COLUMN = "<img src=x>"
SECRET = "SECRET-CELL"
DATASET = "people.csv"
EXPLANATION = "dataset_1 column_1 <img src=x> stays as written"
HYPOTHESIS = "Hypothesis: dataset_1 lost a value."
IMPLICATION = "Possible effect for column_1."


def _batch():
    frame = pd.DataFrame(
        {COLUMN: ["AA", None, "AA", "BB"], "Note": ["1", "2", SECRET, "4"]},
        dtype=object,
    )
    loaded = [LoadedDataset(DATASET, DATASET, "csv", None, frame, None)]
    batch = build_batch_report(loaded)
    frames = {DATASET: frame}
    return batch, frames


def _missing(batch, frames) -> Recommendation:
    recommendations = build_recommendations(batch, frames)
    return next(item for item in recommendations if item.issue_type == "missing_values")


def _page(batch, frames) -> None:
    from dashboard.recommendations_view import render_recommendations

    render_recommendations(batch, frames, fingerprint="fp", generation=1)


def _render(batch, frames, store: dict | None = None):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_page, args=(batch, frames), default_timeout=30)
    if store is not None:
        app.session_state[AI_STORE_KEY] = store
    app.run()
    assert not app.exception
    return app


def _texts(app) -> list[str]:
    return [item.value for item in app.text]


def _blocked(*_args, **_kwargs):
    raise AssertionError("explanation requested")


def _no_gemini(_payload):
    raise AssertionError("gemini called")


def _store(recommendations: list[Recommendation], explanation: StoredExplanation) -> dict:
    return {
        "fingerprint": "fp",
        "generation": 1,
        "token": evidence_token(recommendations),
        "items": {explanation.recommendation_id: explanation},
    }


def _saved(item: Recommendation, **kwargs) -> StoredExplanation:
    body = {
        "recommendation_id": item.recommendation_id,
        "dataset_ref": item.dataset_ref,
        "column_ref": item.column_ref,
        "status": "ready",
    }
    body.update(kwargs)
    return StoredExplanation(**body)


def test_share_uses_the_local_counts() -> None:
    assert _share_text(1, 4) == "25%"
    assert _share_text(1, 3) == "33.3%"
    assert _share_text(0, 0) is None


def _markdown(app) -> str:
    return "\n".join(block.value for block in app.markdown)


def test_confirmed_evidence_uses_local_labels_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dashboard.recommendations_view.explanation_available", lambda: False)
    monkeypatch.setattr("dashboard.recommendations_view.explain_recommendation", _blocked)
    monkeypatch.setattr("ai_agents.upload_explainer.gemini_narrative_source", _no_gemini)
    batch, frames = _batch()
    item = _missing(batch, frames)
    app = _render(batch, frames)
    texts = _texts(app)
    markdown = _markdown(app)
    assert f"Missing values in {COLUMN}" in markdown
    assert all(block.allow_html is False for block in app.markdown)
    assert DATASET in [caption.value for caption in app.caption]
    assert item.evidence in [caption.value for caption in app.caption]
    assert "Local analysis" in markdown
    metrics = {(metric.label, metric.value) for metric in app.metric}
    assert ("Affected records", "1") in metrics
    assert ("Data rows", "4") in metrics
    assert ("Affected percentage", "25%") in metrics
    assert _denominator_label(item) == "Data rows"
    assert "Possible causes to investigate" in markdown
    assert "Hypothesis: the value was never collected." in texts
    assert any("Local hypotheses. Unverified." in caption.value for caption in app.caption)
    assert "Recommended next checks" in markdown
    assert "Compare the missing count with the column profile on this page." in texts
    assert any("From the local recommendation." in caption.value for caption in app.caption)
    payload = json.loads(app.code[0].value)
    assert payload == explanation_payload(item)
    assert COLUMN not in app.code[0].value
    assert SECRET not in app.code[0].value
    assert DATASET not in app.code[0].value
    explain = next(button for button in app.button if button.label == "Explain this recommendation")
    approval = next(box for box in app.checkbox if box.label == "I approve sending this summary")
    assert explain.disabled
    assert approval.disabled
    next(button for button in app.button if button.label == "Fill with a constant in the cleaning form").click().run()
    assert app.session_state[PREFILL_KEY]["dataset_id"] == DATASET
    assert app.session_state[PREFILL_KEY]["columns"] == [COLUMN]


def test_ready_explanation_keeps_gemini_wording_separate_from_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dashboard.recommendations_view.explanation_available", lambda: False)
    monkeypatch.setattr("ai_agents.upload_explainer.gemini_narrative_source", _no_gemini)
    batch, frames = _batch()
    recommendations = build_recommendations(batch, frames)
    item = next(entry for entry in recommendations if entry.issue_type == "missing_values")
    saved = _saved(
        item,
        explanation=EXPLANATION,
        possible_implication=IMPLICATION,
        hypotheses=[HYPOTHESIS],
    )
    app = _render(batch, frames, _store(recommendations, saved))
    texts = _texts(app)
    assert EXPLANATION in texts
    assert HYPOTHESIS in texts
    assert IMPLICATION in texts
    assert f"Missing values in {COLUMN}" in _markdown(app)
    assert ("Affected records", "1") in {(metric.label, metric.value) for metric in app.metric}
    assert ("Affected percentage", "25%") in {(metric.label, metric.value) for metric in app.metric}
    labels = [block.value for block in app.markdown]
    assert "**AI-generated explanation**" in labels
    assert "**AI-generated hypotheses**" in labels
    assert "**AI-generated possible implication**" in labels
    assert any("AI-generated and unverified." in caption.value for caption in app.caption)
    original = next(block for block in app.expander if block.label == "Original AI explanation")
    original_text = [entry.value for entry in original.text]
    assert EXPLANATION in original_text
    assert HYPOTHESIS in original_text
    assert IMPLICATION in original_text
    assert all(block.allow_html is False for block in app.markdown)
    assert EXPLANATION not in labels


def test_incomplete_and_failed_explanations_leave_the_local_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dashboard.recommendations_view.explanation_available", lambda: False)
    batch, frames = _batch()
    recommendations = build_recommendations(batch, frames)
    item = next(entry for entry in recommendations if entry.issue_type == "missing_values")
    empty = _render(
        batch,
        frames,
        _store(recommendations, _saved(item, explanation=None, possible_implication=None, hypotheses=[])),
    )
    empty_labels = [block.value for block in empty.markdown]
    assert "**AI-generated explanation**" not in empty_labels
    assert "Original AI explanation" not in [block.label for block in empty.expander]
    assert "Hypothesis: the value was never collected." in _texts(empty)
    assert "Recommended next checks" in _markdown(empty)

    failed = _render(
        batch,
        frames,
        _store(
            recommendations,
            _saved(
                item,
                status="error",
                error="The explanation service timed out. The offline recommendations are unchanged.",
                explanation=None,
                hypotheses=[],
            ),
        ),
    )
    assert any("timed out" in error.value for error in failed.error)
    assert "**AI-generated explanation**" not in [block.value for block in failed.markdown]
    assert "Hypothesis: the value was never collected." in _texts(failed)
    assert ("Affected records", "1") in {(metric.label, metric.value) for metric in failed.metric}


def test_empty_next_checks_are_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    item = Recommendation(
        recommendation_id="rec-empty",
        issue_type="missing_values",
        dataset_id=DATASET,
        dataset_ref="dataset_1",
        column=COLUMN,
        column_ref="column_1",
        affected_count=0,
        denominator=0,
        evidence_counts={"missing_count": 0},
        evidence="No measured gap.",
        why_it_matters="Local implication.",
        hypotheses=["Hypothesis: local only."],
        verification_steps=["   "],
        remediation=[],
        approval="user_approval",
    )
    monkeypatch.setattr("dashboard.recommendations_view.explanation_available", lambda: False)
    monkeypatch.setattr("dashboard.recommendations_view.build_recommendations", lambda batch, frames: [item])
    batch, frames = _batch()
    app = _render(batch, frames)
    assert "Recommended next checks" not in _markdown(app)
    assert "Affected percentage" not in [metric.label for metric in app.metric]
    assert f"Missing values in {COLUMN}" in _markdown(app)
    assert all(block.allow_html is False for block in app.markdown)


def test_consent_is_required_and_the_payload_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake(payload, *, consent, recommendation_id="", dataset_ref="", column_ref=None):
        calls.append(
            {
                "payload": payload,
                "consent": consent,
                "recommendation_id": recommendation_id,
                "dataset_ref": dataset_ref,
                "column_ref": column_ref,
            }
        )
        return StoredExplanation(
            recommendation_id=recommendation_id,
            dataset_ref=dataset_ref,
            column_ref=column_ref,
            status="error",
            error="The explanation service reported a quota limit. The offline recommendations are unchanged.",
        )

    monkeypatch.setattr("dashboard.recommendations_view.explanation_available", lambda: True)
    monkeypatch.setattr("dashboard.recommendations_view.explain_recommendation", fake)
    monkeypatch.setattr("ai_agents.upload_explainer.gemini_narrative_source", _no_gemini)
    batch, frames = _batch()
    item = _missing(batch, frames)
    app = _render(batch, frames)
    assert json.loads(app.code[0].value) == explanation_payload(item)
    next(button for button in app.button if button.label == "Explain this recommendation").click().run()
    assert calls == []
    assert any("approval box" in error.value.casefold() for error in app.error)
    next(box for box in app.checkbox if box.label == "I approve sending this summary").set_value(True).run()
    next(button for button in app.button if button.label == "Explain this recommendation").click().run()
    assert len(calls) == 1
    assert calls[0]["consent"] is True
    assert calls[0]["payload"] == explanation_payload(item)
    assert calls[0]["recommendation_id"] == item.recommendation_id
    assert COLUMN not in json.dumps(calls[0]["payload"])
    assert any("quota" in error.value for error in app.error)
    assert "Hypothesis: the value was never collected." in _texts(app)


def test_uploaded_insights_page_shows_the_three_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("dashboard.recommendations_view.explanation_available", lambda: False)
    monkeypatch.setattr("ai_agents.upload_explainer.gemini_narrative_source", _no_gemini)
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(SRC / "dashboard" / "app.py"), default_timeout=60)
    app.run()
    next(item for item in app.radio if item.label == "Page").set_value("Analyze a file").run()
    content = f"{COLUMN},Note\nAA,1\n,2\nAA,{SECRET}\nBB,4\n".encode()
    app.file_uploader[0].upload("people.csv", content, "text/csv").run()
    next(item for item in app.radio if item.label == "Workflow").set_value("3. Insights").run()
    assert not app.exception
    markdown = _markdown(app)
    assert f"Missing values in {COLUMN}" in markdown
    assert "Possible causes to investigate" in markdown
    assert "Recommended next checks" in markdown
    assert "Local analysis" in markdown
    named = [block for block in app.markdown if COLUMN in block.value]
    assert named
    assert all(block.allow_html is False for block in named)
    assert "Explain with AI" in [block.label for block in app.expander]
    explain = [button for button in app.button if button.label == "Explain this recommendation"]
    assert explain
    assert all(button.disabled for button in explain)
