"""Navigation and session persistence for the upload workflow."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pytest
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ai_agents.investigator import InvestigatorConfigError, LlmSettings  # noqa: E402
from ai_agents.upload_explainer import explanation_available  # noqa: E402
from dashboard.cleaning_view import FINGERPRINT_KEY, PENDING_KEY, SESSION_KEY  # noqa: E402
from dashboard.session_store import (  # noqa: E402
    CLEAN_STEP_LABEL,
    KEY_COLUMNS_KEY,
    PAYLOAD_KEY,
    REQUIRED_KEY,
    STEP_MEMORY_KEY,
    STEP_REQUEST_KEY,
    UPLOADER_GENERATION_KEY,
    WORKFLOW_WIDGET_KEY,
    StoredUpload,
    apply_step_request,
    clear_uploaded_files,
    ensure_cleaning_state,
    entries_for,
    remember_step,
    resolve_default_sheets,
    should_replace_session,
    snapshot_uploads,
)
from universal_dq.batch import collect_datasets  # noqa: E402
from universal_dq.recommendations import AI_STORE_KEY  # noqa: E402


def test_changing_steps_keeps_the_same_cleaning_session() -> None:
    payload = [StoredUpload("people.csv", b"Code\nA\nB\n")]
    loaded = collect_datasets(entries_for(payload))
    state: dict = {}
    first = ensure_cleaning_state(state, payload, loaded)
    session = state[SESSION_KEY]
    remember_step(state, "2. Analyze")
    remember_step(state, "5. Export")
    second = ensure_cleaning_state(state, payload, loaded)
    assert first == second
    assert state[SESSION_KEY] is session
    assert should_replace_session(state, first) is False


def test_snapshot_keeps_worksheets_until_the_bytes_change() -> None:
    existing = [StoredUpload("book.xlsx", b"abc", ("Sheet1",))]
    assert snapshot_uploads(existing, [("book.xlsx", b"abc")]) == existing
    replaced = snapshot_uploads(existing, [("book.xlsx", b"xyz")])
    assert replaced[0].sheets is None
    assert replaced[0].content == b"xyz"


def test_explicit_empty_worksheet_selection_is_preserved() -> None:
    buffer = BytesIO()
    workbook = Workbook()
    workbook.active.title = "Alpha"
    workbook.create_sheet("Beta")
    workbook.save(buffer)
    content = buffer.getvalue()
    defaulted = resolve_default_sheets(StoredUpload("book.xlsx", content))
    assert defaulted.sheets == ("Alpha",)
    explicit = resolve_default_sheets(StoredUpload("book.xlsx", content, tuple()))
    assert explicit.sheets == tuple()


def test_clearing_uploads_is_separate_from_the_remembered_step() -> None:
    state = {
        STEP_MEMORY_KEY: "4. Clean",
        SESSION_KEY: object(),
        PAYLOAD_KEY: [StoredUpload("people.csv", b"Code\nA\n")],
        FINGERPRINT_KEY: "abc",
        PENDING_KEY: object(),
        AI_STORE_KEY: {"items": {}},
        REQUIRED_KEY: ["Code"],
        KEY_COLUMNS_KEY: ["Code"],
    }
    clear_uploaded_files(state)
    assert SESSION_KEY not in state
    assert PAYLOAD_KEY not in state
    assert FINGERPRINT_KEY not in state
    assert PENDING_KEY not in state
    assert AI_STORE_KEY not in state
    assert state[STEP_MEMORY_KEY] == "4. Clean"
    assert state[UPLOADER_GENERATION_KEY] == 1


def test_step_request_does_not_change_the_fingerprint() -> None:
    state = {FINGERPRINT_KEY: "abc", STEP_REQUEST_KEY: CLEAN_STEP_LABEL}
    apply_step_request(state)
    assert state[WORKFLOW_WIDGET_KEY] == CLEAN_STEP_LABEL
    assert state[FINGERPRINT_KEY] == "abc"
    assert STEP_REQUEST_KEY not in state


def test_explanation_request_stays_off_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(environ: object = None) -> LlmSettings:
        raise InvestigatorConfigError("missing")

    monkeypatch.setattr("ai_agents.investigator.resolve_llm_settings", missing)
    assert explanation_available() is False


def test_explanation_request_is_allowed_when_settings_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ai_agents.investigator.resolve_llm_settings",
        lambda environ=None: LlmSettings(model="gemini/demo", api_key_env="PROVIDER_KEY"),
    )
    assert explanation_available() is True


def _open_analyzer():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(SRC / "dashboard" / "app.py"), default_timeout=60)
    app.run()
    next(item for item in app.radio if item.label == "Page").set_value("Analyze a file").run()
    return app


def _workflow(app):
    return next(item for item in app.radio if item.label == "Workflow")


def test_workflow_selector_stays_on_the_page_and_privacy_stays_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    app = _open_analyzer()
    assert [item.label for item in app.sidebar.radio] == ["Active data source", "Page"]
    workflow = _workflow(app)
    assert list(workflow.options) == ["1. Upload", "2. Analyze", "3. Insights", "4. Clean", "5. Export"]
    privacy = next(item for item in app.expander if item.label == "Privacy & processing details")
    details = " ".join(block.value for block in privacy.markdown)
    assert "not sent to Gemini, CrewAI, or another external model" in details
    assert "not written to the project folder" in details
    assert "Uploaded rows stay in this session" in " ".join(item.value for item in app.caption)
    workflow.set_value("4. Clean").run()
    assert app.session_state[STEP_MEMORY_KEY] == "4. Clean"
    workflow.set_value("1. Upload").run()
    assert app.session_state[STEP_MEMORY_KEY] == "1. Upload"


def _button(app, label: str):
    return next(item for item in app.button if item.label == label)


def test_workflow_navigation_preserves_session_preview_and_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    import dashboard.recommendations_view as recommendations_view

    monkeypatch.setattr(recommendations_view, "explanation_available", lambda: False)
    app = _open_analyzer()
    app.file_uploader[0].upload("notes.csv", b"Code\nA\nA\n", "text/csv").run()
    fingerprint = app.session_state[FINGERPRINT_KEY]
    generation = app.session_state[SESSION_KEY].generation
    for label in ("2. Analyze", "3. Insights", "4. Clean", "5. Export", "1. Upload"):
        _workflow(app).set_value(label).run()
        assert app.session_state[FINGERPRINT_KEY] == fingerprint
        assert app.session_state[SESSION_KEY].generation == generation
        assert not app.exception

    _workflow(app).set_value("4. Clean").run()
    _button(app, "Preview").click().run()
    assert app.session_state[PENDING_KEY] is not None
    _workflow(app).set_value("2. Analyze").run()
    _workflow(app).set_value("4. Clean").run()
    assert app.session_state[PENDING_KEY] is not None
    assert app.session_state[FINGERPRINT_KEY] == fingerprint
    assert _button(app, "Reset to original") is not None

    _workflow(app).set_value("5. Export").run()
    labels = {item.label for item in app.download_button}
    assert "Download data-quality report JSON" in labels
    assert "Download data-quality report CSV" in labels
    assert "Download consolidated Excel report" in labels
    assert "Download Excel report with recommendations" in labels
    assert "Download cleaned CSV for notes.csv" in labels
    assert "Download cleaned datasets (Excel)" in labels
    assert "Download cleaning audit workbook" in labels
    assert not app.exception


def test_recommendation_opens_clean_without_applying(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    import dashboard.recommendations_view as recommendations_view

    monkeypatch.setattr(recommendations_view, "explanation_available", lambda: False)
    app = _open_analyzer()
    app.file_uploader[0].upload("notes.csv", b"Code\nA\n\nB\n", "text/csv").run()
    _workflow(app).set_value("3. Insights").run()
    explain = [item for item in app.button if item.label == "Explain this recommendation"]
    assert explain
    assert all(item.disabled for item in explain)
    _button(app, "Fill with a constant in the cleaning form").click().run()
    assert _workflow(app).value == "4. Clean"
    assert app.session_state[SESSION_KEY].generation == 0
    assert app.session_state[PENDING_KEY] is None
    assert not app.exception


def test_removing_files_requires_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    app = _open_analyzer()
    app.file_uploader[0].upload("notes.csv", b"Code\nA\nB\n", "text/csv").run()
    fingerprint = app.session_state[FINGERPRINT_KEY]
    _button(app, "Remove uploaded files").click().run()
    assert app.session_state[FINGERPRINT_KEY] == fingerprint
    assert any("confirmation" in item.value.casefold() for item in app.error)
    app.checkbox[0].set_value(True).run()
    _button(app, "Remove uploaded files").click().run()
    assert FINGERPRINT_KEY not in app.session_state
    assert SESSION_KEY not in app.session_state
    assert not app.exception
