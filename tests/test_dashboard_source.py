"""Uploaded files on Overview, Findings, and Investigation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dashboard.cleaning_view import FINGERPRINT_KEY, PENDING_KEY, SESSION_KEY  # noqa: E402
from dashboard.recommendations_view import explanation_store_for_view  # noqa: E402
from dashboard.session_store import (  # noqa: E402
    BANKING_SOURCE_LABEL,
    FILE_IDENTITY_KEY,
    PAGE_LABELS,
    PAGE_WIDGET_KEY,
    PAYLOAD_KEY,
    SOURCE_MEMORY_KEY,
    UPLOADED_SOURCE_LABEL,
    StoredUpload,
    fallback_hidden_page,
    remember_source,
    should_select_uploaded_source,
)
from dashboard.upload_context import frames_for, open_upload_session, profile_batches  # noqa: E402
from dashboard.upload_summary import (  # noqa: E402
    EMPTY_UPLOAD_MESSAGE,
    upload_findings,
    upload_overview,
)
from universal_dq.batch import collect_datasets  # noqa: E402
from universal_dq.cleaning import apply_operation  # noqa: E402
from universal_dq.models import OperationSpec  # noqa: E402
from universal_dq.recommendations import AI_STORE_KEY, build_recommendations, evidence_token  # noqa: E402


def test_source_switches_only_after_a_new_successful_upload() -> None:
    assert should_select_uploaded_source(bytes_changed=True, dataset_count=1) is True
    assert should_select_uploaded_source(bytes_changed=False, dataset_count=2) is False
    assert should_select_uploaded_source(bytes_changed=True, dataset_count=0) is False


def test_later_page_opens_do_not_reparse_or_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    real = collect_datasets

    def wrapped(entries):
        calls["count"] += 1
        return real(entries)

    monkeypatch.setattr("dashboard.upload_context.collect_datasets", wrapped)
    content = b"Code,Note\nA,1\nB,2\n"
    state = {PAYLOAD_KEY: [StoredUpload("notes.csv", content)]}
    first = open_upload_session(state)
    assert first is not None
    assert first.bytes_changed is True
    assert calls["count"] == 1
    second = open_upload_session(state)
    assert second is not None
    assert second.bytes_changed is False
    assert calls["count"] == 1
    assert second.session is first.session
    state[PAYLOAD_KEY] = [StoredUpload("notes.csv", content, ("Other",))]
    third = open_upload_session(state)
    assert third is not None
    assert third.bytes_changed is False
    assert calls["count"] == 2
    assert should_select_uploaded_source(bytes_changed=third.bytes_changed, dataset_count=len(third.session.order)) is False


def test_original_and_working_counts_stay_independent() -> None:
    state = {PAYLOAD_KEY: [StoredUpload("notes.csv", b"Code\nA\n\nB\n")]}
    opened = open_upload_session(state)
    assert opened is not None
    dataset_id = opened.session.order[0]
    original, current = profile_batches(state, opened, None, None)
    assert upload_overview(original).missing_values == 1
    assert upload_overview(current).missing_values == 1
    updated = apply_operation(
        opened.session,
        OperationSpec(operation="fill_constant", dataset_id=dataset_id, columns=["Code"], fill_value="Z"),
        approved=True,
        now=datetime.now(timezone.utc),
    )
    state[SESSION_KEY] = updated
    again = open_upload_session(state)
    assert again is not None
    assert again.bytes_changed is False
    original_after, working_after = profile_batches(state, again, None, None)
    assert upload_overview(original_after).missing_values == 1
    assert upload_overview(working_after).missing_values == 0
    working_rows = upload_findings(working_after, frames_for(updated, original=False))
    assert all(item.issue != "Missing values" for item in working_rows)
    original_rows = upload_findings(original_after, frames_for(updated, original=True))
    assert any(item.issue == "Missing values" and item.affected_count == 1 for item in original_rows)
    original_frame = updated.copies[dataset_id].original
    working_frame = updated.copies[dataset_id].working
    assert str(original_frame.iloc[1, 0]).strip() == ""
    assert working_frame.iloc[1, 0] == "Z"


def test_findings_keep_rules_observations_and_mismatches_apart() -> None:
    state = {
        PAYLOAD_KEY: [
            StoredUpload("people.csv", b"Employee Code\nE100\nE200\nE300\nE400\n"),
            StoredUpload("tasks.csv", b"Employee Code,Line\nE100,1\nE200,2\nE300,3\nE100,4\nMISSING,5\n"),
        ]
    }
    opened = open_upload_session(state)
    assert opened is not None
    _original, current = profile_batches(state, opened, None, None)
    rows = upload_findings(current, frames_for(opened.session, original=False))
    assert any(item.kind == "inferred_mismatch" and item.affected_count == 1 for item in rows)
    assert all(not hasattr(item, "severity") for item in rows)
    assert "score" not in UploadRowFields(rows)

    ruled = {PAYLOAD_KEY: [StoredUpload("notes.csv", b"Code\nA\nA\n\n")]}
    ruled_open = open_upload_session(ruled)
    assert ruled_open is not None
    _original_rules, with_rules = profile_batches(ruled, ruled_open, ["Code"], ["Code"])
    ruled_rows = upload_findings(with_rules, frames_for(ruled_open.session, original=False))
    assert any(item.kind == "user_rule" and item.issue == "Required column" for item in ruled_rows)
    assert any(item.kind == "user_rule" and item.issue == "Unique key" for item in ruled_rows)

    plain = {PAYLOAD_KEY: [StoredUpload("wide.csv", _mostly_unique_csv())]}
    plain_open = open_upload_session(plain)
    assert plain_open is not None
    _left, wide = profile_batches(plain, plain_open, None, None)
    wide_rows = upload_findings(wide, frames_for(plain_open.session, original=False))
    assert all(item.kind != "user_rule" for item in wide_rows)
    assert all(item.issue != "possible key" for item in wide_rows)


def test_original_view_does_not_drop_working_explanations() -> None:
    state = {PAYLOAD_KEY: [StoredUpload("notes.csv", b"Code\nA\n\nB\n")]}
    opened = open_upload_session(state)
    assert opened is not None
    dataset_id = opened.session.order[0]
    updated = apply_operation(
        opened.session,
        OperationSpec(operation="fill_constant", dataset_id=dataset_id, columns=["Code"], fill_value="Z"),
        approved=True,
        now=datetime.now(timezone.utc),
    )
    state[SESSION_KEY] = updated
    opened = open_upload_session(state)
    assert opened is not None
    original, working = profile_batches(state, opened, None, None)
    working_token = evidence_token(build_recommendations(working, frames_for(updated, original=False)))
    original_token = evidence_token(build_recommendations(original, frames_for(updated, original=True)))
    assert working_token != original_token
    state[AI_STORE_KEY] = {
        "fingerprint": opened.fingerprint,
        "generation": updated.generation,
        "token": working_token,
        "items": {"kept": "working-copy"},
    }
    hidden = explanation_store_for_view(
        state,
        fingerprint=opened.fingerprint,
        generation=updated.generation,
        token=original_token,
        mutate=False,
    )
    assert hidden["items"] == {}
    assert state[AI_STORE_KEY]["items"]["kept"] == "working-copy"
    shown = explanation_store_for_view(
        state,
        fingerprint=opened.fingerprint,
        generation=updated.generation,
        token=working_token,
        mutate=False,
    )
    assert shown["items"]["kept"] == "working-copy"


def test_empty_upload_message_is_not_the_banking_report() -> None:
    assert "DQ001" not in EMPTY_UPLOAD_MESSAGE
    assert "fictional commercial bank" not in EMPTY_UPLOAD_MESSAGE
    state = {SOURCE_MEMORY_KEY: UPLOADED_SOURCE_LABEL, PENDING_KEY: "preview", FINGERPRINT_KEY: "abc"}
    remember_source(state, BANKING_SOURCE_LABEL)
    assert state[FINGERPRINT_KEY] == "abc"
    assert state[PENDING_KEY] == "preview"
    assert FILE_IDENTITY_KEY not in state


def test_hidden_page_selection_returns_to_overview() -> None:
    assert PAGE_LABELS == ("Overview", "Analyze a file")
    state = {PAGE_WIDGET_KEY: "Findings"}
    fallback_hidden_page(state)
    assert state[PAGE_WIDGET_KEY] == "Overview"
    state[PAGE_WIDGET_KEY] = "Investigation"
    fallback_hidden_page(state)
    assert state[PAGE_WIDGET_KEY] == "Overview"
    state[PAGE_WIDGET_KEY] = "AI SQL Studio"
    fallback_hidden_page(state)
    assert state[PAGE_WIDGET_KEY] == "Overview"
    state[PAGE_WIDGET_KEY] = "Analyze a file"
    fallback_hidden_page(state)
    assert state[PAGE_WIDGET_KEY] == "Analyze a file"


def test_dashboard_source_switches_and_empty_state_stays_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    from streamlit.testing.v1 import AppTest

    import dashboard.recommendations_view as recommendations_view

    monkeypatch.setattr(recommendations_view, "explanation_available", lambda: False)
    app = AppTest.from_file(str(SRC / "dashboard" / "app.py"), default_timeout=60)
    app.run()
    source = _radio(app, "Active data source")
    assert source.value == "Banking demonstration"
    assert list(_radio(app, "Page").options) == ["Overview", "Analyze a file"]
    assert "fictional commercial bank" in _text(app)

    _radio(app, "Page").set_value("Analyze a file").run()
    app.file_uploader[0].upload("notes.csv", b"Code\nA\n\nB\n", "text/csv").run()
    assert _radio(app, "Active data source").value == "My uploaded files"
    fingerprint = app.session_state[FINGERPRINT_KEY]
    generation = app.session_state[SESSION_KEY].generation
    _radio(app, "Page").set_value("Overview").run()
    assert "fictional commercial bank" not in _text(app)
    assert "Active version: Current cleaned working copy" in _text(app)
    assert "No severity is assigned" in _text(app)
    labels = {item.label for item in app.metric}
    assert "Datasets analyzed" in labels
    assert "Missing values" in labels
    assert "Unmatched references" in labels
    _radio(app, "Uploaded data").set_value("Original uploaded data").run()
    assert "Active version: Original uploaded data" in _text(app)
    assert app.session_state[FINGERPRINT_KEY] == fingerprint
    assert app.session_state[SESSION_KEY].generation == generation

    _radio(app, "Page").set_value("Analyze a file").run()
    _radio(app, "Workflow").set_value("3. Insights").run()
    assert "Explain with AI" in _text(app) or any(
        item.label == "Explain this recommendation" for item in app.button
    )
    assert app.session_state[FINGERPRINT_KEY] == fingerprint
    assert app.session_state[SESSION_KEY].generation == generation

    _radio(app, "Active data source").set_value("Banking demonstration").run()
    assert app.session_state[FINGERPRINT_KEY] == fingerprint
    _radio(app, "Page").set_value("Overview").run()
    assert "fictional commercial bank" in _text(app)
    assert app.session_state[FINGERPRINT_KEY] == fingerprint

    _radio(app, "Page").set_value("Analyze a file").run()
    _radio(app, "Workflow").set_value("1. Upload").run()
    app.checkbox[0].set_value(True).run()
    next(item for item in app.button if item.label == "Remove uploaded files").click().run()
    _radio(app, "Active data source").set_value("My uploaded files").run()
    _radio(app, "Page").set_value("Overview").run()
    assert "No uploaded file is available" in _text(app)
    assert "DQ001" not in _text(app)
    assert "fictional commercial bank" not in _text(app)


def _radio(app, label: str):
    return next(item for item in app.radio if item.label == label)


def _text(app) -> str:
    chunks: list[str] = []
    for name in ("markdown", "caption", "info", "warning", "error", "text"):
        for item in getattr(app, name, []):
            chunks.append(str(getattr(item, "value", "")))
    return "\n".join(chunks)


def _mostly_unique_csv() -> bytes:
    lines = ["Code,Note"]
    lines.append("A,1")
    lines.append("A,2")
    for index in range(18):
        lines.append(f"K{index},{index}")
    return ("\n".join(lines) + "\n").encode()


def UploadRowFields(rows) -> str:
    return " ".join(item.evidence for item in rows)
