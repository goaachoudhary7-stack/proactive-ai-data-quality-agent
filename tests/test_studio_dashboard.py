"""AI SQL Studio stays available as a module and stays off the sidebar."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dashboard.session_store import PAYLOAD_KEY, StoredUpload  # noqa: E402
from studio.payload import consent_matches, payload_hash  # noqa: E402
from studio.state import CONSENT_KEY, QUESTION_MEMORY_KEY  # noqa: E402


def _studio_page() -> None:
    from dashboard.studio_page import render_studio

    render_studio()


def _open_studio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(ROOT)
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_studio_page, default_timeout=60)
    return app


def test_saved_question_returns_when_the_box_is_absent() -> None:
    from studio.state import QUESTION_WIDGET_KEY, prepare_question

    state = {QUESTION_MEMORY_KEY: "How many rows are in dataset_1?"}
    prepare_question(state)
    assert state[QUESTION_WIDGET_KEY] == "How many rows are in dataset_1?"


def test_studio_empty_state_stays_off_the_banking_results(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open_studio(monkeypatch)
    app.run()
    assert not app.exception
    info = " ".join(item.value for item in app.info)
    assert "No uploaded file is available" in info
    main = " ".join(item.value for item in list(app.markdown) + list(app.code) + list(app.caption))
    assert "DQ001" not in main
    assert "fictional commercial bank" not in main.casefold()


def test_studio_shows_the_mapping_and_binds_consent_to_the_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _open_studio(monkeypatch)
    app.session_state[PAYLOAD_KEY] = [
        StoredUpload("quarterly-sales.csv", b"Account Name,Amount\nSECRET-CELL-ZX81,10\n")
    ]
    app.run()
    assert not app.exception
    frame = app.dataframe[0].value
    assert "quarterly-sales.csv" in set(frame["File"])
    assert "Account Name" in set(frame["Column"])
    assert "dataset_1" in set(frame["SQL table"])
    assert "column_1" in set(frame["SQL column"])

    app.text_area[0].set_value("How many rows are in dataset_1?").run()
    summary = json.loads(app.code[0].value)
    blob = json.dumps(summary)
    assert summary["question"] == "How many rows are in dataset_1?"
    assert summary["data_version"] == "working"
    assert summary["tables"][0]["table_ref"] == "dataset_1"
    assert "quarterly-sales.csv" not in blob
    assert "Account Name" not in blob
    assert "SECRET-CELL-ZX81" not in blob
    assert consent_matches(app.session_state.get(CONSENT_KEY), summary) is False

    next(item for item in app.checkbox if item.label == "I approve this exact summary").set_value(True).run()
    approved = json.loads(app.code[0].value)
    assert consent_matches(app.session_state[CONSENT_KEY], approved)
    assert app.session_state[CONSENT_KEY]["hash"] == payload_hash(approved)
    assert app.session_state[QUESTION_MEMORY_KEY] == "How many rows are in dataset_1?"

    app.run()
    restored = json.loads(app.code[0].value)
    assert consent_matches(app.session_state[CONSENT_KEY], restored)
    assert app.session_state[QUESTION_MEMORY_KEY] == "How many rows are in dataset_1?"

    app.text_area[0].set_value("Show duplicate values of column_1 in dataset_1.").run()
    revised = json.loads(app.code[0].value)
    assert consent_matches(app.session_state.get(CONSENT_KEY), revised) is False
    assert CONSENT_KEY not in app.session_state
