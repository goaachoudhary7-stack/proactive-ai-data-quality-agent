"""Catalog, privacy payload, consent, and query binding for the SQL studio."""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dashboard.session_store import (  # noqa: E402
    PAYLOAD_KEY,
    STEP_MEMORY_KEY,
    StoredUpload,
    clear_uploaded_files,
)
from dashboard.upload_context import open_upload_session, profile_batches  # noqa: E402
from studio.binding import (  # noqa: E402
    drop_stale_query,
    make_binding,
    seal_query,
    stored_query_matches,
)
from studio.catalog import build_catalog, catalog_hash, example_questions, normalize_description  # noqa: E402
from studio.payload import (  # noqa: E402
    build_payload,
    consent_matches,
    grant_consent,
    payload_hash,
    payload_problems,
)
from studio.state import (  # noqa: E402
    CONSENT_KEY,
    DESCRIPTIONS_KEY,
    QUERY_KEY,
    set_column_description,
    set_dataset_description,
)
from universal_dq.cleaning import apply_operation  # noqa: E402
from universal_dq.models import OperationSpec  # noqa: E402

SECRET_CELL = "SECRET-CELL-ZX81"
FILE_NAME = "quarterly-sales.csv"
COLUMN_NAME = "Account Name"
SHEET_NAME = "Alpha Sheet"


def _strings(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.append(str(key))
            found.extend(_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_strings(item))
    return found


def _opened(content: bytes, name: str = FILE_NAME, sheets: tuple[str, ...] | None = None):
    state = {PAYLOAD_KEY: [StoredUpload(name, content, sheets)]}
    opened = open_upload_session(state)
    assert opened is not None
    return state, opened


def _xlsx() -> bytes:
    workbook = Workbook()
    first = workbook.active
    first.title = SHEET_NAME
    first.append([COLUMN_NAME, "Amount"])
    first.append([SECRET_CELL, 10])
    second = workbook.create_sheet("Beta")
    second.append(["Item"])
    second.append(["beta-cell"])
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_catalog_names_csv_files_and_excel_worksheets() -> None:
    _state, opened = _opened(b"Account Name,Amount\nSECRET-CELL-ZX81,10\n")
    csv_catalog = build_catalog(opened.session, version="working", types={"unused": {}})
    assert [item.table_ref for item in csv_catalog.datasets] == ["dataset_1"]
    assert [column.column_ref for column in csv_catalog.datasets[0].columns] == ["column_1", "column_2"]
    assert csv_catalog.datasets[0].source_name == FILE_NAME
    assert csv_catalog.datasets[0].columns[0].column_name == COLUMN_NAME
    assert csv_catalog.datasets[0].sheet_name is None

    excel_state = {PAYLOAD_KEY: [StoredUpload("book.xlsx", _xlsx(), (SHEET_NAME, "Beta"))]}
    excel = open_upload_session(excel_state)
    assert excel is not None
    catalog = build_catalog(excel.session, version="original")
    assert [item.table_ref for item in catalog.datasets] == ["dataset_1", "dataset_2"]
    assert [item.sheet_name for item in catalog.datasets] == [SHEET_NAME, "Beta"]
    assert catalog.datasets[0].columns[0].column_ref == "column_1"
    assert catalog.datasets[1].columns[0].column_name == "Item"
    assert catalog.datasets[0].row_count == 1
    assert excel.session.order.index(catalog.datasets[0].dataset_id) == 0


def test_catalog_reads_both_versions_without_changing_frames() -> None:
    _state, opened = _opened(b"Code,Note\nA,\nB,ok\n")
    dataset_id = opened.session.order[0]
    original = opened.session.copies[dataset_id].original
    original_id = id(original)
    marker = original.iloc[0, 1]
    updated = apply_operation(
        opened.session,
        OperationSpec(
            operation="fill_constant",
            dataset_id=dataset_id,
            columns=["Note"],
            fill_value="ZZ-FILL",
        ),
        approved=True,
        now=datetime.now(timezone.utc),
    )
    before = build_catalog(opened.session, version="original")
    after = build_catalog(updated, version="working")
    assert before.datasets[0].row_count == after.datasets[0].row_count
    assert id(opened.session.copies[dataset_id].original) == original_id
    assert opened.session.copies[dataset_id].original.iloc[0, 1] == marker
    assert updated.copies[dataset_id].working.iloc[0, 1] == "ZZ-FILL"
    assert updated.copies[dataset_id].working.iloc[1, 1] == "ok"
    assert "ZZ-FILL" not in _strings(build_payload(after, "How many rows are in dataset_1?"))
    assert updated.generation == opened.session.generation + 1


def test_payload_keeps_descriptions_and_omits_local_names_and_cells() -> None:
    state, opened = _opened(f"{COLUMN_NAME},Amount\n{SECRET_CELL},10\n".encode())
    dataset_id = opened.session.order[0]
    _original, current = profile_batches(state, opened, None, None)
    set_dataset_description(state, "working", dataset_id, "  product family  ")
    set_column_description(state, "working", dataset_id, COLUMN_NAME, "product label")
    set_column_description(state, "working", dataset_id, "Amount", "   ")
    datasets, columns = _maps(state)
    catalog = build_catalog(
        opened.session,
        version="working",
        descriptions=datasets,
        column_descriptions=columns,
        types=__types(current),
    )
    payload = build_payload(catalog, "  Which rows are missing a value?  ")
    blob = "\n".join(_strings(payload))
    assert payload["data_version"] == "working"
    assert payload["question"] == "Which rows are missing a value?"
    assert payload["tables"][0]["description"] == "product family"
    assert payload["tables"][0]["columns"][0]["description"] == "product label"
    assert "description" not in payload["tables"][0]["columns"][1]
    assert FILE_NAME not in blob
    assert COLUMN_NAME not in blob
    assert SECRET_CELL not in blob
    assert dataset_id not in blob
    assert "source_name" not in blob
    assert "sheet_name" not in blob
    assert catalog.datasets[0].columns[0].inferred_type == "text"
    assert catalog.datasets[0].columns[1].inferred_type == "number"


def test_excel_sheet_names_stay_out_of_the_payload() -> None:
    state = {PAYLOAD_KEY: [StoredUpload("book.xlsx", _xlsx(), (SHEET_NAME, "Beta"))]}
    opened = open_upload_session(state)
    assert opened is not None
    catalog = build_catalog(opened.session, version="working")
    blob = "\n".join(_strings(build_payload(catalog, "Compare the number of records across datasets.")))
    assert SHEET_NAME not in blob
    assert "Beta" not in blob
    assert "book.xlsx" not in blob
    assert SECRET_CELL not in blob
    assert "beta-cell" not in blob
    assert [item.sheet_name for item in catalog.datasets] == [SHEET_NAME, "Beta"]


def test_descriptions_are_capped_and_blank_text_is_omitted() -> None:
    assert normalize_description("  ") is None
    assert normalize_description("a   b") == "a b"
    assert normalize_description("x" * 150) == "x" * 120
    state: dict = {}
    set_dataset_description(state, "working", "notes.csv", "x" * 150)
    assert state[DESCRIPTIONS_KEY]["working"]["datasets"]["notes.csv"] == "x" * 120
    set_column_description(state, "working", "notes.csv", "Note", "   ")
    assert "notes.csv" not in state[DESCRIPTIONS_KEY]["working"]["columns"]


def test_consent_matches_only_the_canonical_payload() -> None:
    state, opened = _opened(b"Code\nA\nB\n")
    working = build_catalog(opened.session, version="working")
    original = build_catalog(opened.session, version="original")
    payload = build_payload(working, "How many rows are in dataset_1?")
    other_order = {"tables": payload["tables"], "question": payload["question"], "data_version": payload["data_version"]}
    assert payload_hash(payload) == payload_hash(other_order)
    stored = grant_consent(payload)
    assert consent_matches(stored, payload)
    assert consent_matches(stored, other_order)
    changed = build_payload(working, "How many rows are in dataset_1? ")
    assert consent_matches(stored, changed)
    revised = build_payload(working, "Show duplicate values of column_1 in dataset_1.")
    assert consent_matches(stored, revised) is False
    assert consent_matches(stored, build_payload(original, payload["question"])) is False
    assert payload_problems(build_payload(working, "  "))
    try:
        grant_consent(build_payload(working, " "))
    except ValueError as exc:
        assert "question" in str(exc)
    else:
        raise AssertionError("blank question was approved")


def test_consent_and_query_binding_follow_the_catalog_version_and_generation() -> None:
    state, opened = _opened(b"Code,Note\nA,one\nA,one\n")
    dataset_id = opened.session.order[0]
    current = build_catalog(opened.session, version="working")
    payload = build_payload(current, "Show duplicate values of column_1 in dataset_1.")
    state[CONSENT_KEY] = grant_consent(payload)
    binding = make_binding(current, fingerprint=opened.fingerprint, generation=opened.session.generation, payload=payload)
    seal_query(state, binding, "SELECT column_1 FROM dataset_1")
    assert stored_query_matches(state, binding) is True

    described = build_catalog(
        opened.session,
        version="working",
        descriptions={dataset_id: "accounts"},
    )
    described_payload = build_payload(described, payload["question"])
    assert consent_matches(state[CONSENT_KEY], described_payload) is False
    described_binding = make_binding(
        described,
        fingerprint=opened.fingerprint,
        generation=opened.session.generation,
        payload=described_payload,
    )
    assert stored_query_matches(state, described_binding) is False
    drop_stale_query(state, described_binding)
    assert QUERY_KEY not in state

    seal_query(state, binding, "SELECT column_1 FROM dataset_1")
    renamed = apply_operation(
        opened.session,
        OperationSpec(operation="rename_columns", dataset_id=dataset_id, renames={"Note": "Comment"}),
        approved=True,
        now=datetime.now(timezone.utc),
    )
    renamed_catalog = build_catalog(renamed, version="working")
    renamed_payload = build_payload(renamed_catalog, payload["question"])
    renamed_binding = make_binding(
        renamed_catalog,
        fingerprint=opened.fingerprint,
        generation=renamed.generation,
        payload=renamed_payload,
    )
    assert renamed_catalog.datasets[0].columns[1].column_name == "Comment"
    assert catalog_hash(renamed_catalog) != catalog_hash(current)
    assert stored_query_matches(state, renamed_binding) is False
    assert renamed.copies[dataset_id].original.columns.tolist() == ["Code", "Note"]

    deduped = apply_operation(
        opened.session,
        OperationSpec(operation="drop_exact_duplicates", dataset_id=dataset_id),
        approved=True,
        now=datetime.now(timezone.utc),
    )
    deduped_catalog = build_catalog(deduped, version="working")
    deduped_payload = build_payload(deduped_catalog, payload["question"])
    assert deduped_payload["tables"][0]["row_count"] == 1
    assert consent_matches(state[CONSENT_KEY], deduped_payload) is False
    deduped_binding = make_binding(
        deduped_catalog,
        fingerprint=opened.fingerprint,
        generation=deduped.generation,
        payload=deduped_payload,
    )
    assert stored_query_matches(state, deduped_binding) is False


def test_example_questions_use_descriptions_when_the_user_wrote_them() -> None:
    state, opened = _opened(b"Code\nA\n")
    dataset_id = opened.session.order[0]
    plain = build_catalog(opened.session, version="working")
    assert example_questions(plain)[0] == "How many rows are in dataset_1?"
    labeled = build_catalog(opened.session, version="working", descriptions={dataset_id: "accounts"})
    assert example_questions(labeled)[0] == "How many rows are in accounts?"
    assert all(FILE_NAME not in item for item in example_questions(labeled))


def test_clearing_uploads_drops_studio_consent_and_keeps_the_workflow_step() -> None:
    state = {
        STEP_MEMORY_KEY: "4. Clean",
        PAYLOAD_KEY: [StoredUpload("notes.csv", b"Code\nA\n")],
        DESCRIPTIONS_KEY: {"working": {"datasets": {"notes.csv": "accounts"}}},
        CONSENT_KEY: {"hash": "abc"},
        QUERY_KEY: {"sql": "SELECT 1"},
        "dq-studio-question": "How many rows?",
        "studio-example-0": True,
    }
    clear_uploaded_files(state)
    assert PAYLOAD_KEY not in state
    assert DESCRIPTIONS_KEY not in state
    assert CONSENT_KEY not in state
    assert QUERY_KEY not in state
    assert "dq-studio-question" not in state
    assert "studio-example-0" not in state
    assert state[STEP_MEMORY_KEY] == "4. Clean"


def test_studio_modules_do_not_call_a_model_or_duckdb() -> None:
    forbidden = ("duckdb", "crewai", "google.genai", "google.generativeai")
    for path in list((SRC / "studio").glob("*.py")) + [SRC / "dashboard" / "studio_page.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                for name in forbidden:
                    assert not module.startswith(name), path.name
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "GEMINI_API_KEY" not in node.value
                assert "injected_issues" not in node.value


def _maps(state: dict) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    slot = state[DESCRIPTIONS_KEY]["working"]
    return dict(slot["datasets"]), {key: dict(value) for key, value in slot["columns"].items()}


def __types(batch) -> dict[str, dict[str, str]]:
    from studio.catalog import types_from_batch

    return types_from_batch(batch)
