"""Session keys for the upload workflow.

Changing the visible step does not change the upload fingerprint or replace
the cleaning session. Bytes live here so a later step can unmount the file
uploader without forgetting the files.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, replace
from typing import Any

from dashboard.cleaning_view import ERRORS_KEY, FINGERPRINT_KEY, PENDING_KEY, SESSION_KEY
from universal_dq.batch import LoadedDataset
from universal_dq.cleaning import content_fingerprint, new_cleaning_session
from universal_dq.loader import FileLoadError, inspect_upload
from studio.state import clear_studio_state
from universal_dq.recommendations import AI_STORE_KEY, PREFILL_KEY

PAYLOAD_KEY = "dq_upload_payload"
REQUIRED_KEY = "dq_required_columns"
KEY_COLUMNS_KEY = "dq_key_columns"
SELECTED_DATASET_KEY = "dq_selected_dataset"
STEP_MEMORY_KEY = "dq_workflow_step"
STEP_REQUEST_KEY = "dq_workflow_step_request"
UPLOADER_GENERATION_KEY = "dq_uploader_generation"
WORKFLOW_WIDGET_KEY = "dq-workflow-step"
FILE_IDENTITY_KEY = "dq_upload_file_identity"
BATCH_CACHE_KEY = "dq_batch_cache"
SOURCE_WIDGET_KEY = "dq-active-source"
SOURCE_MEMORY_KEY = "dq_active_source"
SOURCE_REQUEST_KEY = "dq_source_request"
PAGE_WIDGET_KEY = "dq-page"
PAGE_REQUEST_KEY = "dq_page_request"
VERSION_WIDGET_KEY = "dq-upload-version"
VERSION_MEMORY_KEY = "dq_upload_version"

BANKING_SOURCE_LABEL = "Banking demonstration"
UPLOADED_SOURCE_LABEL = "My uploaded files"
SOURCE_LABELS: tuple[str, ...] = (BANKING_SOURCE_LABEL, UPLOADED_SOURCE_LABEL)
ORIGINAL_VERSION_LABEL = "Original uploaded data"
WORKING_VERSION_LABEL = "Current cleaned working copy"
VERSION_LABELS: tuple[str, ...] = (ORIGINAL_VERSION_LABEL, WORKING_VERSION_LABEL)
PAGE_LABELS: tuple[str, ...] = ("Overview", "Analyze a file")

WORKFLOW_STEPS: tuple[tuple[str, str], ...] = (
    ("upload", "1. Upload"),
    ("analyze", "2. Analyze"),
    ("insights", "3. Insights"),
    ("clean", "4. Clean"),
    ("export", "5. Export"),
)
STEP_LABELS: tuple[str, ...] = tuple(label for _step_id, label in WORKFLOW_STEPS)
CLEAN_STEP_LABEL = "4. Clean"


@dataclass(frozen=True)
class StoredUpload:
    """One uploaded file kept in the session, plus the worksheets to read."""

    name: str
    content: bytes
    sheets: tuple[str, ...] | None = None


def step_id(label: str) -> str:
    """Map a workflow label to its stable id."""
    for step, text in WORKFLOW_STEPS:
        if text == label:
            return step
    return "upload"


def apply_step_request(state: MutableMapping[str, Any]) -> None:
    """Honor a request to change steps before the workflow control is drawn."""
    requested = state.pop(STEP_REQUEST_KEY, None)
    if requested in STEP_LABELS:
        state[WORKFLOW_WIDGET_KEY] = requested


def restore_step_widget(state: MutableMapping[str, Any]) -> None:
    """Put the remembered step back when the control was not on the last page."""
    if WORKFLOW_WIDGET_KEY not in state and state.get(STEP_MEMORY_KEY) in STEP_LABELS:
        state[WORKFLOW_WIDGET_KEY] = state[STEP_MEMORY_KEY]


def remember_step(state: MutableMapping[str, Any], label: str) -> None:
    """Record the current step without touching the upload fingerprint."""
    state[STEP_MEMORY_KEY] = label


def apply_source_request(state: MutableMapping[str, Any]) -> None:
    """Apply a source change requested by a successful upload."""
    requested = state.pop(SOURCE_REQUEST_KEY, None)
    if requested in SOURCE_LABELS:
        state[SOURCE_WIDGET_KEY] = requested


def restore_source_widget(state: MutableMapping[str, Any]) -> None:
    """Restore the source control. The first visit uses the banking demonstration."""
    if SOURCE_WIDGET_KEY not in state:
        remembered = state.get(SOURCE_MEMORY_KEY, BANKING_SOURCE_LABEL)
        if remembered not in SOURCE_LABELS:
            remembered = BANKING_SOURCE_LABEL
        state[SOURCE_WIDGET_KEY] = remembered


def remember_source(state: MutableMapping[str, Any], label: str) -> None:
    """Record the source without changing the upload fingerprint."""
    state[SOURCE_MEMORY_KEY] = label


def request_uploaded_source(state: MutableMapping[str, Any]) -> None:
    """Ask the next run to show uploaded files. Cleaning does not call this."""
    state[SOURCE_REQUEST_KEY] = UPLOADED_SOURCE_LABEL


def should_select_uploaded_source(*, bytes_changed: bool, dataset_count: int) -> bool:
    """True only after a new file with at least one analyzed dataset."""
    return bytes_changed and dataset_count > 0


def apply_page_request(state: MutableMapping[str, Any]) -> None:
    """Open a page requested by a recommendation before the page control is drawn."""
    requested = state.pop(PAGE_REQUEST_KEY, None)
    if requested in PAGE_LABELS:
        state[PAGE_WIDGET_KEY] = requested


def fallback_hidden_page(state: MutableMapping[str, Any]) -> None:
    """Return a page that is no longer in the sidebar to Overview."""
    if state.get(PAGE_WIDGET_KEY) not in PAGE_LABELS:
        state[PAGE_WIDGET_KEY] = "Overview"


def restore_version_widget(state: MutableMapping[str, Any]) -> None:
    """Restore the original-or-working control."""
    if VERSION_WIDGET_KEY not in state:
        remembered = state.get(VERSION_MEMORY_KEY, WORKING_VERSION_LABEL)
        if remembered not in VERSION_LABELS:
            remembered = WORKING_VERSION_LABEL
        state[VERSION_WIDGET_KEY] = remembered


def remember_version(state: MutableMapping[str, Any], label: str) -> None:
    """Record which uploaded version is on screen."""
    state[VERSION_MEMORY_KEY] = label


def snapshot_uploads(
    existing: list[StoredUpload],
    incoming: list[tuple[str, bytes]],
) -> list[StoredUpload]:
    """Keep worksheet selections when the same bytes stay in the same order."""
    if len(existing) == len(incoming) and all(
        old.name == name and old.content == content for old, (name, content) in zip(existing, incoming, strict=True)
    ):
        return list(existing)
    return [StoredUpload(name, content) for name, content in incoming]


def resolve_default_sheets(item: StoredUpload) -> StoredUpload:
    """Select the first worksheet once. An explicit empty selection stays empty."""
    if item.sheets is not None:
        return item
    try:
        inspection = inspect_upload(item.content, item.name)
    except FileLoadError:
        return item
    if inspection.file_format != "xlsx" or not inspection.sheet_names:
        return item
    return replace(item, sheets=(inspection.sheet_names[0],))


def entries_for(payload: list[StoredUpload]) -> list[tuple[str, bytes, tuple[str, ...] | None]]:
    """Build the fingerprint input from stored files."""
    return [(item.name, item.content, item.sheets) for item in payload]


def should_replace_session(state: MutableMapping[str, Any], fingerprint: str) -> bool:
    """True only when the bytes, worksheets, or session itself changed."""
    return state.get(FINGERPRINT_KEY) != fingerprint or SESSION_KEY not in state


def ensure_cleaning_state(
    state: MutableMapping[str, Any],
    payload: list[StoredUpload],
    loaded: list[LoadedDataset],
) -> str:
    """Reuse the working copy when the fingerprint is unchanged."""
    fingerprint = content_fingerprint(entries_for(payload))
    if not should_replace_session(state, fingerprint):
        return fingerprint
    state[FINGERPRINT_KEY] = fingerprint
    state[SESSION_KEY] = new_cleaning_session(loaded)
    state[ERRORS_KEY] = [
        item for item in loaded if item.error is not None or item.frame is None
    ]
    state[PENDING_KEY] = None
    state.pop(AI_STORE_KEY, None)
    state.pop(PREFILL_KEY, None)
    state.pop(REQUIRED_KEY, None)
    state.pop(KEY_COLUMNS_KEY, None)
    state.pop("dq-required-columns", None)
    state.pop("dq-key-columns", None)
    return fingerprint


def clear_uploaded_files(state: MutableMapping[str, Any]) -> None:
    """Drop the upload and its working copy. This is not a cleaning reset."""
    for key in (
        PAYLOAD_KEY,
        FINGERPRINT_KEY,
        SESSION_KEY,
        ERRORS_KEY,
        PENDING_KEY,
        AI_STORE_KEY,
        PREFILL_KEY,
        REQUIRED_KEY,
        KEY_COLUMNS_KEY,
        SELECTED_DATASET_KEY,
        FILE_IDENTITY_KEY,
        BATCH_CACHE_KEY,
        "dq-required-columns",
        "dq-key-columns",
        "dq-selected-dataset",
    ):
        state.pop(key, None)
    clear_studio_state(state)
    state[UPLOADER_GENERATION_KEY] = int(state.get(UPLOADER_GENERATION_KEY, 0)) + 1
