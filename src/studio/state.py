"""Session keys for the SQL studio.

Descriptions, consent, and a future query binding live with the upload.
Removing the uploaded files clears them. Cleaning history uses other keys.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

DESCRIPTIONS_KEY = "dq_studio_descriptions"
QUESTION_MEMORY_KEY = "dq_studio_question"
QUESTION_REQUEST_KEY = "dq_studio_question_request"
QUESTION_WIDGET_KEY = "dq-studio-question"
CONSENT_KEY = "dq_studio_consent"
QUERY_KEY = "dq_studio_query"
VERSION_IDS: tuple[str, ...] = ("original", "working")


def ensure_descriptions(state: MutableMapping[str, Any]) -> dict[str, Any]:
    """Return the description store, repairing a missing or partial value."""
    current = state.get(DESCRIPTIONS_KEY)
    if not isinstance(current, dict):
        current = {}
        state[DESCRIPTIONS_KEY] = current
    for version in VERSION_IDS:
        slot = current.get(version)
        if not isinstance(slot, dict):
            slot = {}
            current[version] = slot
        if not isinstance(slot.get("datasets"), dict):
            slot["datasets"] = {}
        columns = slot.get("columns")
        if not isinstance(columns, dict):
            columns = {}
            slot["columns"] = columns
        for key, value in list(columns.items()):
            if not isinstance(value, dict):
                columns.pop(key, None)
    return current


def dataset_description(state: MutableMapping[str, Any], version: str, dataset_id: str) -> str:
    """The saved dataset description, or an empty string."""
    store = ensure_descriptions(state)
    value = store.get(version, {}).get("datasets", {}).get(dataset_id, "")
    return value if isinstance(value, str) else ""


def column_description(
    state: MutableMapping[str, Any],
    version: str,
    dataset_id: str,
    column_name: str,
) -> str:
    """The saved column description, or an empty string."""
    store = ensure_descriptions(state)
    columns = store.get(version, {}).get("columns", {}).get(dataset_id, {})
    if not isinstance(columns, dict):
        return ""
    value = columns.get(column_name, "")
    return value if isinstance(value, str) else ""


def set_dataset_description(
    state: MutableMapping[str, Any],
    version: str,
    dataset_id: str,
    text: object,
) -> None:
    """Save one dataset description. A blank value removes it."""
    from studio.catalog import normalize_description

    store = ensure_descriptions(state)
    slot = store[version]["datasets"]
    cleaned = normalize_description(text)
    if cleaned is None:
        slot.pop(dataset_id, None)
    else:
        slot[dataset_id] = cleaned


def set_column_description(
    state: MutableMapping[str, Any],
    version: str,
    dataset_id: str,
    column_name: str,
    text: object,
) -> None:
    """Save one column description. A blank value removes it."""
    from studio.catalog import normalize_description

    store = ensure_descriptions(state)
    columns = store[version]["columns"]
    current = columns.get(dataset_id)
    if not isinstance(current, dict):
        current = {}
        columns[dataset_id] = current
    cleaned = normalize_description(text)
    if cleaned is None:
        current.pop(column_name, None)
        if not current:
            columns.pop(dataset_id, None)
    else:
        current[column_name] = cleaned


def description_map(state: MutableMapping[str, Any], version: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Copy the descriptions for one version."""
    store = ensure_descriptions(state)
    slot = store[version]
    datasets = {key: value for key, value in slot["datasets"].items() if isinstance(value, str)}
    columns: dict[str, dict[str, str]] = {}
    for dataset_id, names in slot["columns"].items():
        if isinstance(names, dict):
            columns[dataset_id] = {key: value for key, value in names.items() if isinstance(value, str)}
    return datasets, columns


def prepare_question(state: MutableMapping[str, Any]) -> None:
    """Apply an example-question request before the question box is drawn."""
    requested = state.pop(QUESTION_REQUEST_KEY, None)
    if isinstance(requested, str):
        state[QUESTION_WIDGET_KEY] = requested
        state[QUESTION_MEMORY_KEY] = requested
        return
    if QUESTION_WIDGET_KEY not in state and isinstance(state.get(QUESTION_MEMORY_KEY), str):
        state[QUESTION_WIDGET_KEY] = state[QUESTION_MEMORY_KEY]


def clear_studio_state(state: MutableMapping[str, Any]) -> None:
    """Drop studio descriptions, consent, and any stored query binding."""
    for key in list(state):
        if key.startswith(("dq_studio_", "dq-studio-", "studio-")):
            state.pop(key, None)
