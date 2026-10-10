"""Open the upload session and reuse profiles for the same fingerprint.

Parsed tables live in the cleaning session for this browser session. A later
page does not read the file bytes again while the fingerprint is unchanged.
Batch profiles are cached the same way. Nothing here is stored on disk or in
a process-wide cache.
"""

from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from dashboard.cleaning_view import ERRORS_KEY, SESSION_KEY
from dashboard.session_store import (
    BATCH_CACHE_KEY,
    FILE_IDENTITY_KEY,
    KEY_COLUMNS_KEY,
    PAYLOAD_KEY,
    REQUIRED_KEY,
    StoredUpload,
    ensure_cleaning_state,
    entries_for,
    resolve_default_sheets,
    should_replace_session,
)
from universal_dq.batch import LoadedDataset, build_batch_report, collect_datasets
from universal_dq.cleaning import CleaningSession, content_fingerprint, datasets_for_quality
from universal_dq.models import BatchReport


@dataclass
class OpenedUpload:
    """The in-memory upload, without a new parse when the fingerprint matches."""

    fingerprint: str
    session: CleaningSession
    errors: list[LoadedDataset]
    bytes_changed: bool


@dataclass
class BatchCache:
    """Profiles for one fingerprint, generation, and rule selection."""

    fingerprint: str
    generation: int
    required: tuple[str, ...]
    keys: tuple[str, ...]
    working_required: tuple[str, ...]
    working_keys: tuple[str, ...]
    original: BatchReport
    current: BatchReport


def open_upload_session(state: MutableMapping[str, Any]) -> OpenedUpload | None:
    """Return the current upload. File bytes are parsed only when the fingerprint changes."""
    payload = [resolve_default_sheets(item) for item in list(state.get(PAYLOAD_KEY) or [])]
    if payload != list(state.get(PAYLOAD_KEY) or []):
        state[PAYLOAD_KEY] = payload
    if not payload:
        return None
    fingerprint = content_fingerprint(entries_for(payload))
    identity = _file_identity(payload)
    bytes_changed = state.get(FILE_IDENTITY_KEY) != identity
    state[FILE_IDENTITY_KEY] = identity
    if should_replace_session(state, fingerprint):
        loaded = collect_datasets(entries_for(payload))
        ensure_cleaning_state(state, payload, loaded)
        state.pop(BATCH_CACHE_KEY, None)
    session = state[SESSION_KEY]
    errors = list(state.get(ERRORS_KEY) or [])
    return OpenedUpload(fingerprint, session, errors, bytes_changed)


def stored_rule_columns(
    state: MutableMapping[str, Any],
    session: CleaningSession,
    errors: list[LoadedDataset],
) -> tuple[list[str] | None, list[str] | None]:
    """Read saved rule columns. They apply only to one successful dataset."""
    if len(session.order) != 1 or errors:
        return None, None
    names = [str(name) for name in session.copies[session.order[0]].original.columns]
    required = [name for name in list(state.get(REQUIRED_KEY) or []) if name in names]
    key_columns = [name for name in list(state.get(KEY_COLUMNS_KEY) or []) if name in names]
    return (required or None), (key_columns or None)


def profile_batches(
    state: MutableMapping[str, Any],
    opened: OpenedUpload,
    required: list[str] | None,
    key_columns: list[str] | None,
) -> tuple[BatchReport, BatchReport]:
    """Return the original and working profiles, rebuilding only what changed."""
    required_t = tuple(required or ())
    keys_t = tuple(key_columns or ())
    working_required = tuple(_still_present(opened.session, required) or ())
    working_keys = tuple(_still_present(opened.session, key_columns) or ())
    cache = state.get(BATCH_CACHE_KEY)
    if (
        isinstance(cache, BatchCache)
        and cache.fingerprint == opened.fingerprint
        and cache.generation == opened.session.generation
        and cache.required == required_t
        and cache.keys == keys_t
        and cache.working_required == working_required
        and cache.working_keys == working_keys
    ):
        return cache.original, cache.current
    original: BatchReport | None = None
    if (
        isinstance(cache, BatchCache)
        and cache.fingerprint == opened.fingerprint
        and cache.required == required_t
        and cache.keys == keys_t
    ):
        original = cache.original
    if original is None:
        original = build_batch_report(
            datasets_for_quality(opened.session, which="original", errors=opened.errors),
            required_columns=required,
            key_columns=key_columns,
        )
    current = build_batch_report(
        datasets_for_quality(opened.session, which="working", errors=opened.errors),
        required_columns=list(working_required) or None,
        key_columns=list(working_keys) or None,
    )
    state[BATCH_CACHE_KEY] = BatchCache(
        fingerprint=opened.fingerprint,
        generation=opened.session.generation,
        required=required_t,
        keys=keys_t,
        working_required=working_required,
        working_keys=working_keys,
        original=original,
        current=current,
    )
    return original, current


def frames_for(session: CleaningSession, *, original: bool) -> dict:
    """Column frames for recommendations. The frames are not copied or modified here."""
    return {
        dataset_id: (item.original if original else item.working)
        for dataset_id, item in session.copies.items()
    }


def _file_identity(payload: list[StoredUpload]) -> tuple[tuple[str, str], ...]:
    """Names and content hashes. Worksheet choices are not part of this identity."""
    return tuple((item.name, hashlib.sha256(item.content).hexdigest()) for item in payload)


def _still_present(session: CleaningSession, selected: list[str] | None) -> list[str] | None:
    if not selected:
        return None
    for dataset_id in session.order:
        columns = [str(name) for name in session.copies[dataset_id].working.columns]
        if any(name not in columns for name in selected):
            return None
    return selected
