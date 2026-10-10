"""Bind a future SQL statement to one catalog, version, and cleaning generation.

This module stores that binding. It does not generate SQL and it does not run it.
A stored statement is eligible only while every binding field still matches.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from studio.catalog import StudioCatalog, catalog_hash
from studio.payload import payload_hash
from studio.state import QUERY_KEY


@dataclass(frozen=True)
class QueryBinding:
    """The catalog and summary a stored statement was sealed against."""

    fingerprint: str
    version: str
    generation: int
    catalog_hash: str
    payload_hash: str


def make_binding(
    catalog: StudioCatalog,
    *,
    fingerprint: str,
    generation: int,
    payload: dict[str, Any],
) -> QueryBinding:
    """Capture the current catalog, version, generation, and summary."""
    return QueryBinding(
        fingerprint=fingerprint,
        version=catalog.version,
        generation=generation,
        catalog_hash=catalog_hash(catalog),
        payload_hash=payload_hash(payload),
    )


def seal_query(state: MutableMapping[str, Any], binding: QueryBinding, sql: str) -> None:
    """Remember SQL text for this binding. The text is not executed."""
    text = sql.strip()
    if not text:
        raise ValueError("A query binding requires SQL text.")
    state[QUERY_KEY] = {
        "fingerprint": binding.fingerprint,
        "version": binding.version,
        "generation": binding.generation,
        "catalog_hash": binding.catalog_hash,
        "payload_hash": binding.payload_hash,
        "sql": text,
    }


def stored_query_matches(state: MutableMapping[str, Any], binding: QueryBinding) -> bool:
    """True when a stored statement still belongs to this catalog and summary."""
    stored = state.get(QUERY_KEY)
    if not isinstance(stored, dict):
        return False
    sql = stored.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return False
    return (
        stored.get("fingerprint") == binding.fingerprint
        and stored.get("version") == binding.version
        and stored.get("generation") == binding.generation
        and stored.get("catalog_hash") == binding.catalog_hash
        and stored.get("payload_hash") == binding.payload_hash
    )


def drop_stale_query(state: MutableMapping[str, Any], binding: QueryBinding) -> None:
    """Remove a stored statement that no longer matches this binding."""
    if QUERY_KEY in state and not stored_query_matches(state, binding):
        state.pop(QUERY_KEY, None)
