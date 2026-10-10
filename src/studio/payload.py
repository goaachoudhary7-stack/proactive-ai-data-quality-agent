"""The exact summary that a later Gemini request would send.

Consent is the hash of the canonical JSON. A changed question, description,
row count, column list, or data version produces a different hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from studio.catalog import StudioCatalog

QUESTION_LIMIT = 2000


def build_payload(catalog: StudioCatalog, question: str) -> dict[str, Any]:
    """Build the summary from safe names, types, row counts, and approved descriptions."""
    tables: list[dict[str, Any]] = []
    for dataset in catalog.datasets:
        columns: list[dict[str, Any]] = []
        for column in dataset.columns:
            item: dict[str, Any] = {
                "column_ref": column.column_ref,
                "type": column.inferred_type,
            }
            if column.description:
                item["description"] = column.description
            columns.append(item)
        table: dict[str, Any] = {
            "columns": columns,
            "row_count": dataset.row_count,
            "table_ref": dataset.table_ref,
        }
        if dataset.description:
            table["description"] = dataset.description
        tables.append(table)
    return {
        "data_version": catalog.version,
        "question": question.strip(),
        "tables": tables,
    }


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize the summary in a stable form."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_hash(payload: dict[str, Any]) -> str:
    """Hash the canonical summary."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_problems(payload: dict[str, Any]) -> tuple[str, ...]:
    """Reasons this summary cannot be approved yet."""
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return ("Enter a question before approving the summary.",)
    if len(question) > QUESTION_LIMIT:
        return (f"Shorten the question to {QUESTION_LIMIT} characters or fewer before approving the summary.",)
    return ()


def grant_consent(payload: dict[str, Any]) -> dict[str, str]:
    """Record approval of this exact summary. The summary itself is not stored."""
    problems = payload_problems(payload)
    if problems:
        raise ValueError(problems[0])
    return {"hash": payload_hash(payload)}


def consent_matches(stored: object, payload: dict[str, Any]) -> bool:
    """True when the stored approval still matches this summary."""
    if not isinstance(stored, dict):
        return False
    return stored.get("hash") == payload_hash(payload)
