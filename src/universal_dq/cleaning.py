"""Approved cleaning of in-memory working copies.

The original frame for each dataset is kept. Preview counts a copy and
discards it. Apply runs only when the caller sets ``approved``. Fill
constants stay on the in-memory operation so undo can replay them. They are
not copied into the audit entry.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd

from universal_dq.batch import LoadedDataset
from universal_dq.models import (
    AnalysisReport,
    AuditEntry,
    CandidateKey,
    CandidateRelationship,
    MetricSnapshot,
    OperationSpec,
    PreviewResult,
    RelationshipSnapshot,
)
from universal_dq.profile import analyze_frame, exact_duplicate_extra_mask, is_missing
from universal_dq.relationships import infer_candidate_keys, infer_relationships

ROW_OPERATIONS = frozenset({"drop_exact_duplicates", "drop_empty_rows"})
COLUMN_OPERATIONS = frozenset(
    {
        "trim_whitespace",
        "whitespace_to_missing",
        "fill_constant",
        "fill_median",
        "change_case",
        "convert_type",
        "rename_columns",
    }
)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_FORMAT_TOKEN = re.compile(r"%[A-Za-z]")
_KNOWN_DATE_TOKENS = {
    "%Y",
    "%y",
    "%m",
    "%d",
    "%b",
    "%B",
    "%H",
    "%M",
    "%S",
    "%f",
    "%z",
    "%p",
    "%I",
}
_YEAR_TOKENS = {"%Y", "%y"}
_MONTH_TOKENS = {"%m", "%b", "%B"}
_DAY_TOKENS = {"%d"}


class CleaningRejected(Exception):
    """The working copy was left unchanged."""


@dataclass
class DatasetCopy:
    """Original upload and the working copy for one dataset."""

    dataset_id: str
    source_name: str
    file_format: str
    sheet_name: str | None
    original: pd.DataFrame
    working: pd.DataFrame


@dataclass
class CleaningSession:
    """Frames and audit for the current upload selection."""

    copies: dict[str, DatasetCopy]
    order: list[str]
    applied: list[OperationSpec] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)
    generation: int = 0


@dataclass(frozen=True)
class PendingPreview:
    """A preview that can be applied only for this generation and spec."""

    spec: OperationSpec
    preview: PreviewResult
    generation: int


def content_fingerprint(
    files: list[tuple[str, bytes, tuple[str, ...] | None]],
) -> str:
    """Hash file bytes and selected worksheets, not just names and sizes."""
    digest = hashlib.sha256()
    for name, content, sheets in files:
        digest.update(b"file\0")
        digest.update(name.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
        digest.update(b"\0")
        if sheets is None:
            digest.update(b"*")
        else:
            digest.update(b"\x1f".join(sheet.encode("utf-8", errors="replace") for sheet in sheets))
        digest.update(b"\0")
    return digest.hexdigest()


def new_cleaning_session(datasets: list[LoadedDataset]) -> CleaningSession:
    """Copy each successful table. Failed files are omitted."""
    copies: dict[str, DatasetCopy] = {}
    order: list[str] = []
    for item in datasets:
        if item.error is not None or item.frame is None or item.file_format is None:
            continue
        original = item.frame.copy()
        order.append(item.dataset_id)
        copies[item.dataset_id] = DatasetCopy(
            dataset_id=item.dataset_id,
            source_name=item.source_name,
            file_format=item.file_format,
            sheet_name=item.sheet_name,
            original=original,
            working=original.copy(),
        )
    return CleaningSession(copies=copies, order=order)


def restore_or_create(
    previous: CleaningSession | None,
    previous_fingerprint: str | None,
    fingerprint: str,
    datasets: list[LoadedDataset],
) -> CleaningSession:
    """Reuse the working copies when the upload bytes and sheets are unchanged."""
    if previous is not None and previous_fingerprint == fingerprint:
        return previous
    return new_cleaning_session(datasets)


def same_request(pending: PendingPreview, spec: OperationSpec, generation: int) -> bool:
    """True when this preview still matches the form and the working copy."""
    return pending.generation == generation and pending.spec == spec


def preview_operation(session: CleaningSession, spec: OperationSpec) -> PreviewResult:
    """Count the effect on a copy. The session frames stay as they are."""
    normalized, reason = _normalize(session, spec)
    if reason is not None or normalized is None:
        return _blocked(spec, reason or "This operation cannot be previewed.")
    item = session.copies[normalized.dataset_id]
    before_frame = item.working
    outcome = _execute(before_frame, normalized)
    if (
        normalized.operation == "fill_median"
        and outcome.cells_changed == 0
        and outcome.columns_skipped
    ):
        return _blocked(normalized, " ".join(outcome.columns_skipped))
    if (
        normalized.operation != "rename_columns"
        and outcome.cells_changed == 0
        and outcome.rows_removed == 0
        and outcome.conversion_failures == 0
        and not outcome.columns_skipped
    ):
        return _blocked(normalized, "This operation would not change any rows or cells.")
    warnings = _key_warnings(session, normalized)
    return PreviewResult(
        applicable=True,
        dataset_id=normalized.dataset_id,
        operation=normalized.operation,
        columns=list(normalized.columns),
        rows_before=outcome.rows_before,
        rows_after=outcome.rows_after,
        rows_removed=outcome.rows_removed,
        cells_changed=outcome.cells_changed,
        conversion_failures=outcome.conversion_failures,
        columns_skipped=outcome.columns_skipped,
        warnings=warnings,
    )


def apply_operation(
    session: CleaningSession,
    spec: OperationSpec,
    *,
    approved: bool,
    now: datetime | None = None,
) -> CleaningSession:
    """Apply one approved operation to the target working copy."""
    if not approved:
        raise CleaningRejected("Approve the preview before applying this operation.")
    preview = preview_operation(session, spec)
    if not preview.applicable:
        raise CleaningRejected(preview.block_reason or "This operation cannot be applied.")
    normalized, reason = _normalize(session, spec)
    if reason is not None or normalized is None:
        raise CleaningRejected(reason or "This operation cannot be applied.")
    before = _bundle(session)
    updated = _with_working(session, normalized.dataset_id, _execute(session.copies[normalized.dataset_id].working, normalized).frame)
    after = _bundle(updated)
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(timespec="seconds")
    entry = AuditEntry(
        action_order=len(session.audit) + 1,
        timestamp=timestamp,
        dataset_id=normalized.dataset_id,
        source_name=session.copies[normalized.dataset_id].source_name,
        sheet_name=session.copies[normalized.dataset_id].sheet_name,
        operation=normalized.operation,
        columns=list(normalized.columns),
        parameter_summary=describe_parameters(normalized),
        rows_before=preview.rows_before,
        rows_after=preview.rows_after,
        rows_removed=preview.rows_removed,
        cells_changed=preview.cells_changed,
        conversion_failures=preview.conversion_failures,
        columns_skipped=list(preview.columns_skipped),
        key_warning=" ".join(preview.warnings),
        metrics_before=before.metrics[normalized.dataset_id],
        metrics_after=after.metrics[normalized.dataset_id],
        relationships_before=before.relationships,
        relationships_after=after.relationships,
    )
    return CleaningSession(
        copies=updated.copies,
        order=list(session.order),
        applied=[*session.applied, normalized],
        audit=[*session.audit, entry],
        generation=session.generation + 1,
    )


def undo(session: CleaningSession) -> CleaningSession:
    """Drop the latest approved operation and rebuild the working copies."""
    if not session.applied:
        return session
    return _replay(session, session.applied[:-1], session.audit[:-1])


def reset(session: CleaningSession) -> CleaningSession:
    """Restore every working copy from its original upload."""
    if not session.applied and not session.audit:
        return session
    copies = {
        dataset_id: DatasetCopy(
            dataset_id=item.dataset_id,
            source_name=item.source_name,
            file_format=item.file_format,
            sheet_name=item.sheet_name,
            original=item.original,
            working=item.original.copy(),
        )
        for dataset_id, item in session.copies.items()
    }
    return CleaningSession(
        copies=copies,
        order=list(session.order),
        generation=session.generation + 1,
    )


def describe_parameters(spec: OperationSpec) -> str:
    """Explain the operation without fill constants or cell values."""
    if spec.operation == "drop_exact_duplicates":
        return "Exact duplicate rows beyond the first occurrence were removed."
    if spec.operation == "trim_whitespace":
        return "Leading and trailing whitespace was removed from text in the selected columns."
    if spec.operation == "whitespace_to_missing":
        return "Whitespace-only text in the selected columns was stored as missing."
    if spec.operation == "drop_empty_rows":
        return "Rows with no values in any column were removed."
    if spec.operation == "fill_constant":
        return "Missing cells were filled with a user-entered constant. The constant is not stored in this audit."
    if spec.operation == "fill_median":
        return "Missing cells were filled with the median of the numeric values in that column. The median is not stored in this audit."
    if spec.operation == "change_case":
        return f"Text case was set to {spec.case}."
    if spec.operation == "convert_type" and spec.target_type == "date":
        return (
            f"Text was parsed as a date with the explicit format {spec.date_format}. "
            "Values that did not match that format were left unchanged."
        )
    if spec.operation == "convert_type":
        return (
            f"Values were converted to {spec.target_type}. "
            "Values that could not be converted were left unchanged."
        )
    pairs = ", ".join(f"{old} -> {new}" for old, new in spec.renames.items())
    return f"Columns were renamed: {pairs}."


def datasets_for_quality(
    session: CleaningSession,
    *,
    which: Literal["original", "working"] = "working",
    errors: list[LoadedDataset] | None = None,
) -> list[LoadedDataset]:
    """Rebuild loader records from the original upload or the working copy."""
    loaded: list[LoadedDataset] = []
    for dataset_id in session.order:
        item = session.copies[dataset_id]
        frame = item.original if which == "original" else item.working
        loaded.append(
            LoadedDataset(
                dataset_id=item.dataset_id,
                source_name=item.source_name,
                file_format=item.file_format,
                sheet_name=item.sheet_name,
                frame=frame,
                error=None,
            )
        )
    if errors:
        loaded.extend(errors)
    return loaded


@dataclass
class _Bundle:
    metrics: dict[str, MetricSnapshot]
    keys: list[CandidateKey]
    relationships: list[RelationshipSnapshot]


@dataclass
class _Outcome:
    frame: pd.DataFrame
    rows_before: int
    rows_after: int
    rows_removed: int
    cells_changed: int
    conversion_failures: int
    columns_skipped: list[str]


def _blocked(spec: OperationSpec, reason: str) -> PreviewResult:
    return PreviewResult(
        applicable=False,
        block_reason=reason,
        dataset_id=spec.dataset_id,
        operation=spec.operation,
        columns=list(spec.columns),
    )


def _normalize(
    session: CleaningSession,
    spec: OperationSpec,
) -> tuple[OperationSpec | None, str | None]:
    if spec.dataset_id not in session.copies:
        return None, "Choose a dataset that was loaded."
    frame = session.copies[spec.dataset_id].working
    columns = [str(name) for name in frame.columns]
    if spec.operation in ROW_OPERATIONS:
        return spec.model_copy(update={"columns": [], "renames": {}}), None
    if spec.operation == "rename_columns":
        return _normalize_rename(spec, columns)
    selected = _unique(spec.columns)
    missing = [name for name in selected if name not in columns]
    if missing:
        names = ", ".join(missing)
        return None, f"These columns are not in the working copy: {names}."
    if not selected:
        return None, "Select at least one column."
    if spec.operation == "fill_constant":
        if spec.fill_value is None or spec.fill_value == "":
            return None, "Enter a constant before filling missing cells."
    elif spec.operation == "change_case":
        if spec.case not in {"upper", "lower", "title"}:
            return None, "Choose uppercase, lowercase, or title case."
    elif spec.operation == "convert_type":
        if spec.target_type not in {"numeric", "date", "text"}:
            return None, "Choose numeric, date, or text."
        if spec.target_type == "date":
            date_reason = _date_format_reason(spec.date_format)
            if date_reason is not None:
                return None, date_reason
    elif spec.operation not in COLUMN_OPERATIONS:
        return None, "Choose a cleaning operation."
    return spec.model_copy(update={"columns": selected}), None


def _normalize_rename(
    spec: OperationSpec,
    columns: list[str],
) -> tuple[OperationSpec | None, str | None]:
    if not spec.renames:
        return None, "Enter a new name for at least one column."
    unknown = [name for name in spec.renames if name not in columns]
    if unknown:
        names = ", ".join(unknown)
        return None, f"These columns are not in the working copy: {names}."
    resulting: list[str] = []
    applied: dict[str, str] = {}
    for name in columns:
        raw = spec.renames.get(name, name)
        renamed = raw.strip() if isinstance(raw, str) else str(raw).strip()
        if renamed == "":
            return None, "Enter a name for every selected column."
        resulting.append(renamed)
        if renamed != name:
            applied[name] = renamed
    duplicates = sorted({name for name in resulting if resulting.count(name) > 1})
    if duplicates:
        names = ", ".join(duplicates)
        return None, f"Column names must be unique. These names would be repeated: {names}."
    if not applied:
        return None, "Enter a column name that is different from the current name."
    return spec.model_copy(update={"columns": list(applied), "renames": applied}), None


def _date_format_reason(date_format: str | None) -> str | None:
    if date_format is None or date_format.strip() == "":
        return "Enter a date format with a year, a month, and a day. Ambiguous dates are not guessed."
    tokens = set(_FORMAT_TOKEN.findall(date_format))
    unknown = sorted(token for token in tokens if token not in _KNOWN_DATE_TOKENS)
    if unknown:
        names = ", ".join(unknown)
        return f"This date format contains unsupported tokens: {names}."
    if not (tokens & _YEAR_TOKENS and tokens & _MONTH_TOKENS and tokens & _DAY_TOKENS):
        return "Enter a date format with a year, a month, and a day. Ambiguous dates are not guessed."
    return None


def _execute(frame: pd.DataFrame, spec: OperationSpec) -> _Outcome:
    rows_before = int(len(frame))
    if spec.operation == "drop_exact_duplicates":
        mask = exact_duplicate_extra_mask(frame)
        removed = int(mask.sum())
        result = frame.loc[~mask].reset_index(drop=True)
        return _outcome(result, rows_before, removed, 0, 0, [])
    if spec.operation == "drop_empty_rows":
        empty = frame.apply(lambda row: all(is_missing(value) for value in row), axis=1)
        removed = int(empty.sum())
        result = frame.loc[~empty].reset_index(drop=True)
        return _outcome(result, rows_before, removed, 0, 0, [])
    if spec.operation == "rename_columns":
        mapping = {name: spec.renames.get(name, name) for name in frame.columns}
        result = frame.copy()
        result.columns = [mapping[str(name)] for name in frame.columns]
        return _outcome(result, rows_before, 0, 0, 0, [])
    if spec.operation == "fill_median":
        return _fill_median(frame, spec.columns, rows_before)
    result = frame.copy()
    cells_changed = 0
    failures = 0
    for column in spec.columns:
        values: list[object] = []
        for current in result[column].tolist():
            updated, changed, failed = _convert_cell(current, spec)
            values.append(updated)
            cells_changed += int(changed)
            failures += int(failed)
        result[column] = pd.Series(values, index=result.index, dtype=object)
    return _outcome(result, rows_before, 0, cells_changed, failures, [])


def _fill_median(frame: pd.DataFrame, columns: list[str], rows_before: int) -> _Outcome:
    result = frame.copy()
    cells_changed = 0
    skipped: list[str] = []
    for column in columns:
        values = result[column].tolist()
        present = [value for value in values if not is_missing(value)]
        if not present or any(not _is_real_number(value) for value in present):
            skipped.append(f"{column}: median fill needs every non-missing value to already be numeric.")
            continue
        numbers_only = [float(value) for value in present]
        median = statistics.median(numbers_only)
        if all(_is_strict_int(value) for value in present) and float(median).is_integer():
            stored: object = int(median)
        else:
            stored = float(median)
        filled: list[object] = []
        for value in values:
            if is_missing(value):
                filled.append(stored)
                cells_changed += 1
            else:
                filled.append(value)
        result[column] = pd.Series(filled, index=result.index, dtype=object)
    return _outcome(result, rows_before, 0, cells_changed, 0, skipped)


def _convert_cell(value: object, spec: OperationSpec) -> tuple[object, bool, bool]:
    if spec.operation == "trim_whitespace":
        if isinstance(value, str):
            updated = value.strip()
            return updated, updated != value, False
        return value, False, False
    if spec.operation == "whitespace_to_missing":
        if isinstance(value, str) and value.strip() == "":
            return None, True, False
        return value, False, False
    if spec.operation == "fill_constant":
        if is_missing(value):
            return spec.fill_value, True, False
        return value, False, False
    if spec.operation == "change_case":
        if isinstance(value, str) and not is_missing(value):
            updated = _apply_case(value, spec.case or "upper")
            return updated, updated != value, False
        return value, False, False
    if spec.operation == "convert_type" and spec.target_type == "numeric":
        return _convert_numeric(value)
    if spec.operation == "convert_type" and spec.target_type == "date":
        return _convert_date(value, spec.date_format or "")
    if spec.operation == "convert_type" and spec.target_type == "text":
        return _convert_text(value)
    return value, False, False


def _convert_numeric(value: object) -> tuple[object, bool, bool]:
    if is_missing(value):
        return value, False, False
    if isinstance(value, (bool, np.bool_)):
        return value, False, True
    if _is_strict_int(value):
        return value, False, False
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return value, False, True
        return value, False, False
    if isinstance(value, str):
        text = value.strip()
        if _NUMBER.fullmatch(text) is None:
            return value, False, True
        parsed: object = float(text) if ("." in text or "e" in text.lower()) else int(text)
        if isinstance(parsed, float) and not math.isfinite(parsed):
            return value, False, True
        return parsed, True, False
    return value, False, True


def _convert_date(value: object, date_format: str) -> tuple[object, bool, bool]:
    if is_missing(value):
        return value, False, False
    if isinstance(value, datetime):
        return value, False, False
    if isinstance(value, date):
        return value, False, False
    if isinstance(value, str):
        try:
            return datetime.strptime(value, date_format), True, False
        except ValueError:
            return value, False, True
    return value, False, True


def _convert_text(value: object) -> tuple[object, bool, bool]:
    if is_missing(value) or isinstance(value, str):
        return value, False, False
    if isinstance(value, (bool, np.bool_)):
        return ("True" if bool(value) else "False"), True, False
    if _is_strict_int(value):
        return str(int(value)), True, False
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return value, False, True
        if number.is_integer():
            return str(int(number)), True, False
        return str(number), True, False
    if isinstance(value, datetime):
        return value.isoformat(sep=" "), True, False
    if isinstance(value, date):
        return value.isoformat(), True, False
    return str(value), True, False


def _apply_case(value: str, case: str) -> str:
    if case == "lower":
        return value.lower()
    if case == "title":
        return value.title()
    return value.upper()


def _outcome(
    frame: pd.DataFrame,
    rows_before: int,
    rows_removed: int,
    cells_changed: int,
    conversion_failures: int,
    columns_skipped: list[str],
) -> _Outcome:
    return _Outcome(
        frame=frame,
        rows_before=rows_before,
        rows_after=int(len(frame)),
        rows_removed=rows_removed,
        cells_changed=cells_changed,
        conversion_failures=conversion_failures,
        columns_skipped=columns_skipped,
    )


def _bundle(session: CleaningSession) -> _Bundle:
    metrics: dict[str, MetricSnapshot] = {}
    comparable: list[tuple[str, pd.DataFrame, AnalysisReport]] = []
    for dataset_id in session.order:
        item = session.copies[dataset_id]
        report = analyze_frame(
            item.working,
            source_name=item.source_name,
            file_format=item.file_format,
            sheet_name=item.sheet_name,
        )
        metrics[dataset_id] = _metrics(report)
        comparable.append((dataset_id, item.working, report))
    keys = [
        key
        for dataset_id, frame, report in comparable
        for key in infer_candidate_keys(dataset_id, frame, report)
    ]
    relationships = [_relationship_snapshot(item) for item in infer_relationships(comparable)]
    return _Bundle(metrics=metrics, keys=keys, relationships=relationships)


def _metrics(report: AnalysisReport) -> MetricSnapshot:
    missing = {column.name: column.missing.count for column in report.columns}
    return MetricSnapshot(
        data_rows=report.data_rows,
        rows_in_duplicate_groups=report.exact_duplicates.rows_in_duplicate_groups.count,
        extra_duplicate_copies=report.exact_duplicates.extra_copies.count,
        missing_cells=sum(missing.values()),
        missing_by_column=missing,
    )


def _relationship_snapshot(item: CandidateRelationship) -> RelationshipSnapshot:
    return RelationshipSnapshot(
        parent_dataset=item.parent_dataset,
        parent_column=item.parent_column,
        child_dataset=item.child_dataset,
        child_column=item.child_column,
        cardinality=item.cardinality,
        orphan_count=item.orphan_count,
        matched_count=item.matched_count,
        overlap_ratio=item.overlap_ratio,
    )


def _key_warnings(session: CleaningSession, spec: OperationSpec) -> list[str]:
    bundle = _bundle(session)
    item_keys = [key.column for key in bundle.keys if key.dataset_id == spec.dataset_id]
    parents = [item.parent_column for item in bundle.relationships if item.parent_dataset == spec.dataset_id]
    children = [item.child_column for item in bundle.relationships if item.child_dataset == spec.dataset_id]
    if spec.operation in ROW_OPERATIONS:
        watched = list(dict.fromkeys([*item_keys, *parents, *children]))
    else:
        watched = [column for column in spec.columns if column in set(item_keys) | set(parents) | set(children)]
    messages: list[str] = []
    for column in watched:
        roles: list[str] = []
        if column in item_keys:
            roles.append("candidate primary key")
        if column in parents:
            roles.append("possible parent key")
        if column in children:
            roles.append("possible child key")
        if not roles:
            continue
        role_text = " and ".join(roles)
        messages.append(
            f"{column} is a {role_text}. "
            "Changing it can change suggested keys, overlap, and orphan counts. "
            "Orphan references will not be deleted or repaired."
        )
    return messages


def _with_working(session: CleaningSession, dataset_id: str, frame: pd.DataFrame) -> CleaningSession:
    copies: dict[str, DatasetCopy] = {}
    for key, item in session.copies.items():
        if key == dataset_id:
            copies[key] = DatasetCopy(
                dataset_id=item.dataset_id,
                source_name=item.source_name,
                file_format=item.file_format,
                sheet_name=item.sheet_name,
                original=item.original,
                working=frame,
            )
        else:
            copies[key] = item
    return CleaningSession(
        copies=copies,
        order=list(session.order),
        applied=list(session.applied),
        audit=list(session.audit),
        generation=session.generation,
    )


def _replay(
    session: CleaningSession,
    specs: list[OperationSpec],
    audit: list[AuditEntry],
) -> CleaningSession:
    copies = {
        dataset_id: DatasetCopy(
            dataset_id=item.dataset_id,
            source_name=item.source_name,
            file_format=item.file_format,
            sheet_name=item.sheet_name,
            original=item.original,
            working=item.original.copy(),
        )
        for dataset_id, item in session.copies.items()
    }
    rebuilt = CleaningSession(copies=copies, order=list(session.order))
    for spec in specs:
        outcome = _execute(rebuilt.copies[spec.dataset_id].working, spec)
        rebuilt = _with_working(rebuilt, spec.dataset_id, outcome.frame)
    return CleaningSession(
        copies=rebuilt.copies,
        order=list(session.order),
        applied=list(specs),
        audit=list(audit),
        generation=session.generation + 1,
    )


def _unique(names: list[str]) -> list[str]:
    chosen: list[str] = []
    for name in names:
        if name not in chosen:
            chosen.append(name)
    return chosen


def _is_real_number(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _is_strict_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
