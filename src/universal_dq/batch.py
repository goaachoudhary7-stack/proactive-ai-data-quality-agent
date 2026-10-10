"""Load several CSV files and Excel worksheets, then profile each one.

One unreadable file does not discard the files that loaded. Bytes stay in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from universal_dq.loader import (
    MAX_TOTAL_UPLOAD_BYTES,
    MAX_UPLOAD_BYTES,
    FileLoadError,
    inspect_upload,
    load_tabular_file,
)
from universal_dq.models import (
    PROCESSING_NOTE,
    AnalysisReport,
    BatchReport,
    DatasetResult,
)
from universal_dq.profile import AnalysisInputError, analyze_frame
from universal_dq.relationships import LIMITATIONS, infer_candidate_keys, infer_relationships


@dataclass
class LoadedDataset:
    """A dataset held for the current analysis, or the error that stopped it."""

    dataset_id: str
    source_name: str
    file_format: str | None
    sheet_name: str | None
    frame: pd.DataFrame | None
    error: str | None


def collect_datasets(
    files: list[tuple[str, bytes, tuple[str, ...] | None]],
    *,
    max_file_bytes: int = MAX_UPLOAD_BYTES,
    max_total_bytes: int = MAX_TOTAL_UPLOAD_BYTES,
) -> list[LoadedDataset]:
    """Load each selected CSV or worksheet. Duplicate filenames stay distinct."""
    datasets: list[LoadedDataset] = []
    name_counts: dict[str, int] = {}
    total_bytes = 0
    for filename, content, sheets in files:
        source_name = _display_filename(filename)
        name_counts[source_name] = name_counts.get(source_name, 0) + 1
        label = _file_label(source_name, name_counts[source_name])
        size = len(content)
        if size > max_file_bytes:
            limit_mb = max_file_bytes / (1024 * 1024)
            datasets.append(
                _failed(
                    label,
                    source_name,
                    f"The file is larger than the {limit_mb:g} MB per-file upload limit.",
                )
            )
            continue
        if total_bytes + size > max_total_bytes:
            limit_mb = max_total_bytes / (1024 * 1024)
            datasets.append(
                _failed(
                    label,
                    source_name,
                    "This file was skipped because the combined upload would exceed "
                    f"the {limit_mb:g} MB total limit.",
                )
            )
            continue
        total_bytes += size
        datasets.extend(_load_one(label, source_name, content, sheets, max_file_bytes))
    return datasets


def build_batch_report(
    datasets: list[LoadedDataset],
    *,
    required_columns: list[str] | None = None,
    key_columns: list[str] | None = None,
    max_file_bytes: int = MAX_UPLOAD_BYTES,
    max_total_bytes: int = MAX_TOTAL_UPLOAD_BYTES,
) -> BatchReport:
    """Profile every loaded dataset and suggest keys only where evidence is strong."""
    results: list[DatasetResult] = []
    comparable: list[tuple[str, pd.DataFrame, AnalysisReport]] = []
    for dataset in datasets:
        if dataset.error is not None or dataset.frame is None or dataset.file_format is None:
            results.append(
                DatasetResult(
                    dataset_id=dataset.dataset_id,
                    source_name=dataset.source_name,
                    file_format=dataset.file_format,  # type: ignore[arg-type]
                    sheet_name=dataset.sheet_name,
                    error=dataset.error or "This dataset could not be analyzed.",
                )
            )
            continue
        try:
            report = analyze_frame(
                dataset.frame,
                source_name=dataset.source_name,
                file_format=dataset.file_format,
                sheet_name=dataset.sheet_name,
                required_columns=required_columns,
                key_columns=key_columns,
            )
        except AnalysisInputError as exc:
            results.append(
                DatasetResult(
                    dataset_id=dataset.dataset_id,
                    source_name=dataset.source_name,
                    file_format=dataset.file_format,  # type: ignore[arg-type]
                    sheet_name=dataset.sheet_name,
                    error=str(exc),
                )
            )
            continue
        results.append(
            DatasetResult(
                dataset_id=dataset.dataset_id,
                source_name=dataset.source_name,
                file_format=dataset.file_format,  # type: ignore[arg-type]
                sheet_name=dataset.sheet_name,
                report=report,
            )
        )
        comparable.append((dataset.dataset_id, dataset.frame, report))

    keys = [
        key
        for dataset_id, frame, report in comparable
        for key in infer_candidate_keys(dataset_id, frame, report)
    ]
    return BatchReport(
        datasets=results,
        candidate_keys=keys,
        relationships=infer_relationships(comparable),
        limitations=list(LIMITATIONS),
        processing_note=PROCESSING_NOTE,
        file_byte_limit=max_file_bytes,
        total_byte_limit=max_total_bytes,
    )


def _load_one(
    label: str,
    source_name: str,
    content: bytes,
    sheets: tuple[str, ...] | None,
    max_file_bytes: int,
) -> list[LoadedDataset]:
    try:
        inspection = inspect_upload(content, source_name, max_bytes=max_file_bytes)
    except FileLoadError as exc:
        return [_failed(label, source_name, str(exc))]
    if inspection.file_format == "csv":
        return [_load_table(label, source_name, content, None, max_file_bytes)]
    selected = inspection.sheet_names[:1] if sheets is None else sheets
    if not selected:
        return [
            _failed(
                label,
                source_name,
                "Select at least one worksheet. Each selected worksheet is analyzed separately.",
                file_format="xlsx",
            )
        ]
    loaded: list[LoadedDataset] = []
    for sheet_name in selected:
        dataset_id = f"{label} · {sheet_name}"
        loaded.append(_load_table(dataset_id, source_name, content, sheet_name, max_file_bytes))
    return loaded


def _load_table(
    dataset_id: str,
    source_name: str,
    content: bytes,
    sheet_name: str | None,
    max_file_bytes: int,
) -> LoadedDataset:
    try:
        table = load_tabular_file(
            content,
            source_name,
            sheet_name=sheet_name,
            max_bytes=max_file_bytes,
        )
    except FileLoadError as exc:
        return _failed(
            dataset_id,
            source_name,
            str(exc),
            file_format="xlsx" if sheet_name is not None else "csv",
            sheet_name=sheet_name,
        )
    return LoadedDataset(
        dataset_id=dataset_id,
        source_name=table.source_name,
        file_format=table.file_format,
        sheet_name=table.sheet_name,
        frame=table.frame,
        error=None,
    )


def _failed(
    dataset_id: str,
    source_name: str,
    message: str,
    *,
    file_format: str | None = None,
    sheet_name: str | None = None,
) -> LoadedDataset:
    return LoadedDataset(dataset_id, source_name, file_format, sheet_name, None, message)


def _display_filename(filename: str) -> str:
    cleaned = filename.replace("\\", "/").split("/")[-1].strip()
    return cleaned or "upload"


def _file_label(filename: str, occurrence: int) -> str:
    if occurrence == 1:
        return filename
    return f"{filename} ({occurrence})"
