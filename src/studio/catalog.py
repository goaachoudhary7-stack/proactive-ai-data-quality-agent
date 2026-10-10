"""Safe SQL names for datasets already loaded in a cleaning session.

Local mappings keep the file, worksheet, and column header. Those names are
for the page. The Gemini summary uses dataset_N and column_N only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from universal_dq.cleaning import CleaningSession
from universal_dq.models import BatchReport

DESCRIPTION_LIMIT = 120
CatalogVersion = Literal["original", "working"]


@dataclass(frozen=True)
class ColumnMapping:
    """One column, with the local header and the safe SQL name."""

    column_ref: str
    column_name: str
    inferred_type: str
    description: str | None


@dataclass(frozen=True)
class DatasetMapping:
    """One uploaded dataset and the safe SQL table name."""

    table_ref: str
    dataset_id: str
    source_name: str
    sheet_name: str | None
    file_format: str
    row_count: int
    description: str | None
    columns: tuple[ColumnMapping, ...]


@dataclass(frozen=True)
class StudioCatalog:
    """The selected version of every successful dataset."""

    version: CatalogVersion
    datasets: tuple[DatasetMapping, ...]


def normalize_description(text: object) -> str | None:
    """Collapse a user-written description. Blank text is omitted."""
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.split())
    if not cleaned:
        return None
    return cleaned[:DESCRIPTION_LIMIT]


def types_from_batch(batch: BatchReport) -> dict[str, dict[str, str]]:
    """Copy inferred types already stored on a profile. Cell values are not read."""
    found: dict[str, dict[str, str]] = {}
    for item in batch.datasets:
        if item.report is None:
            continue
        found[item.dataset_id] = {column.name: column.inferred_type for column in item.report.columns}
    return found


def build_catalog(
    session: CleaningSession,
    *,
    version: CatalogVersion,
    descriptions: Mapping[str, str] | None = None,
    column_descriptions: Mapping[str, Mapping[str, str]] | None = None,
    types: Mapping[str, Mapping[str, str]] | None = None,
) -> StudioCatalog:
    """Name each successful dataset without copying or editing its rows."""
    dataset_text = descriptions or {}
    column_text = column_descriptions or {}
    type_map = types or {}
    datasets: list[DatasetMapping] = []
    for index, dataset_id in enumerate(session.order, start=1):
        item = session.copies[dataset_id]
        frame = item.original if version == "original" else item.working
        names = column_text.get(dataset_id, {})
        column_types = type_map.get(dataset_id, {})
        columns: list[ColumnMapping] = []
        for position, name in enumerate(frame.columns, start=1):
            header = str(name)
            columns.append(
                ColumnMapping(
                    column_ref=f"column_{position}",
                    column_name=header,
                    inferred_type=str(column_types.get(header, "unspecified")),
                    description=normalize_description(names.get(header)),
                )
            )
        datasets.append(
            DatasetMapping(
                table_ref=f"dataset_{index}",
                dataset_id=item.dataset_id,
                source_name=item.source_name,
                sheet_name=item.sheet_name,
                file_format=item.file_format,
                row_count=int(len(frame)),
                description=normalize_description(dataset_text.get(dataset_id)),
                columns=tuple(columns),
            )
        )
    return StudioCatalog(version=version, datasets=tuple(datasets))


def catalog_hash(catalog: StudioCatalog) -> str:
    """Hash the local schema, including headers and descriptions."""
    import hashlib
    import json

    body = {
        "version": catalog.version,
        "datasets": [
            {
                "table_ref": item.table_ref,
                "dataset_id": item.dataset_id,
                "source_name": item.source_name,
                "sheet_name": item.sheet_name,
                "file_format": item.file_format,
                "description": item.description,
                "columns": [
                    {
                        "column_ref": column.column_ref,
                        "column_name": column.column_name,
                        "inferred_type": column.inferred_type,
                        "description": column.description,
                    }
                    for column in item.columns
                ],
            }
            for item in catalog.datasets
        ],
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def example_questions(catalog: StudioCatalog) -> tuple[str, ...]:
    """Local question ideas. These are not sent unless the user selects one."""
    if not catalog.datasets:
        return ()
    first = catalog.datasets[0]
    first_name = first.description or first.table_ref
    questions = [f"How many rows are in {first_name}?"]
    if len(catalog.datasets) > 1:
        questions.append("Compare the number of records across datasets.")
    if first.columns:
        column_name = first.columns[0].description or first.columns[0].column_ref
        questions.append(f"Find rows in {first_name} where {column_name} is missing.")
        questions.append(f"Show duplicate values of {column_name} in {first_name}.")
    return tuple(questions)
