"""Suggest keys and cross-dataset links from completeness, uniqueness, and overlap.

Suggestions are not confirmed business rules. Cell values are counted and then
discarded. They are not included in the result text.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd

from universal_dq.models import AnalysisReport, CandidateKey, CandidateRelationship
from universal_dq.profile import is_missing

_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_MIN_ROWS = 2
_MIN_MATCHED = 2
_MIN_OVERLAP = 0.8
_MIN_NAME_LENGTH = 3

LIMITATIONS = (
    "Candidate keys and relationships are suggestions, not confirmed business rules.",
    "A candidate key must be a single column that is complete and unique across at least 2 data rows.",
    "A relationship suggestion requires similar column names, a complete unique parent column, and at least 80% of the child column's non-missing values found in that parent.",
    "Columns whose names normalize to fewer than 3 characters are not matched.",
    "Orphan counts are numbers of non-missing child values that do not occur in the parent. The values themselves are not listed.",
    "Text that looks like a number is compared as a number, so the text 10 can match the number 10. Other text is compared exactly after removing surrounding whitespace.",
    "The first row of each file is the header. CSV files must be UTF-8. .xls and .xlsb files are not supported.",
    "Excel formula cells without a cached value, and cells outside the top-left of a merged range, are read as empty.",
    "Uploads are held in memory for the analysis and are not written into the project.",
    "Quality checks do not change the uploaded table. Approved cleaning changes a separate working copy. Records are not sent to an external model.",
)


def infer_candidate_keys(dataset_id: str, frame: pd.DataFrame, report: AnalysisReport) -> list[CandidateKey]:
    """Suggest columns that are complete and unique. Do not suggest weak columns."""
    if report.data_rows < _MIN_ROWS:
        return []
    suggestions: list[CandidateKey] = []
    for column in report.columns:
        if column.missing.count != 0:
            continue
        tokens = _column_tokens(frame[column.name])
        distinct = len(set(tokens))
        if len(tokens) != report.data_rows or distinct != report.data_rows:
            continue
        suggestions.append(
            CandidateKey(
                dataset_id=dataset_id,
                column=column.name,
                data_rows=report.data_rows,
                distinct_non_missing=distinct,
                missing_count=0,
                detail=(
                    "Suggestion, not a confirmed key. "
                    f"{column.name} has 0 missing values of {report.data_rows} data rows "
                    f"and {distinct} distinct non-missing values. "
                    "The denominator for both checks is the number of data rows."
                ),
            )
        )
    return suggestions


def infer_relationships(
    datasets: list[tuple[str, pd.DataFrame, AnalysisReport]],
) -> list[CandidateRelationship]:
    """Suggest links only when the name and the value overlap are both strong."""
    profiles = [
        (dataset_id, frame, infer_candidate_keys(dataset_id, frame, report))
        for dataset_id, frame, report in datasets
    ]
    parent_tokens = {
        (dataset_id, key.column): _token_set(frame[key.column])
        for dataset_id, frame, keys in profiles
        for key in keys
    }
    suggestions: list[CandidateRelationship] = []
    for parent_id, parent_column in parent_tokens:
        for child_id, child_frame, _keys in profiles:
            if child_id == parent_id:
                continue
            for child_column in child_frame.columns:
                if not names_are_similar(str(parent_column), str(child_column)):
                    continue
                suggestion = _relationship(
                    parent_id,
                    str(parent_column),
                    parent_tokens[(parent_id, parent_column)],
                    child_id,
                    str(child_column),
                    child_frame[child_column],
                )
                if suggestion is not None:
                    suggestions.append(suggestion)
    return _one_suggestion_per_pair(suggestions)


def names_are_similar(left: str, right: str) -> bool:
    """True when two headers are likely the same kind of identifier."""
    normalized_left = _normalize_name(left)
    normalized_right = _normalize_name(right)
    if len(normalized_left) < _MIN_NAME_LENGTH or len(normalized_right) < _MIN_NAME_LENGTH:
        return False
    if normalized_left == normalized_right:
        return True
    short, long = (
        (normalized_left, normalized_right)
        if len(normalized_left) <= len(normalized_right)
        else (normalized_right, normalized_left)
    )
    if len(short) >= 5 and short in long:
        return True
    return SequenceMatcher(None, normalized_left, normalized_right).ratio() >= 0.92


def _relationship(
    parent_id: str,
    parent_column: str,
    parent_values: set[object],
    child_id: str,
    child_column: str,
    child_series: pd.Series,
) -> CandidateRelationship | None:
    if len(parent_values) < _MIN_ROWS:
        return None
    child_tokens = _column_tokens(child_series)
    if len(child_tokens) < _MIN_ROWS:
        return None
    matched = sum(1 for token in child_tokens if token in parent_values)
    if matched < _MIN_MATCHED:
        return None
    overlap = matched / len(child_tokens)
    if overlap < _MIN_OVERLAP:
        return None
    orphan_count = len(child_tokens) - matched
    matched_counts = Counter(token for token in child_tokens if token in parent_values)
    cardinality = "one-to-many" if any(count > 1 for count in matched_counts.values()) else "one-to-one"
    return CandidateRelationship(
        parent_dataset=parent_id,
        parent_column=parent_column,
        child_dataset=child_id,
        child_column=child_column,
        cardinality=cardinality,
        child_non_missing=len(child_tokens),
        matched_count=matched,
        orphan_count=orphan_count,
        overlap_ratio=overlap,
        detail=(
            "Suggestion, not a confirmed business rule. "
            f"The column names are similar. {matched} of {len(child_tokens)} non-missing child values "
            "occur in the parent column. "
            f"{orphan_count} non-missing child values do not occur in the parent. "
            "The overlap denominator is the number of non-missing child values. "
            f"The parent column is complete and unique ({len(parent_values)} values). "
            f"The suggested cardinality is {cardinality}."
        ),
    )


def _one_suggestion_per_pair(
    suggestions: list[CandidateRelationship],
) -> list[CandidateRelationship]:
    """Keep one direction when both datasets have the same unique column."""
    ordered = sorted(
        suggestions,
        key=lambda item: (
            item.parent_dataset,
            item.parent_column,
            item.child_dataset,
            item.child_column,
        ),
    )
    kept: list[CandidateRelationship] = []
    seen: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for item in ordered:
        pair = tuple(
            sorted(
                (
                    (item.parent_dataset, item.parent_column),
                    (item.child_dataset, item.child_column),
                )
            )
        )
        if pair in seen:
            continue
        seen.add(pair)
        kept.append(item)
    return kept


def _column_tokens(series: pd.Series) -> list[object]:
    return [token for value in series.tolist() if (token := _token(value)) is not None]


def _token_set(series: pd.Series) -> set[object]:
    return set(_column_tokens(series))


def _token(value: object) -> object | None:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, str):
        text = value.strip()
        if _NUMBER.fullmatch(text):
            return ("number", float(text))
        return ("text", text)
    try:
        return ("number", float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ("other", type(value).__name__)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())
