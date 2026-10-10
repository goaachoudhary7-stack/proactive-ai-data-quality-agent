"""Deterministic checks that use whatever columns the file actually has."""

from __future__ import annotations

import math
import numbers
import re
from datetime import date, datetime

import numpy as np
import pandas as pd

from universal_dq.models import (
    PROCESSING_NOTE,
    AnalysisReport,
    ColumnProfile,
    ExactDuplicates,
    KeyDuplicates,
    MixedTypeFinding,
    OutlierFinding,
    Share,
)

_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")
_SLASH_DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{4}")
_BOOLEAN = {"true", "false"}
_MIN_OUTLIER_VALUES = 8
_MISSING = ("missing",)


class AnalysisInputError(Exception):
    """The user selected a column that is not in the uploaded file."""


def analyze_frame(
    frame: pd.DataFrame,
    *,
    source_name: str,
    file_format: str,
    sheet_name: str | None,
    required_columns: list[str] | None = None,
    key_columns: list[str] | None = None,
) -> AnalysisReport:
    """Profile a table that is already in memory. The frame is not modified."""
    columns = [str(name) for name in frame.columns]
    required = _selected_columns(required_columns, columns, "Required columns")
    keys = _selected_columns(key_columns, columns, "Unique-key columns")
    data_rows = int(len(frame))
    profiles = [
        _column_profile(
            frame[name],
            name=name,
            data_rows=data_rows,
            required=name in required,
            skip_outliers=name in keys,
        )
        for name in columns
    ]
    return AnalysisReport(
        source_name=source_name,
        file_format=file_format,  # type: ignore[arg-type]
        sheet_name=sheet_name,
        data_rows=data_rows,
        column_count=len(columns),
        column_names=columns,
        columns=profiles,
        exact_duplicates=_exact_duplicates(frame),
        key_duplicates=_key_duplicates(frame, keys) if keys else None,
        required_columns=required,
        processing_note=PROCESSING_NOTE,
    )


def is_missing(value: object) -> bool:
    """True for empty cells and whitespace-only strings.

    Literal text such as ``NA`` and ``NULL`` is kept. It is not treated as empty.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, numbers.Real):
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return result if isinstance(result, bool) else False


def _selected_columns(selected: list[str] | None, columns: list[str], label: str) -> list[str]:
    if not selected:
        return []
    unique: list[str] = []
    for name in selected:
        if name not in unique:
            unique.append(name)
    unknown = [name for name in unique if name not in columns]
    if unknown:
        missing = ", ".join(unknown)
        raise AnalysisInputError(
            f"{label} must use names from the header. These names are not in the file: {missing}."
        )
    return unique


def _column_profile(
    series: pd.Series,
    *,
    name: str,
    data_rows: int,
    required: bool,
    skip_outliers: bool,
) -> ColumnProfile:
    categories: dict[str, int] = {}
    numeric_values: list[float] = []
    missing_count = 0
    for value in series.tolist():
        if is_missing(value):
            missing_count += 1
            continue
        kind = _category(value)
        categories[kind] = categories.get(kind, 0) + 1
        if kind == "number":
            numeric_values.append(_as_float(value))
    non_missing = data_rows - missing_count
    inferred = _inferred_type(categories, non_missing)
    if required and missing_count > 0:
        role = "user_rule_violation"
    elif required:
        role = "user_rule_clear"
    else:
        role = "observation"
    return ColumnProfile(
        name=name,
        inferred_type=inferred,
        missing=_share(missing_count, data_rows, "data rows"),
        missing_role=role,
        mixed_type=_mixed_type(categories, non_missing),
        outlier=_outlier(inferred, numeric_values, skip_outliers=skip_outliers),
    )


def _category(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "datetime"
    if isinstance(value, numbers.Real):
        return "number"
    if isinstance(value, str):
        return _string_category(value)
    return "other"


def _string_category(value: str) -> str:
    if value.casefold() in _BOOLEAN:
        return "boolean"
    if _NUMBER.fullmatch(value):
        return "number"
    if _ISO_DATE.fullmatch(value) or _SLASH_DATE.fullmatch(value):
        return "datetime"
    return "text"


def _as_float(value: object) -> float:
    return float(value)  # type: ignore[arg-type]


def _inferred_type(categories: dict[str, int], non_missing: int) -> str:
    if non_missing == 0:
        return "empty"
    kinds = [kind for kind, count in categories.items() if count > 0]
    if len(kinds) == 1:
        return kinds[0]
    return "mixed"


def _mixed_type(categories: dict[str, int], non_missing: int) -> MixedTypeFinding | None:
    present = {kind: count for kind, count in categories.items() if count > 0}
    if len(present) < 2:
        return None
    listed = ", ".join(f"{kind} ({count})" for kind, count in sorted(present.items()))
    return MixedTypeFinding(
        non_missing_count=non_missing,
        categories=present,
        detail=(
            "Informational finding. Possible mixed types among "
            f"{non_missing} non-missing values: {listed}. "
            "The denominator is the number of non-missing values in this column."
        ),
    )


def _outlier(
    inferred: str,
    values: list[float],
    *,
    skip_outliers: bool,
) -> OutlierFinding | None:
    if inferred != "number":
        return None
    count = len(values)
    if skip_outliers:
        return OutlierFinding(
            label="skipped",
            non_missing_count=count,
            detail=(
                "Outlier check skipped because this column is part of the selected unique key."
            ),
        )
    if count < _MIN_OUTLIER_VALUES:
        return OutlierFinding(
            label="skipped",
            non_missing_count=count,
            detail=(
                "Outlier check skipped because this numeric column has fewer than "
                f"{_MIN_OUTLIER_VALUES} non-missing values."
            ),
        )
    series = pd.Series(values, dtype="float64")
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0 or math.isnan(iqr):
        return OutlierFinding(
            label="skipped",
            non_missing_count=count,
            detail=(
                "Outlier check skipped because the middle half of the values has no spread, "
                "so the interquartile range is not informative."
            ),
        )
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    outliers = int(((series < lower) | (series > upper)).sum())
    return OutlierFinding(
        label="informational",
        non_missing_count=count,
        outlier_count=outliers,
        denominator=count,
        lower_fence=lower,
        upper_fence=upper,
        detail=(
            "Informational finding. Possible numerical outliers: "
            f"{outliers} of {count} non-missing numeric values fall outside the "
            f"interquartile-range fences ({lower:.6g} to {upper:.6g}). "
            "The denominator is the number of non-missing numeric values."
        ),
    )


def exact_duplicate_extra_mask(frame: pd.DataFrame) -> pd.Series:
    """True for exact duplicate rows after the first occurrence.

    The comparison is the same one used by the duplicate counts.
    """
    if len(frame) == 0 or len(frame.columns) == 0:
        return pd.Series(False, index=frame.index)
    comparison = _comparison_frame(frame)
    return comparison.duplicated(keep="first")


def _exact_duplicates(frame: pd.DataFrame) -> ExactDuplicates:
    data_rows = int(len(frame))
    in_groups, extra = _duplicate_counts(frame, list(frame.columns))
    return ExactDuplicates(
        rows_in_duplicate_groups=_share(in_groups, data_rows, "data rows"),
        extra_copies=_share(extra, data_rows, "data rows"),
        detail=(
            "Exact duplicate rows match on every column. "
            "Rows in duplicate groups counts every copy, including the first. "
            "Extra copies counts only the copies beyond the first occurrence. "
            "Both denominators are the number of data rows. "
            "This is a detected pattern, not a user-defined rule."
        ),
    )


def _key_duplicates(frame: pd.DataFrame, keys: list[str]) -> KeyDuplicates:
    data_rows = int(len(frame))
    if data_rows == 0:
        empty = _share(0, 0, "rows with a complete key")
        return KeyDuplicates(
            key_columns=keys,
            incomplete_key_rows=0,
            rows_in_duplicate_groups=empty,
            extra_copies=empty,
            status="not_comparable",
            detail=(
                "The unique-key check is separate from exact full-row duplicates. "
                "Not calculated because there are 0 data rows."
            ),
        )
    comparison = _comparison_frame(frame)
    incomplete_mask = pd.Series(False, index=comparison.index)
    for name in keys:
        incomplete_mask = incomplete_mask | comparison[name].map(lambda value: value is _MISSING)
    incomplete = int(incomplete_mask.sum())
    complete = comparison.loc[~incomplete_mask]
    in_groups, extra = _duplicate_counts(complete, keys)
    complete_rows = int(len(complete))
    if complete_rows == 0:
        status = "not_comparable"
        detail = (
            "The unique-key check is separate from exact full-row duplicates. "
            f"{incomplete} of {data_rows} data rows have an incomplete key and were excluded. "
            "Not calculated because no row has a complete key."
        )
    elif in_groups > 0:
        status = "violation"
        detail = (
            "User-selected unique-key rule. "
            f"{in_groups} of {complete_rows} rows with a complete key participate in a duplicate key group. "
            f"{extra} of those rows are extra copies beyond the first occurrence. "
            f"{incomplete} data rows were excluded because the key is incomplete. "
            "This result is separate from exact full-row duplicates."
        )
    else:
        status = "clear"
        detail = (
            "User-selected unique-key rule. "
            f"0 of {complete_rows} rows with a complete key are duplicate keys. "
            f"{incomplete} data rows were excluded because the key is incomplete. "
            "This result is separate from exact full-row duplicates."
        )
    return KeyDuplicates(
        key_columns=keys,
        incomplete_key_rows=incomplete,
        rows_in_duplicate_groups=_share(in_groups, complete_rows, "rows with a complete key"),
        extra_copies=_share(extra, complete_rows, "rows with a complete key"),
        status=status,
        detail=detail,
    )


def _duplicate_counts(frame: pd.DataFrame, columns: list[str]) -> tuple[int, int]:
    if len(frame) == 0 or not columns:
        return 0, 0
    comparison = _comparison_frame(frame)
    subset = comparison[columns]
    in_groups = int(subset.duplicated(keep=False).sum())
    extra = int(subset.duplicated(keep="first").sum())
    return in_groups, extra


def _comparison_frame(frame: pd.DataFrame) -> pd.DataFrame:
    compared = pd.DataFrame(index=frame.index)
    for name in frame.columns:
        compared[name] = frame[name].map(_comparison_value)
    return compared


def _comparison_value(value: object) -> object:
    if value is _MISSING or is_missing(value):
        return _MISSING
    if isinstance(value, (str, bool, np.bool_, datetime, date)):
        return value
    if isinstance(value, numbers.Real):
        return float(value)
    try:
        hash(value)
    except TypeError:
        return ("unhashable", type(value).__name__)
    return value


def _share(count: int, denominator: int, unit: str) -> Share:
    if denominator == 0:
        return Share(count=0, denominator=0, ratio=None, unit=unit)
    return Share(count=count, denominator=denominator, ratio=count / denominator, unit=unit)
