"""Schema-independent checks for an uploaded CSV or Excel worksheet.

This package does not import the banking data-quality engine or CrewAI.
"""

from .batch import LoadedDataset, build_batch_report, collect_datasets
from .cleaning import (
    CleaningRejected,
    CleaningSession,
    apply_operation,
    content_fingerprint,
    new_cleaning_session,
    preview_operation,
    reset,
    undo,
)
from .cleaning_export import cleaned_dataset_csv, cleaned_datasets_xlsx, cleaning_audit_xlsx
from .loader import (
    MAX_TOTAL_UPLOAD_BYTES,
    MAX_UPLOAD_BYTES,
    FileLoadError,
    LoadedTable,
    UploadInspection,
    inspect_upload,
    load_tabular_file,
)
from .models import AnalysisReport, BatchReport
from .profile import AnalysisInputError, analyze_frame
from .report import report_to_csv, report_to_json, report_to_xlsx

__all__ = [
    "MAX_TOTAL_UPLOAD_BYTES",
    "MAX_UPLOAD_BYTES",
    "AnalysisInputError",
    "AnalysisReport",
    "BatchReport",
    "FileLoadError",
    "LoadedDataset",
    "LoadedTable",
    "UploadInspection",
    "CleaningRejected",
    "CleaningSession",
    "analyze_frame",
    "apply_operation",
    "build_batch_report",
    "cleaned_dataset_csv",
    "cleaned_datasets_xlsx",
    "cleaning_audit_xlsx",
    "collect_datasets",
    "content_fingerprint",
    "inspect_upload",
    "load_tabular_file",
    "new_cleaning_session",
    "preview_operation",
    "report_to_csv",
    "report_to_json",
    "report_to_xlsx",
    "reset",
    "undo",
]
