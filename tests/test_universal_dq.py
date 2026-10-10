"""Tests for the schema-independent CSV and Excel analyzer.

These tests build tables in memory. They do not call a model, start Streamlit,
or read the banking ground-truth manifest.
"""

from __future__ import annotations

import ast
import json
import sys
from io import BytesIO
from pathlib import Path

import pytest
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ai_agents.findings_loader import load_findings_report  # noqa: E402
from universal_dq.loader import (  # noqa: E402
    MAX_UPLOAD_BYTES,
    FileLoadError,
    inspect_upload,
    load_tabular_file,
)
from universal_dq.profile import AnalysisInputError, analyze_frame  # noqa: E402
from universal_dq.report import report_to_csv, report_to_json  # noqa: E402

FINDINGS_PATH = ROOT / "outputs" / "dq_findings.json"
UNIVERSAL_DIR = SRC / "universal_dq"


def _csv(text: str) -> bytes:
    return text.encode("utf-8")


def _xlsx(sheets: dict[str, list[list[object]]]) -> bytes:
    workbook = Workbook()
    names = list(sheets)
    first = workbook.active
    first.title = names[0]
    for name in names[1:]:
        workbook.create_sheet(name)
    for name, rows in sheets.items():
        worksheet = workbook[name]
        for row in rows:
            worksheet.append(row)
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _analyze_csv(text: str, **kwargs):
    loaded = load_tabular_file(_csv(text), "upload.csv")
    return analyze_frame(
        loaded.frame,
        source_name=loaded.source_name,
        file_format=loaded.file_format,
        sheet_name=loaded.sheet_name,
        **kwargs,
    )


def test_arbitrary_column_names_are_profiled() -> None:
    report = _analyze_csv("Employee,Department,Salary,Joining Date\nAna,HR,10,2020-01-15\n")
    assert report.column_names == ["Employee", "Department", "Salary", "Joining Date"]
    assert report.data_rows == 1
    assert report.column_count == 4
    assert report.columns[0].inferred_type == "text"
    assert report.columns[2].inferred_type == "number"
    assert report.columns[3].inferred_type == "datetime"


def test_literal_na_and_null_are_kept_and_whitespace_is_missing() -> None:
    report = _analyze_csv("Code,Note\nNA,NULL\n   ,ok\n")
    code = report.columns[0]
    note = report.columns[1]
    assert code.missing.count == 1
    assert code.missing.denominator == 2
    assert code.missing.ratio == pytest.approx(0.5)
    assert note.missing.count == 0
    assert code.missing_role == "observation"
    assert note.inferred_type == "text"


def test_exact_duplicate_groups_and_extra_copies() -> None:
    report = _analyze_csv("Code,Amount\nE1,10\nE1,20\nE2,10\nE2,10\nE3,1\n")
    groups = report.exact_duplicates.rows_in_duplicate_groups
    extra = report.exact_duplicates.extra_copies
    assert groups.count == 2
    assert extra.count == 1
    assert groups.denominator == 5
    assert extra.denominator == 5
    assert groups.ratio == pytest.approx(2 / 5)
    assert extra.ratio == pytest.approx(1 / 5)


def test_three_copies_count_two_extra_rows() -> None:
    report = _analyze_csv("Item\nA\nA\nA\nB\n")
    assert report.exact_duplicates.rows_in_duplicate_groups.count == 3
    assert report.exact_duplicates.extra_copies.count == 2


def test_unique_key_duplicates_stay_separate_from_full_rows() -> None:
    text = "Code,Amount\nE1,10\nE1,20\nE2,10\nE2,10\n,5\n"
    report = _analyze_csv(text, key_columns=["Code"])
    assert report.exact_duplicates.rows_in_duplicate_groups.count == 2
    assert report.exact_duplicates.extra_copies.count == 1
    key = report.key_duplicates
    assert key is not None
    assert key.status == "violation"
    assert key.incomplete_key_rows == 1
    assert key.rows_in_duplicate_groups.count == 4
    assert key.rows_in_duplicate_groups.denominator == 4
    assert key.extra_copies.count == 2
    assert key.rows_in_duplicate_groups.count != report.exact_duplicates.rows_in_duplicate_groups.count


def test_required_column_is_a_user_rule_and_other_missing_values_are_observations() -> None:
    text = "Code,Note\nNA,ok\nE2,   \n"
    observed = _analyze_csv(text)
    required = _analyze_csv(text, required_columns=["Note"])
    assert observed.columns[1].missing_role == "observation"
    assert required.columns[1].missing_role == "user_rule_violation"
    assert required.columns[1].missing.count == 1
    assert required.columns[0].missing_role == "observation"


def test_mixed_types_and_outliers_are_informational_and_omit_cell_values() -> None:
    rows = ["Kind,Amount"]
    rows.extend(f"zebra-text-token-42,{number}" for number in [1, 2, 3, 4, 5, 6, 7])
    rows.append("10,999999")
    report = _analyze_csv("\n".join(rows) + "\n")
    mixed = report.columns[0].mixed_type
    outlier = report.columns[1].outlier
    assert mixed is not None
    assert mixed.label == "informational"
    assert mixed.categories["text"] == 7
    assert mixed.categories["number"] == 1
    assert outlier is not None
    assert outlier.label == "informational"
    assert outlier.outlier_count == 1
    assert outlier.denominator == 8
    payload = report_to_json(report) + report_to_csv(report)
    assert "zebra-text-token-42" not in payload
    assert "999999" not in payload


def test_outlier_check_skips_small_or_constant_columns() -> None:
    small = _analyze_csv("Amount\n1\n2\n3\n4\n5\n6\n7\n")
    constant = _analyze_csv("Amount\n" + "\n".join(["5"] * 8) + "\n")
    assert small.columns[0].outlier is not None
    assert small.columns[0].outlier.label == "skipped"
    assert "fewer than 8" in small.columns[0].outlier.detail
    assert constant.columns[0].outlier is not None
    assert "no spread" in constant.columns[0].outlier.detail


def test_clean_file_has_no_missing_values_or_duplicates() -> None:
    report = _analyze_csv("Product,Region\nPen,East\nBook,West\n")
    assert report.columns[0].missing.count == 0
    assert report.columns[0].missing.ratio == 0
    assert report.exact_duplicates.rows_in_duplicate_groups.count == 0
    assert report.exact_duplicates.extra_copies.ratio == 0
    assert report.key_duplicates is None


def test_empty_dataset_does_not_divide_by_zero() -> None:
    report = _analyze_csv("Employee,Department\n")
    assert report.data_rows == 0
    assert report.columns[0].missing.ratio is None
    assert report.exact_duplicates.rows_in_duplicate_groups.ratio is None
    assert report.exact_duplicates.extra_copies.ratio is None
    keyed = _analyze_csv("Employee,Department\n", key_columns=["Employee"])
    assert keyed.key_duplicates is not None
    assert keyed.key_duplicates.rows_in_duplicate_groups.ratio is None
    assert keyed.key_duplicates.status == "not_comparable"


def test_excel_worksheet_selection_reads_only_the_selected_sheet() -> None:
    content = _xlsx(
        {
            "Summary": [["Token"], ["only-on-sheet-one"]],
            "Orders": [["Product", "Region"], ["Pen", "East"], ["Pen", "East"]],
        }
    )
    inspection = inspect_upload(content, "book.xlsx")
    assert inspection.sheet_names == ("Summary", "Orders")
    loaded = load_tabular_file(content, "book.xlsx", sheet_name="Orders")
    report = analyze_frame(
        loaded.frame,
        source_name=loaded.source_name,
        file_format=loaded.file_format,
        sheet_name=loaded.sheet_name,
    )
    assert report.sheet_name == "Orders"
    assert report.column_names == ["Product", "Region"]
    assert report.exact_duplicates.extra_copies.count == 1
    assert "only-on-sheet-one" not in report_to_json(report)


def test_duplicate_and_blank_headers_are_rejected() -> None:
    with pytest.raises(FileLoadError, match="repeats"):
        load_tabular_file(_csv("Salary,Department,Salary\n1,HR,2\n"), "pay.csv")
    with pytest.raises(FileLoadError, match="blank"):
        load_tabular_file(_csv("Employee,,Salary\nAna,,10\n"), "pay.csv")
    content = _xlsx({"Sheet1": [["Vendor", "Vendor"], ["A", "B"]]})
    with pytest.raises(FileLoadError, match="repeats"):
        load_tabular_file(content, "vendors.xlsx")


def test_empty_malformed_and_unsupported_files_have_clear_errors() -> None:
    with pytest.raises(FileLoadError, match="empty"):
        load_tabular_file(b"", "empty.csv")
    with pytest.raises(FileLoadError, match="UTF-8"):
        load_tabular_file("Café".encode("latin-1"), "notes.csv")
    with pytest.raises(FileLoadError, match="not supported"):
        load_tabular_file(b"hello", "notes.xls")
    with pytest.raises(FileLoadError, match="could not be opened"):
        load_tabular_file(b"this is not a workbook", "broken.xlsx")
    with pytest.raises(FileLoadError, match="no header row"):
        load_tabular_file(_xlsx({"Sheet1": []}), "blank.xlsx")


def test_upload_limit_is_20_mb() -> None:
    assert MAX_UPLOAD_BYTES == 20 * 1024 * 1024
    with pytest.raises(FileLoadError, match="20 MB"):
        load_tabular_file(b"x" * (MAX_UPLOAD_BYTES + 1), "big.csv")


def test_unknown_column_selection_is_rejected() -> None:
    loaded = load_tabular_file(_csv("Product,Price\nPen,2\n"), "shop.csv")
    with pytest.raises(AnalysisInputError, match="not in the file"):
        analyze_frame(
            loaded.frame,
            source_name="shop.csv",
            file_format="csv",
            sheet_name=None,
            key_columns=["customer_id"],
        )


def test_report_downloads_include_denominators_without_raw_rows() -> None:
    report = _analyze_csv("Vendor,Invoice Amount\nAcme,secret-cell-value\nAcme,secret-cell-value\n")
    payload = json.loads(report_to_json(report))
    csv_text = report_to_csv(report)
    assert payload["exact_duplicates"]["extra_copies"]["count"] == 1
    assert payload["exact_duplicates"]["rows_in_duplicate_groups"]["denominator"] == 2
    assert "secret-cell-value" not in report_to_json(report)
    assert "secret-cell-value" not in csv_text
    assert "rows_in_duplicate_groups" in csv_text
    assert "extra_copies" in csv_text
    assert "processed by the Streamlit server" in payload["processing_note"]


def test_whitespace_only_duplicates_match_and_original_text_is_preserved() -> None:
    loaded = load_tabular_file(_csv("Code,Note\nNA,   \nNA,\n"), "codes.csv")
    assert loaded.frame.iloc[0, 0] == "NA"
    report = analyze_frame(
        loaded.frame,
        source_name="codes.csv",
        file_format="csv",
        sheet_name=None,
    )
    assert loaded.frame.iloc[0, 0] == "NA"
    assert report.exact_duplicates.rows_in_duplicate_groups.count == 2
    assert report.columns[0].missing.count == 0
    assert report.columns[1].missing.count == 2


def test_banking_dashboard_data_still_loads() -> None:
    report = load_findings_report(FINDINGS_PATH)
    assert [item.rule_id for item in report.findings] == [
        "DQ001",
        "DQ002",
        "DQ003",
        "DQ004",
        "DQ005",
        "DQ006",
        "DQ007",
    ]
    source = (SRC / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "Analyze a file" in source
    assert "render_overview" in source
    assert "render_upload" in source


def test_analyzer_does_not_import_the_banking_engine_or_an_llm() -> None:
    forbidden_modules = ("crewai", "dq_engine", "google.genai", "google.generativeai")
    dashboard_files = list((SRC / "dashboard").glob("*.py"))
    for path in list(UNIVERSAL_DIR.glob("*.py")) + dashboard_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                for forbidden in forbidden_modules:
                    assert not module.startswith(forbidden)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "injected_issues" not in node.value
                assert "GEMINI_API_KEY" not in node.value
                assert "customer_id" not in node.value
