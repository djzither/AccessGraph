from pathlib import Path
import warnings

import pandas as pd
import pytest

from DataLayer.rights_sheets_loader import RightsSheetsLoader


EXPECTED_SCHEMA = [
    "EmployeeType",
    "EmployeeTypeCanonical",
    "JobTitle",
    "Department",
    "Supervisor",
    "AccessCategory",
    "AccessName",
    "SourceFile",
]


def _write_excel(path: Path, sheet_name: str, rows: list[list[object]]) -> None:
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)


def test_detects_header_row_and_explicit_access_aliases(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "student_employee_access.xlsx"
    _write_excel(
        source,
        "Data Base",
        [
            ["Student access reference"],
            [None],
            ["Supervisor", "Title", "Dept", "Category", "Permission"],
            ["Tamara Moss", "Customer Service Rep", "FS", "AD Rights", "DCE.STUDENT\nDCE.IS.STUDENT"],
            [None, None, None, None, None],
            ["Notes: do not import this row"],
        ],
    )

    loader = RightsSheetsLoader(raw_dir)
    parsed = loader._load_and_normalize("student_employee_access.xlsx", "Student")

    assert list(parsed.columns) == EXPECTED_SCHEMA
    assert parsed["AccessName"].tolist() == ["DCE.STUDENT", "DCE.IS.STUDENT"]
    assert parsed["Department"].tolist() == ["financial services", "financial services"]
    assert parsed["SourceFile"].unique().tolist() == ["student_employee_access.xlsx"]
    assert loader.validation[0]["header_row"] == 2


def test_parses_wide_access_category_columns(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "full_time_employee_access.xlsx"
    _write_excel(
        source,
        "Full Time Employees Data Base",
        [
            ["Human title row"],
            ["Employee", "Manager", "JobTitle", "Department", "AD Rights", "Email Group"],
            ["Jane Smith", "Boss Person", "Program Administrator", "IT", "VPN\nDCE-DomainUsers", "DCE.STAFF"],
        ],
    )

    loader = RightsSheetsLoader(raw_dir)
    parsed = loader._load_and_normalize("full_time_employee_access.xlsx", "Full Time")

    assert list(parsed.columns) == EXPECTED_SCHEMA
    assert set(parsed["AccessCategory"]) == {"AD Rights", "Email Group"}
    assert set(parsed["AccessName"]) == {"VPN", "DCE-DomainUsers", "DCE.STAFF"}
    assert set(parsed["Supervisor"]) == {"Boss Person"}


def test_load_reference_sheets_combines_both_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_excel(
        raw_dir / "student_employee_access.xlsx",
        "Data Base",
        [
            ["Title", "Department", "Access"],
            ["Student Assistant", "FS", "DCE.STUDENT"],
        ],
    )
    _write_excel(
        raw_dir / "full_time_employee_access.xlsx",
        "Full Time Employees Data Base",
        [
            ["Job Title", "Department", "AD Rights"],
            ["Accountant", "Finance", "DCE.FINANCE"],
        ],
    )

    parsed = RightsSheetsLoader(raw_dir).load_reference_sheets()

    assert list(parsed.columns) == EXPECTED_SCHEMA
    assert len(parsed) == 2
    assert set(parsed["EmployeeType"]) == {"Student", "Full Time"}
    assert set(parsed["EmployeeTypeCanonical"]) == {"STUDENT", "FULL_TIME"}
    assert set(parsed["SourceFile"]) == {
        "student_employee_access.xlsx",
        "full_time_employee_access.xlsx",
    }


def test_warns_when_access_name_is_mostly_empty(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_excel(
        raw_dir / "student_employee_access.xlsx",
        "Data Base",
        [
            ["Job Title", "Department", "Access Name"],
            ["Assistant", "IT", None],
            ["Assistant", "IT", ""],
            ["Assistant", "IT", "VPN"],
        ],
    )

    loader = RightsSheetsLoader(raw_dir)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parsed = loader._load_and_normalize("student_employee_access.xlsx", "Student")

    assert parsed["AccessName"].tolist() == ["VPN"]
    assert any("AccessName is mostly empty" in str(w.message) for w in caught)


def test_missing_access_mapping_fails(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_excel(
        raw_dir / "student_employee_access.xlsx",
        "Data Base",
        [
            ["Job Title", "Department"],
            ["Assistant", "IT"],
        ],
    )

    with pytest.raises(ValueError, match="AccessName is mostly empty"):
        RightsSheetsLoader(raw_dir)._load_and_normalize("student_employee_access.xlsx", "Student")
