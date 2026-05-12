"""
tests/test_reference_loader.py
──────────────────────────────
Validates that both reference spreadsheets load correctly and that the
combined reference_df meets all structural requirements.

Run:
    python -m pytest tests/test_reference_loader.py -v
or standalone:
    python tests/test_reference_loader.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from DataLayer.reference_loader import (
    load_fulltime_reference,
    load_reference_df,
    load_student_reference,
)

STUDENT_PATH = Path("data/raw/student_employee_access.xlsx")
FULLTIME_PATH = Path("data/raw/full_time_employee_access.xlsx")

REQUIRED_OUTPUT_COLS = [
    "EmployeeType",
    "JobTitle",
    "Department",
    "Supervisor",
    "AccessCategory",
    "AccessName",
    "ReferenceEmployeeName",
]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def student_df():
    return load_student_reference(STUDENT_PATH, verbose=False)


@pytest.fixture(scope="module")
def fulltime_df():
    return load_fulltime_reference(FULLTIME_PATH, verbose=False)


@pytest.fixture(scope="module")
def combined_df():
    return load_reference_df(STUDENT_PATH, FULLTIME_PATH, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Individual file tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStudentFile:
    def test_loads_without_error(self, student_df):
        assert student_df is not None

    def test_not_empty(self, student_df):
        assert len(student_df) > 0, "Student DataFrame is empty"

    def test_has_required_columns(self, student_df):
        missing = set(REQUIRED_OUTPUT_COLS) - set(student_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_access_name_not_mostly_null(self, student_df):
        null_pct = student_df["AccessName"].isna().mean()
        assert null_pct < 0.5, f"AccessName is {null_pct*100:.1f}% null in student data"

    def test_employee_type_is_student(self, student_df):
        types = student_df["EmployeeType"].unique().tolist()
        assert types == ["Student"], f"Unexpected EmployeeType values: {types}"

    def test_no_completely_blank_access_rows(self, student_df):
        blank = student_df[
            student_df["AccessName"].isna()
            | (student_df["AccessName"].str.strip() == "")
        ]
        assert len(blank) == 0, f"{len(blank)} rows have blank AccessName"


class TestFullTimeFile:
    def test_loads_without_error(self, fulltime_df):
        assert fulltime_df is not None

    def test_not_empty(self, fulltime_df):
        assert len(fulltime_df) > 0, "FullTime DataFrame is empty"

    def test_has_required_columns(self, fulltime_df):
        missing = set(REQUIRED_OUTPUT_COLS) - set(fulltime_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_access_name_not_mostly_null(self, fulltime_df):
        null_pct = fulltime_df["AccessName"].isna().mean()
        assert null_pct < 0.5, f"AccessName is {null_pct*100:.1f}% null in fulltime data"

    def test_employee_type_is_fulltime(self, fulltime_df):
        types = fulltime_df["EmployeeType"].unique().tolist()
        assert types == ["Full Time"], f"Unexpected EmployeeType values: {types}"

    def test_no_completely_blank_access_rows(self, fulltime_df):
        blank = fulltime_df[
            fulltime_df["AccessName"].isna()
            | (fulltime_df["AccessName"].str.strip() == "")
        ]
        assert len(blank) == 0, f"{len(blank)} rows have blank AccessName"

    def test_reference_employee_name_populated(self, fulltime_df):
        null_pct = fulltime_df["ReferenceEmployeeName"].isna().mean()
        assert null_pct < 0.3, (
            f"ReferenceEmployeeName is {null_pct*100:.1f}% null in fulltime data"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Combined DataFrame tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedDf:
    def test_not_empty(self, combined_df):
        assert len(combined_df) > 0

    def test_has_all_required_columns(self, combined_df):
        missing = set(REQUIRED_OUTPUT_COLS) - set(combined_df.columns)
        assert not missing, f"Missing columns in combined_df: {missing}"

    def test_columns_exact(self, combined_df):
        assert list(combined_df.columns) == REQUIRED_OUTPUT_COLS, (
            f"Column order/set mismatch: {list(combined_df.columns)}"
        )

    def test_access_name_has_non_null_values(self, combined_df):
        non_null = combined_df["AccessName"].notna().sum()
        assert non_null > 0, "AccessName has zero non-null values"

    def test_employee_type_distinguishes_both(self, combined_df):
        types = set(combined_df["EmployeeType"].unique())
        assert "Student" in types, "No Student rows in combined_df"
        assert "Full Time" in types, "No Full Time rows in combined_df"

    def test_no_blank_access_rows(self, combined_df):
        blank = combined_df[
            combined_df["AccessName"].isna()
            | (combined_df["AccessName"].str.strip() == "")
        ]
        assert len(blank) == 0, f"{len(blank)} blank AccessName rows remain"

    def test_student_row_count_reasonable(self, combined_df):
        student_rows = (combined_df["EmployeeType"] == "Student").sum()
        assert student_rows > 200, f"Only {student_rows} student rows; expected > 200"

    def test_fulltime_row_count_reasonable(self, combined_df):
        ft_rows = (combined_df["EmployeeType"] == "Full Time").sum()
        assert ft_rows > 500, f"Only {ft_rows} full-time rows; expected > 500"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (no pytest required)
# ─────────────────────────────────────────────────────────────────────────────

def _run_standalone() -> None:
    print("=" * 60)
    print("Reference Loader Validation")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {label}")
            passed += 1
        else:
            print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))
            failed += 1

    try:
        student = load_student_reference(STUDENT_PATH, verbose=True)
        fulltime = load_fulltime_reference(FULLTIME_PATH, verbose=True)
        combined = load_reference_df(STUDENT_PATH, FULLTIME_PATH, verbose=True)
    except Exception as exc:
        print(f"\nFATAL: Could not load files — {exc}")
        raise

    print("\n── Student checks ──")
    check("not empty", len(student) > 0, f"got {len(student)} rows")
    check("required cols present", not (set(REQUIRED_OUTPUT_COLS) - set(student.columns)))
    check("AccessName not mostly null", student["AccessName"].isna().mean() < 0.5)
    check("EmployeeType == Student", set(student["EmployeeType"].unique()) == {"Student"})
    check("no blank AccessName", (student["AccessName"].fillna("").str.strip() != "").all())

    print("\n── FullTime checks ──")
    check("not empty", len(fulltime) > 0)
    check("required cols present", not (set(REQUIRED_OUTPUT_COLS) - set(fulltime.columns)))
    check("AccessName not mostly null", fulltime["AccessName"].isna().mean() < 0.5)
    check("EmployeeType == Full Time", set(fulltime["EmployeeType"].unique()) == {"Full Time"})
    check("no blank AccessName", (fulltime["AccessName"].fillna("").str.strip() != "").all())

    print("\n── Combined checks ──")
    check("not empty", len(combined) > 0)
    check("required cols present", not (set(REQUIRED_OUTPUT_COLS) - set(combined.columns)))
    check("AccessName has values", combined["AccessName"].notna().sum() > 0)
    check("has Student rows", "Student" in combined["EmployeeType"].values)
    check("has Full Time rows", "Full Time" in combined["EmployeeType"].values)
    check("no blank AccessName rows",
          (combined["AccessName"].fillna("").str.strip() != "").all())
    check("student row count > 200",
          (combined["EmployeeType"] == "Student").sum() > 200)
    check("fulltime row count > 500",
          (combined["EmployeeType"] == "Full Time").sum() > 500)

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_standalone()
