"""
DataLayer/reference_loader.py
──────────────────────────────────────────────────────────────────────────────
Loads student and full-time employee reference access spreadsheets and
produces one clean, normalised reference_df.

Evidence gathered from direct file inspection:
───────────────────────────────────────────────────────
STUDENT  (student_employee_access.xlsx)
  Sheet : "Data Base"
  Header: Row 0 (first row)
  Rows  : 109 data rows
  Key columns (0-indexed):
    00 Supervisors       → Supervisor
    02 Job Title         → JobTitle
    04 Department        → Department
    07-17                → access category columns (newline-delimited values)
  No dedicated employee/reference name column.

FULL-TIME  (full_time_employee_access.xlsx)
  Sheet : "Full Time Employees Data Base"
  Header: Row 0 (first row)
  Rows  : 297 data rows
  Key columns (0-indexed):
    00 Employee                    → ReferenceEmployeeName
    01 Direct Report / Supervisor  → Supervisor
    02 Job Title                   → JobTitle
    03 Department                  → Department
    08-17                          → access category columns

Both files store multiple access values in a single cell separated by "\\n".
There is NO single "AccessCategory / AccessName" column pair; instead every
named access column (AD Rights, Email Groups, HCEB Doors, Orion, etc.) is
itself an access category, and each newline-delimited value inside it is one
AccessName.

The loader pivots these wide access columns into long rows so the output has
one row per (employee × access-name).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from DataLayer.access_exclusions import filter_reference_df, is_excluded_access
from DataLayer.permission_normalization import normalize_single_permission

# ── Public paths (override via load_reference_df arguments) ──────────────────
DEFAULT_STUDENT_PATH = Path("data/raw/student_employee_access.xlsx")
DEFAULT_FULLTIME_PATH = Path("data/raw/full_time_employee_access.xlsx")

# ── Access-category columns to extract from each file ────────────────────────
STUDENT_ACCESS_COLS = [
    "HCEB Doors",
    "AD Rights",
    "Email Groups",
    "Cvent",
    "Orion",
    "Orion Test",
    "FSY Orion\n(FSY Manager)",
    "CRM Access",
    "Adobe",
    "Drupal",
    "Extras",
    "Teamwork",
]

FULLTIME_ACCESS_COLS = [
    "HCEB Doors",
    "AD Rights",
    "Email Group",
    "Email Folders",
    "Cvent",
    "Box Access",
    "Tableau",
    "CRM Access",
    "Extras",
    "Orion/Orion Test/FSY Orion",
    "Teamwork Company",
]

# ── Alternate column-name aliases (schema tolerance) ─────────────────────────
_ALIASES: dict[str, list[str]] = {
    "JobTitle": [
        "Job Title", "JobTitle", "Title", "job_title", "jobtitle",
    ],
    "Department": [
        "Department", "Dept", "dept", "department",
    ],
    "Supervisor": [
        "Supervisors", "Direct Report / Supervisor", "Supervisor",
        "Manager", "supervisor", "manager",
    ],
    "ReferenceEmployeeName": [
        "Employee", "Employee Name", "Reference Employee",
        "Name", "employee", "name",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_ws(text: object) -> str:
    """Strip and collapse internal whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _resolve_col(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Return the first alias of *canonical* that exists as a df column."""
    for alias in _ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return None


def _split_access_values(cell_value: object) -> list[str]:
    """Split a newline-delimited access cell into individual access names."""
    if not isinstance(cell_value, str):
        return []
    parts = [_normalise_ws(p) for p in cell_value.split("\n")]
    out: list[str] = []
    for p in parts:
        t = normalize_single_permission(p)
        if t:
            out.append(t)
    return out


def _load_raw(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read an Excel sheet with header on row 0 (the first row)."""
    df = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=0,
        dtype=str,
        engine="openpyxl",
    )
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    return df


def _melt_access_columns(
    df: pd.DataFrame,
    access_cols: list[str],
    id_vars: dict[str, str],
    employee_type: str,
) -> pd.DataFrame:
    """
    Pivot wide access columns into long rows.

    For each access column, split cell values on newline and emit one row per
    (employee × access-name).  The column name itself becomes AccessCategory.
    """
    records: list[dict] = []
    present_access_cols = [c for c in access_cols if c in df.columns]

    for _, row in df.iterrows():
        base: dict = {"EmployeeType": employee_type}
        for output_col, source_col in id_vars.items():
            base[output_col] = _normalise_ws(row.get(source_col, "")) or None

        for acc_col in present_access_cols:
            raw = row.get(acc_col, None)
            names = _split_access_values(raw)
            for name in names:
                if is_excluded_access(acc_col, name):
                    continue
                rec = dict(base)
                rec["AccessCategory"] = _normalise_ws(acc_col)
                an = normalize_single_permission(name)
                if not an:
                    continue
                rec["AccessName"] = an
                records.append(rec)

    return filter_reference_df(pd.DataFrame(records))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_student_reference(
    path: Path | str = DEFAULT_STUDENT_PATH,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the student employee access spreadsheet."""
    path = Path(path)
    sheet = "Data Base"

    if verbose:
        print(f"\n[student] Loading {path}")
        print(f"[student] Sheet: '{sheet}'")

    raw = _load_raw(path, sheet)

    if verbose:
        print(f"[student] Rows loaded (after blank-row drop): {len(raw)}")

    id_vars: dict[str, str] = {}
    for canonical in ("JobTitle", "Department", "Supervisor"):
        resolved = _resolve_col(raw, canonical)
        if resolved:
            id_vars[canonical] = resolved
        else:
            warnings.warn(
                f"[student] Could not find column for '{canonical}'. "
                f"Available: {list(raw.columns)}"
            )

    df_long = _melt_access_columns(
        raw,
        access_cols=STUDENT_ACCESS_COLS,
        id_vars=id_vars,
        employee_type="Student",
    )
    if "ReferenceEmployeeName" not in df_long.columns:
        df_long["ReferenceEmployeeName"] = None

    if verbose:
        print(f"[student] Rows after melt (one per access name): {len(df_long)}")
        _warn_if_access_null(df_long, "student")

    return df_long


def load_fulltime_reference(
    path: Path | str = DEFAULT_FULLTIME_PATH,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the full-time employee access spreadsheet."""
    path = Path(path)
    sheet = "Full Time Employees Data Base"

    if verbose:
        print(f"\n[fulltime] Loading {path}")
        print(f"[fulltime] Sheet: '{sheet}'")

    raw = _load_raw(path, sheet)

    if verbose:
        print(f"[fulltime] Rows loaded (after blank-row drop): {len(raw)}")

    id_vars: dict[str, str] = {}
    for canonical in ("ReferenceEmployeeName", "JobTitle", "Department", "Supervisor"):
        resolved = _resolve_col(raw, canonical)
        if resolved:
            id_vars[canonical] = resolved
        else:
            warnings.warn(
                f"[fulltime] Could not find column for '{canonical}'. "
                f"Available: {list(raw.columns)}"
            )

    df_long = _melt_access_columns(
        raw,
        access_cols=FULLTIME_ACCESS_COLS,
        id_vars=id_vars,
        employee_type="Full Time",  # matches AccessRecommendationEngine expectations
    )

    if verbose:
        print(f"[fulltime] Rows after melt (one per access name): {len(df_long)}")
        _warn_if_access_null(df_long, "fulltime")

    return df_long


def load_reference_df(
    student_path: Path | str = DEFAULT_STUDENT_PATH,
    fulltime_path: Path | str = DEFAULT_FULLTIME_PATH,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load both reference spreadsheets and return one combined, normalised
    reference_df.

    Output columns (always present)
    ────────────────────────────────
    EmployeeType          : "Student" | "Full Time"
    JobTitle              : str | None
    Department            : str | None
    Supervisor            : str | None
    AccessCategory        : str
    AccessName            : str
    ReferenceEmployeeName : str | None
    """
    student_df = load_student_reference(student_path, verbose=verbose)
    fulltime_df = load_fulltime_reference(fulltime_path, verbose=verbose)

    combined = filter_reference_df(pd.concat([student_df, fulltime_df], ignore_index=True))

    required_cols = [
        "EmployeeType",
        "JobTitle",
        "Department",
        "Supervisor",
        "AccessCategory",
        "AccessName",
        "ReferenceEmployeeName",
    ]
    for col in required_cols:
        if col not in combined.columns:
            combined[col] = None

    combined = combined[required_cols].copy()

    before = len(combined)
    combined.dropna(subset=["AccessName"], inplace=True)
    combined = combined[combined["AccessName"].str.strip() != ""]
    after = len(combined)

    if verbose:
        print(f"\n[combined] Total rows before blank-access drop : {before}")
        print(f"[combined] Total rows after  blank-access drop : {after}")
        print(f"[combined] Final columns: {list(combined.columns)}")
        print(f"\n[combined] EmployeeType distribution:")
        print(combined["EmployeeType"].value_counts().to_string())
        print(f"\n[combined] Sample rows:")
        print(combined.head(5).to_string(index=False))

    return combined.reset_index(drop=True)


# ── Diagnostic helper ────────────────────────────────────────────────────────

def _warn_if_access_null(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        warnings.warn(f"[{label}] DataFrame is empty after processing.")
        return
    null_pct = df["AccessName"].isna().mean() * 100
    if null_pct > 50:
        warnings.warn(
            f"[{label}] AccessName is {null_pct:.1f}% null — "
            "check that the correct access columns are listed in "
            f"{'STUDENT_ACCESS_COLS' if label == 'student' else 'FULLTIME_ACCESS_COLS'}."
        )


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    student_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STUDENT_PATH
    fulltime_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FULLTIME_PATH

    df = load_reference_df(student_path, fulltime_path, verbose=True)
    print(f"\nDone. Shape: {df.shape}")
