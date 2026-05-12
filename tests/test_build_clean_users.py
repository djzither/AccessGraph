from pathlib import Path
import numpy as np

import pandas as pd

from DataLayer.build_clean_users import build_clean_users


def _write_reference_excel(path: Path, sheet_name: str, rows: list[list[object]]) -> None:
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)


def test_build_clean_users_writes_parquet_and_groupslist(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_dir / "ce_ad_user_rights_all.xlsx"
    out_file = processed_dir / "clean_users.parquet"

    raw_df = pd.DataFrame(
        [
            {
                "SamAccountName": "user1",
                "DisplayName": "User One",
                "Title": "Help Desk",
                "Department": "IT",
                "Manager": "mgr1",
                "Groups": "DCE-DomainUsers; Cannot find an object with identity; m.Email",
            },
            {
                "SamAccountName": "user2",
                "DisplayName": "User Two",
                "Title": "Help Desk",
                "Department": "IT",
                "Manager": "mgr1",
                "Groups": "VPN;Email",
            },
        ]
    )
    raw_df.to_excel(raw_file, index=False)

    _write_reference_excel(
        raw_dir / "student_employee_access.xlsx",
        "Data Base",
        [
            ["Header note"],
            ["Title", "Dept", "Supervisor", "Access"],
            ["Help Desk", "IT", "mgr1", "VPN"],
        ],
    )
    _write_reference_excel(
        raw_dir / "full_time_employee_access.xlsx",
        "Full Time Employees Data Base",
        [
            ["Job Title", "Department", "Manager", "AD Rights"],
            ["Help Desk", "IT", "mgr1", "Email"],
        ],
    )

    reference_out = processed_dir / "access_reference.parquet"
    build_clean_users(
        raw_dir=raw_dir,
        raw_file=raw_file.name,
        output_path=out_file,
        reference_output_path=reference_out,
    )

    assert out_file.exists()
    built = pd.read_parquet(out_file)

    required = {
        "SamAccountName",
        "DisplayName",
        "Title",
        "Department",
        "Manager",
        "GroupsList",
        "EmployeeType",
    }
    assert required.issubset(set(built.columns))
    assert set(built["EmployeeType"]) <= {"FULL_TIME", "STUDENT"}

    groups0 = built.iloc[0]["GroupsList"]
    assert isinstance(groups0, (list, tuple, np.ndarray))
    groups0 = list(groups0)
    assert "Cannot find an object with identity" not in groups0
    assert "m.Email" not in groups0

    metadata_file = out_file.with_suffix(".metadata.json")
    assert metadata_file.exists()

    assert reference_out.exists()
    reference = pd.read_parquet(reference_out)
    assert list(reference.columns) == [
        "EmployeeType",
        "EmployeeTypeCanonical",
        "JobTitle",
        "Department",
        "Supervisor",
        "AccessCategory",
        "AccessName",
        "SourceFile",
    ]
    assert set(reference["AccessName"]) == {"VPN", "Email"}
