from pathlib import Path

import pandas as pd

from DataLayer.access_exclusions import (
    count_excluded_group_entries,
    count_excluded_reference_rows,
)
from DataLayer.cleaner import DataCleaner
from DataLayer.rights_sheets_loader import RightsSheetsLoader
from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


def _write_excel(path: Path, sheet_name: str, rows: list[list[object]]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, header=False, index=False)


def test_cleaner_removes_crm_and_salesforce_groups():
    raw = pd.DataFrame(
        [
            {
                "Groups": "VPN;CRM.Users;Salesforce.Admin;Email",
            }
        ]
    )

    cleaned = DataCleaner().clean_groups(raw)

    assert cleaned["GroupsList"].iloc[0] == ["VPN", "Email"]
    assert count_excluded_group_entries(cleaned) == 0


def test_reference_loader_excludes_crm_access_columns_and_keywords(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_excel(
        raw_dir / "student_employee_access.xlsx",
        "Data Base",
        [
            ["Job Title", "Department", "Supervisor", "AD Rights", "CRM Access"],
            ["Assistant", "IT", "Boss", "VPN", "Salesforce Admin\nCRM Portal"],
        ],
    )

    parsed = RightsSheetsLoader(raw_dir)._load_and_normalize(
        "student_employee_access.xlsx",
        "Student",
    )

    assert parsed["AccessName"].tolist() == ["VPN"]
    assert "CRM Access" not in set(parsed["AccessCategory"])
    assert count_excluded_reference_rows(parsed) == 0


def test_recommendation_engine_does_not_generate_crm_recommendations():
    users = pd.DataFrame(
        [
            {
                "SamAccountName": f"user{i}",
                "DisplayName": f"User {i}",
                "Title": "Analyst",
                "Department": "IT",
                "GroupsList": ["VPN", "CRM.Users", "Salesforce.Admin"],
            }
            for i in range(5)
        ]
    )
    reference = pd.DataFrame(
        [
            {
                "EmployeeType": "Student",
                "JobTitle": "Analyst",
                "Department": "IT",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "VPN",
                "SourceFile": "student_employee_access.xlsx",
            },
            {
                "EmployeeType": "Student",
                "JobTitle": "Analyst",
                "Department": "IT",
                "Supervisor": None,
                "AccessCategory": "CRM Access",
                "AccessName": "Salesforce.Admin",
                "SourceFile": "student_employee_access.xlsx",
            },
        ]
    )

    recs = AccessRecommendationEngine(min_confidence=0.4).recommend_for_hire(
        users_df=users,
        reference_df=reference,
        title="Analyst",
        department="IT",
        employee_type="Student",
    )

    assert not recs["GroupName"].str.contains("crm|salesforce", case=False, na=False).any()
    assert "VPN" in recs["GroupName"].tolist()
