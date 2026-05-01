from pathlib import Path
import pandas as pd

from DataLayer.loader import DataLoader


class RightsSheetsLoader:
    EXCLUDED_ACCESS_CATEGORIES = {
        "area",
        "extras",
        "ring central",
        "cvent",
        "orion",
        "orion test",
        "adobe",
    }

    def __init__(self, raw_path: str):
        self.loader = DataLoader(base_path=raw_path)

    def load_reference_sheets(self) -> pd.DataFrame:
        full_time = self._load_and_normalize(
            file_name="full_time_employee_access.xlsx",
            employee_type="Full Time",
        )

        student = self._load_and_normalize(
            file_name="student_employee_access.xlsx",
            employee_type="Student",
        )

        return pd.concat([full_time, student], ignore_index=True)

    def _load_and_normalize(self, file_name: str, employee_type: str) -> pd.DataFrame:
        df = self.loader.load_file(file_name)

        df.columns = [str(col).strip() for col in df.columns]

        possible_role_cols = ["Job Title", "Title", "JobTitle"]
        role_col = next((col for col in possible_role_cols if col in df.columns), None)

        if role_col is None:
            raise ValueError(f"Could not find job title column in {file_name}")

        department_col = "Department" if "Department" in df.columns else None
        supervisor_col = None

        for col in df.columns:
            if "supervisor" in col.lower() or "report" in col.lower():
                supervisor_col = col
                break

        name_col = None
        for col in df.columns:
            col_lower = col.lower()
            if "name" in col_lower and "access" not in col_lower:
                name_col = col
                break

        access_category_cols = [
            col for col in df.columns
            if col not in [role_col, department_col, supervisor_col]
            and "employee" not in col.lower()
            and "name" not in col.lower()
            and self._should_include_category(col)
        ]

        rows = []

        for _, row in df.iterrows():
            job_title = row.get(role_col)
            department = row.get(department_col) if department_col else None
            supervisor = row.get(supervisor_col) if supervisor_col else None
            employee_name = row.get(name_col) if name_col else None

            if pd.isna(job_title):
                continue

            for category in access_category_cols:
                cell_value = row.get(category)

                if pd.isna(cell_value):
                    continue

                access_items = self._split_access_items(cell_value)

                for access_name in access_items:
                    rows.append(
                        {
                            "EmployeeType": employee_type,
                            "JobTitle": str(job_title).strip(),
                            "Department": str(department).strip() if pd.notna(department) else None,
                            "Supervisor": str(supervisor).strip() if pd.notna(supervisor) else None,
                            "ReferenceEmployeeName": (
                                str(employee_name).strip() if pd.notna(employee_name) else None
                            ),
                            "AccessCategory": category,
                            "AccessName": access_name,
                            "AccessNameClean": access_name.lower().strip(),
                            "SourceFile": file_name,
                        }
                    )

        return pd.DataFrame(rows)

    @classmethod
    def _normalize_category(cls, value: str) -> str:
        return " ".join(str(value).lower().split())

    @classmethod
    def _should_include_category(cls, category: str) -> bool:
        normalized = cls._normalize_category(category)

        if normalized in cls.EXCLUDED_ACCESS_CATEGORIES:
            return False

        if normalized.startswith("fsy orion"):
            return False

        if normalized.startswith("teamwork"):
            return False

        return True

    def _split_access_items(self, value) -> list[str]:
        text = str(value).strip()

        if not text:
            return []

        # normalize separators
        text = text.replace(";", "\n")

        parts = text.split("\n")

        cleaned = []

        for part in parts:
            item = part.strip()

            if not item:
                continue

            # remove junk
            if item.lower() in ["x", "n/a", "na", "none"]:
                continue


            cleaned.append(item)

        return cleaned
