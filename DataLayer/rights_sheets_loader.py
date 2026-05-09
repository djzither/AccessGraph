from __future__ import annotations

from pathlib import Path
import re
import warnings

import pandas as pd

from DataLayer.loader import DataLoader
from DataLayer.access_exclusions import (
    count_excluded_reference_rows,
    filter_reference_df,
    is_excluded_access,
    is_excluded_access_category,
)


class RightsSheetsLoader:
    """Load employee access reference Excel files into a normalized table."""

    OUTPUT_COLUMNS = [
        "EmployeeType",
        "JobTitle",
        "Department",
        "Supervisor",
        "AccessCategory",
        "AccessName",
        "SourceFile",
    ]

    STUDENT_ACCESS_COLS = [
        "HCEB Doors",
        "AD Rights",
        "Email Groups",
        "Cvent",
        "Orion",
        "Orion Test",
        "FSY Orion (FSY Manager)",
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

    FILE_CONFIG = {
        "student_employee_access": {
            "sheet": "Data Base",
            "employee_type": "Student",
            "access_columns": STUDENT_ACCESS_COLS,
        },
        "full_time_employee_access": {
            "sheet": "Full Time Employees Data Base",
            "employee_type": "Full Time",
            "access_columns": FULLTIME_ACCESS_COLS,
        },
    }

    COLUMN_ALIASES = {
        "JobTitle": ["Job Title", "Title", "JobTitle"],
        "Department": ["Department", "Dept"],
        "Supervisor": [
            "Supervisor",
            "Supervisors",
            "Manager",
            "Direct Report / Supervisor",
        ],
        "AccessCategory": ["Access Category", "Category"],
        "AccessName": ["Access Name", "Access", "Permission", "Group", "Rights"],
    }

    IGNORED_WIDE_COLUMNS = {
        "area",
        "employee",
        "employee name",
        "reference employee",
        "name",
        "request participant",
        "type",
        "premium email",
        "ring central",
    }

    EXCLUDED_JOB_TITLES = {
        "ce fsy us counselor",
        "ce fsy us coordinator",
        "ce fsy us assistant coordinator",
        "ce fsy us wellness coordinator",
    }

    DEPARTMENT_ALIASES = {
        "fs": "financial services",
        "fsy": "especially for youth",
        "mms": "multimedia services",
        "it": "information technology",
    }

    def __init__(self, raw_path: str | Path):
        self.loader = DataLoader(base_path=str(raw_path))
        self.validation: list[dict] = []

    def load_reference_sheets(self) -> pd.DataFrame:
        frames = [
            self._load_and_normalize("full_time_employee_access.xlsx", "Full Time"),
            self._load_and_normalize("student_employee_access.xlsx", "Student"),
        ]
        combined = pd.concat(frames, ignore_index=True)
        excluded = count_excluded_reference_rows(combined)
        combined = filter_reference_df(self._finalize_output(combined))

        print(f"[reference] Final combined row count: {len(combined)}")
        print(f"[reference] Excluded CRM rows: {excluded}")
        if not combined.empty:
            print("[reference] Row count by source file:")
            print(combined["SourceFile"].value_counts().to_string())
            print("[reference] Sample parsed rows:")
            print(combined.head(5).to_string(index=False))
        return combined

    def _load_and_normalize(self, file_name: str, employee_type: str | None = None) -> pd.DataFrame:
        path = self.loader._get_path(file_name)
        config = self.FILE_CONFIG.get(Path(file_name).stem, {})
        employee_type = employee_type or config.get("employee_type") or self._employee_type_from_name(file_name)
        preferred_sheet = config.get("sheet")
        expected_access_cols = list(config.get("access_columns", []))

        raw, sheet_name, header_row = self._read_excel_with_detected_header(path, preferred_sheet)
        raw = self._normalize_column_headers(raw)
        raw = raw.dropna(how="all").reset_index(drop=True)

        resolved = {
            canonical: self._resolve_column(raw.columns, aliases)
            for canonical, aliases in self.COLUMN_ALIASES.items()
        }
        access_name_col = resolved["AccessName"]
        access_category_col = resolved["AccessCategory"]
        wide_access_cols = self._resolve_wide_access_columns(raw, expected_access_cols, resolved)

        missing = [
            col for col in ["JobTitle", "Department", "Supervisor"]
            if resolved.get(col) is None
        ]
        if access_name_col is None and not wide_access_cols:
            missing.append("AccessName or access category columns")

        unmapped = self._unmapped_columns(raw.columns, resolved, wide_access_cols)

        rows = []
        access_cells = 0
        empty_access_cells = 0
        excluded_rows = 0
        for _, row in raw.iterrows():
            job_title = self._normalize_job_title(row.get(resolved["JobTitle"])) if resolved["JobTitle"] else None
            department = self._normalize_department(row.get(resolved["Department"])) if resolved["Department"] else None
            supervisor = self._clean_optional_text(row.get(resolved["Supervisor"])) if resolved["Supervisor"] else None

            if self._is_note_or_title_row(job_title, department, supervisor, row):
                continue
            if job_title and self._normalize_title_for_exclusion(job_title) in self.EXCLUDED_JOB_TITLES:
                continue

            if access_name_col is not None:
                access_cells += 1
                names = self._split_access_items(row.get(access_name_col))
                if not names:
                    empty_access_cells += 1
                category = (
                    self._clean_optional_text(row.get(access_category_col))
                    if access_category_col else None
                )
                for access_name in names:
                    if is_excluded_access(category, access_name):
                        excluded_rows += 1
                        continue
                    rows.append(
                        self._record(
                            employee_type,
                            job_title,
                            department,
                            supervisor,
                            category,
                            access_name,
                            file_name,
                        )
                    )
            else:
                for category in wide_access_cols:
                    if is_excluded_access_category(category):
                        excluded_rows += len(self._split_access_items(row.get(category)))
                        continue
                    access_cells += 1
                    names = self._split_access_items(row.get(category))
                    if not names:
                        empty_access_cells += 1
                    for access_name in names:
                        if is_excluded_access(category, access_name):
                            excluded_rows += 1
                            continue
                        rows.append(
                            self._record(
                                employee_type,
                                job_title,
                                department,
                                supervisor,
                                category,
                                access_name,
                                file_name,
                            )
                        )

        result = filter_reference_df(self._finalize_output(pd.DataFrame(rows)))
        empty_ratio = empty_access_cells / access_cells if access_cells else 1.0
        self._record_validation(
            file_name=file_name,
            sheet_name=sheet_name,
            header_row=header_row,
            row_count=len(result),
            missing=missing,
            unmapped=unmapped,
            empty_ratio=empty_ratio,
            excluded_rows=excluded_rows,
            sample=result.head(5),
        )
        return result

    def _read_excel_with_detected_header(
        self,
        path: Path,
        preferred_sheet: str | int | None,
    ) -> tuple[pd.DataFrame, str, int]:
        excel = pd.ExcelFile(path, engine="openpyxl")
        if isinstance(preferred_sheet, str) and preferred_sheet in excel.sheet_names:
            sheet_name = preferred_sheet
        elif isinstance(preferred_sheet, int):
            sheet_name = excel.sheet_names[preferred_sheet]
        else:
            if preferred_sheet is not None:
                warnings.warn(
                    f"[{path.name}] Sheet {preferred_sheet!r} not found; using first sheet."
                )
            sheet_name = excel.sheet_names[0]

        raw = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=None,
            dtype=object,
            engine="openpyxl",
        )
        raw = raw.dropna(axis=1, how="all")
        header_row = self._detect_header_row(raw)
        headers = raw.iloc[header_row].tolist()
        data = raw.iloc[header_row + 1:].copy()
        data.columns = headers
        return data, sheet_name, header_row

    def _detect_header_row(self, raw: pd.DataFrame) -> int:
        best_row = 0
        best_score = -1
        max_scan = min(len(raw), 30)
        for index in range(max_scan):
            values = [self._header_key(value) for value in raw.iloc[index].tolist()]
            values = [value for value in values if value]
            value_set = set(values)

            score = 0
            for aliases in self.COLUMN_ALIASES.values():
                alias_keys = {self._header_key(alias) for alias in aliases}
                if value_set & alias_keys:
                    score += 3

            expected_categories = {
                self._header_key(col)
                for col in self.STUDENT_ACCESS_COLS + self.FULLTIME_ACCESS_COLS
            }
            score += 2 * len(value_set & expected_categories)

            if score > best_score:
                best_score = score
                best_row = index

        if best_score <= 0:
            raise ValueError("Could not detect a header row in access reference workbook.")
        return best_row

    def _normalize_column_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        seen: dict[str, int] = {}
        columns = []
        for col in df.columns:
            normalized = self._clean_header(col)
            if not normalized:
                normalized = "Unnamed"
            count = seen.get(normalized, 0)
            seen[normalized] = count + 1
            columns.append(normalized if count == 0 else f"{normalized}.{count}")
        df.columns = columns
        return df

    def _resolve_wide_access_columns(
        self,
        df: pd.DataFrame,
        expected_access_cols: list[str],
        resolved: dict[str, str | None],
    ) -> list[str]:
        if resolved.get("AccessName") is not None:
            return []

        by_key = {self._header_key(col): col for col in df.columns}
        matched = [
            by_key[self._header_key(col)]
            for col in expected_access_cols
            if self._header_key(col) in by_key
        ]
        if matched:
            return matched

        identity_cols = {col for col in resolved.values() if col}
        candidates = []
        for col in df.columns:
            key = self._header_key(col)
            if col in identity_cols or key in self.IGNORED_WIDE_COLUMNS:
                continue
            non_empty_items = df[col].apply(lambda value: len(self._split_access_items(value)) > 0)
            if non_empty_items.any():
                candidates.append(col)
        return candidates

    def _resolve_column(self, columns, aliases: list[str]) -> str | None:
        by_key = {self._header_key(col): col for col in columns}
        for alias in aliases:
            match = by_key.get(self._header_key(alias))
            if match is not None:
                return match
        return None

    def _unmapped_columns(
        self,
        columns,
        resolved: dict[str, str | None],
        wide_access_cols: list[str],
    ) -> list[str]:
        mapped = {col for col in resolved.values() if col}
        mapped.update(wide_access_cols)
        return [col for col in columns if col not in mapped]

    def _record_validation(
        self,
        *,
        file_name: str,
        sheet_name: str,
        header_row: int,
        row_count: int,
        missing: list[str],
        unmapped: list[str],
        empty_ratio: float,
        excluded_rows: int,
        sample: pd.DataFrame,
    ) -> None:
        info = {
            "source_file": file_name,
            "sheet_name": sheet_name,
            "header_row": header_row,
            "row_count": row_count,
            "missing_columns": missing,
            "unmapped_columns": unmapped,
            "empty_access_ratio": empty_ratio,
            "excluded_crm_rows": excluded_rows,
        }
        self.validation.append(info)

        print(f"[{file_name}] Sheet: {sheet_name}")
        print(f"[{file_name}] Detected header row: {header_row}")
        print(f"[{file_name}] Parsed row count: {row_count}")
        print(f"[{file_name}] Excluded CRM rows: {excluded_rows}")
        print(f"[{file_name}] Missing/unmapped columns: missing={missing}; unmapped={unmapped}")
        if sample.empty:
            print(f"[{file_name}] Sample parsed rows: <none>")
        else:
            print(f"[{file_name}] Sample parsed rows:")
            print(sample.to_string(index=False))

        if empty_ratio > 0.5:
            message = (
                f"[{file_name}] AccessName is mostly empty "
                f"({empty_ratio:.1%} of access cells produced no value)."
            )
            if row_count == 0:
                raise ValueError(message)
            warnings.warn(message)

    def _record(
        self,
        employee_type: str,
        job_title: str | None,
        department: str | None,
        supervisor: str | None,
        access_category: str | None,
        access_name: str,
        source_file: str,
    ) -> dict:
        return {
            "EmployeeType": employee_type,
            "JobTitle": job_title,
            "Department": department,
            "Supervisor": supervisor,
            "AccessCategory": access_category,
            "AccessName": access_name,
            "SourceFile": source_file,
        }

    def _finalize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[self.OUTPUT_COLUMNS]
        if df.empty:
            return df

        for col in self.OUTPUT_COLUMNS:
            df[col] = df[col].apply(self._clean_optional_text)
        df = df[df["AccessName"].fillna("").str.strip() != ""].copy()
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    def _split_access_items(self, value) -> list[str]:
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text:
            return []
        text = text.replace(";", "\n").replace(",", "\n")
        cleaned = []
        for part in text.split("\n"):
            item = self._clean_access_token(part)
            if not item:
                continue
            for token in self._expand_access_token(item):
                if token.lower() in {"x", "n/a", "na", "none", "null"}:
                    continue
                if self._is_likely_person_entry(token):
                    continue
                cleaned.append(token)
        return cleaned

    @classmethod
    def _expand_access_token(cls, item: str) -> list[str]:
        pieces = [piece.strip() for piece in item.split() if piece.strip()]
        if len(pieces) > 1 and all(cls._looks_like_group_atom(piece) for piece in pieces):
            return pieces
        return [item]

    @classmethod
    def _looks_like_group_atom(cls, item: str) -> bool:
        value = item.strip().strip(",;")
        lower = value.lower()
        return (
            "." in lower
            or "_" in lower
            or "\\" in lower
            or lower.startswith("eag-")
            or lower.startswith("dce-")
        )

    @classmethod
    def _is_likely_person_entry(cls, item: str) -> bool:
        value = str(item).strip().strip(",")
        lower = value.lower()
        if lower in {"email", "vpn"}:
            return False
        if " " not in lower and any(separator in lower for separator in [".", "-", "_", "\\"]):
            return False
        if re.fullmatch(r"[a-z]+(?:[ '-][a-z]+)+", lower):
            return True
        if " " not in lower and re.fullmatch(r"[a-z]{5,}\d{0,4}", lower):
            return True
        return False

    @classmethod
    def _clean_access_token(cls, item: object) -> str:
        if pd.isna(item):
            return ""
        token = re.sub(r"\s+", " ", str(item)).strip()
        token = re.sub(r"^[QIR]\s*\((.*?)\)\s*$", r"\1", token, flags=re.IGNORECASE)
        token = re.sub(r"\(.*?\)$", "", token).strip()
        low = token.lower()
        noise = {
            "view only",
            "registrar student",
            "student employee and assistant",
            "data entry and materials",
            "we order",
        }
        if low in noise:
            return ""
        if re.search(r"\bext\b|\d{3}[- ]?\d{4}", low):
            return ""
        if token.startswith("http://") or token.startswith("https://"):
            return ""
        return token

    @classmethod
    def _normalize_department(cls, value) -> str | None:
        text = cls._clean_optional_text(value)
        if text is None:
            return None
        key = " ".join(text.lower().split())
        return cls.DEPARTMENT_ALIASES.get(key, text)

    @classmethod
    def _normalize_job_title(cls, value) -> str | None:
        text = cls._clean_optional_text(value)
        if text is None:
            return None
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?combine.*?\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    @classmethod
    def _normalize_title_for_exclusion(cls, value: str) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"\.", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("councilor", "counselor")
        text = text.replace("coordiator", "coordinator")
        text = text.replace("co-ordinator", "coordinator")
        return text

    @classmethod
    def _clean_optional_text(cls, value) -> str | None:
        if pd.isna(value):
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        if not text or text.lower() in {"nan", "none"}:
            return None
        return text

    @classmethod
    def _clean_header(cls, value) -> str:
        if pd.isna(value):
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    @classmethod
    def _header_key(cls, value) -> str:
        return re.sub(r"[^a-z0-9]+", "", cls._clean_header(value).lower())

    @classmethod
    def _employee_type_from_name(cls, file_name: str) -> str:
        lower = file_name.lower()
        if "student" in lower:
            return "Student"
        if "full" in lower:
            return "Full Time"
        return "Unknown"

    def _is_note_or_title_row(
        self,
        job_title: str | None,
        department: str | None,
        supervisor: str | None,
        row: pd.Series,
    ) -> bool:
        if job_title:
            return False
        if department or supervisor:
            return False
        values = [
            self._clean_optional_text(value)
            for value in row.tolist()
        ]
        values = [value for value in values if value]
        if not values:
            return True
        joined = " ".join(values).lower()
        note_words = ("note", "notes", "title", "access reference", "database")
        return len(values) <= 2 and any(word in joined for word in note_words)
