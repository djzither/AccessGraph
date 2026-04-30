import pandas as pd


class ReferenceMatcher:
    def __init__(self, reference_df: pd.DataFrame):
        self.reference_df = reference_df.copy()

        self.reference_df["AccessNameClean"] = (
            self.reference_df["AccessName"].astype(str).str.lower().str.strip()
        )

        self.reference_df["JobTitleClean"] = (
            self.reference_df["JobTitle"].astype(str).str.lower().str.strip()
        )

        self.reference_df["DepartmentClean"] = (
            self.reference_df["Department"].astype(str).str.lower().str.strip()
        )

        self.reference_df["EmployeeTypeClean"] = (
            self.reference_df["EmployeeType"].astype(str).str.lower().str.strip()
        )

        self.reference_df["SupervisorClean"] = (
            self.reference_df["Supervisor"].astype(str).str.lower().str.strip()
        )

    def match_recommendations(
        self,
        recommendations: pd.DataFrame,
        title: str,
        department: str,
        employee_type: str | None = None,
        supervisor: str | None = None,
    ) -> pd.DataFrame:
        recommendations = recommendations.copy()

        title_clean = str(title).lower().strip()
        department_clean = str(department).lower().strip()

        role_reference = self.reference_df[
            (self.reference_df["JobTitleClean"] == title_clean)
            & (self.reference_df["DepartmentClean"] == department_clean)
        ].copy()

        # Optional employee type filter
        if employee_type is not None:
            employee_type_clean = str(employee_type).lower().strip()

            employee_type_match = role_reference[
                role_reference["EmployeeTypeClean"] == employee_type_clean
            ]

            if not employee_type_match.empty:
                role_reference = employee_type_match

        # Only use supervisor narrowing for full-time employees
        if employee_type is not None and str(employee_type).lower().strip() == "full time":
            if supervisor is not None:
                supervisor_clean = str(supervisor).lower().strip()

                supervisor_match = role_reference[
                    role_reference["SupervisorClean"] == supervisor_clean
                ]

                # fallback if supervisor match is empty
                if not supervisor_match.empty:
                    role_reference = supervisor_match

        role_access_names = set(role_reference["AccessNameClean"])

        recommendations["GroupNameClean"] = (
            recommendations["GroupName"].astype(str).str.lower().str.strip()
        )

        recommendations["ReferenceSheetMatch"] = recommendations["GroupNameClean"].apply(
            lambda group: group in role_access_names
        )

        recommendations["ReferenceCategories"] = recommendations["GroupNameClean"].apply(
            lambda group: self._get_categories(group, role_reference)
        )

        recommendations["FinalDecision"] = recommendations.apply(
            self._final_decision,
            axis=1,
        )

        recommendations["Reason"] = recommendations.apply(
            self._reason,
            axis=1,
        )

        return recommendations.drop(columns=["GroupNameClean"])

    def _get_categories(self, group_name_clean: str, role_reference: pd.DataFrame) -> str:
        matches = role_reference[
            role_reference["AccessNameClean"] == group_name_clean
        ]

        if matches.empty:
            return ""

        categories = sorted(matches["AccessCategory"].dropna().astype(str).unique())
        return ", ".join(categories)

    def _final_decision(self, row) -> str:
        score = row["Score"]
        risk = row["RiskLevel"]
        sheet_match = row["ReferenceSheetMatch"]

        if risk == "High":
            return "Manual Review"

        if sheet_match and score >= 0.8:
            return "Strong Recommend"

        if sheet_match and score >= 0.6:
            return "Suggest"

        if sheet_match and score < 0.6:
            return "Manual Review"

        if score == 1.0:
            return "Auto Assign"

        if score >= 0.8:
            return "Strong Recommend"

        if score >= 0.6:
            return "Suggest"

        if score >= 0.4:
            return "Low Confidence"

        return "Ignore"

    def _reason(self, row) -> str:
        score = row["Score"]
        count = row["UserCountWithGroup"]
        total = row["TotalUsersInRole"]
        sheet_match = row["ReferenceSheetMatch"]

        if sheet_match:
            return f"Found in {count}/{total} similar users and listed in reference sheet."

        if score >= 0.6:
            return f"Found in {count}/{total} similar users but not listed in reference sheet."

        return f"Only found in {count}/{total} similar users and not listed in reference sheet."