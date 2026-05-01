import pandas as pd
from collections import Counter

from DeterministicLayer.permission_filter import PermissionFilter
from DeterministicLayer.permission_matrix import PermissionMatrixBuilder
from MLLayer.recommender import MLRecommender


class AccessRecommendationEngine:
    ROLE_ALIASES = {
        (
            "academic outreach and sales rep",
            "ce academic outreach and sales",
        ): {
            (
                "academic outreach sales rep",
                "marketing and customer support",
            ),
        },
        (
            "computing specialist",
            "ce it help desk",
        ): {
            (
                "computing specialist",
                "it",
            ),
        },
    }

    def __init__(self, min_confidence: float = 0.5):
        self.matrix_builder = PermissionMatrixBuilder(min_confidence=min_confidence)
        self.permission_filter = PermissionFilter()

    def recommend_for_hire(
        self,
        users_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        title: str,
        department: str,
        employee_type: str,
        supervisor: str | None = None,
        copy_from_netid: str | None = None,
        new_hire_netid: str | None = None,
    ) -> pd.DataFrame:

        reference_recs = self._get_reference_recommendations(
            reference_df=reference_df,
            title=title,
            department=department,
            employee_type=employee_type,
            supervisor=supervisor,
            users_df=users_df,
            copy_from_netid=copy_from_netid,
        )

        comparison_cohort = self._select_ad_comparison_cohort(
            users_df=users_df,
            title=title,
            department=department,
            reference_recs=reference_recs,
        )

        ad_recs = self._get_ad_recommendations(
            comparison_cohort=comparison_cohort,
        )

        ml_recs = self._get_ml_recommendations(
            users_df=users_df,
            new_hire_netid=new_hire_netid,
            department=department,
            comparison_cohort=comparison_cohort,
        )

        copy_from_recs = self._get_copy_from_recommendations(
            users_df=users_df,
            copy_from_netid=copy_from_netid,
        )

        merged = self._merge_all_sources(
            reference_recs=reference_recs,
            ad_recs=ad_recs,
            ml_recs=ml_recs,
            copy_from_recs=copy_from_recs,
        )

        merged = self.permission_filter.filter_recommendations(merged)

        merged["FinalScore"] = merged.apply(self._score_row, axis=1)
        merged["FinalDecision"] = merged.apply(self._final_decision, axis=1)
        merged["Reason"] = merged.apply(self._reason, axis=1)

        return merged.sort_values(
            by=["FinalScore", "GroupName"],
            ascending=[False, True],
        )

    def _get_reference_recommendations(
        self,
        reference_df: pd.DataFrame,
        title: str,
        department: str,
        employee_type: str,
        supervisor: str | None,
        users_df: pd.DataFrame,
        copy_from_netid: str | None,
    ) -> pd.DataFrame:

        ref = reference_df.copy()

        ref["JobTitleClean"] = ref["JobTitle"].apply(self._normalize_role_text)
        ref["DepartmentClean"] = ref["Department"].apply(self._normalize_role_text)
        ref["EmployeeTypeClean"] = ref["EmployeeType"].astype(str).str.lower().str.strip()
        ref["SupervisorClean"] = ref["Supervisor"].astype(str).str.lower().str.strip()
        ref["AccessNameClean"] = ref["AccessName"].apply(self._normalize_group_name)
        if "ReferenceEmployeeName" in ref.columns:
            ref["ReferenceEmployeeNameClean"] = (
                ref["ReferenceEmployeeName"].astype(str).str.lower().str.strip()
            )
        else:
            ref["ReferenceEmployeeNameClean"] = ""

        role_candidates = self._role_candidates(title=title, department=department)
        employee_type_clean = str(employee_type).lower().strip()

        matched = ref[
            ref.apply(
                lambda row: (row["JobTitleClean"], row["DepartmentClean"]) in role_candidates,
                axis=1,
            )
            & (ref["EmployeeTypeClean"] == employee_type_clean)
        ].copy()

        if employee_type_clean == "full time" and supervisor is not None:
            supervisor_clean = str(supervisor).lower().strip()

            supervisor_matches = matched[
                matched["SupervisorClean"] == supervisor_clean
            ]

            if not supervisor_matches.empty:
                matched = supervisor_matches

        if employee_type_clean == "full time" and matched.empty and copy_from_netid is not None:
            copy_user = users_df[users_df["SamAccountName"] == copy_from_netid]
            if not copy_user.empty and "DisplayName" in copy_user.columns:
                copy_from_name_clean = (
                    str(copy_user.iloc[0]["DisplayName"]).lower().strip()
                )
                if copy_from_name_clean and copy_from_name_clean != "nan":
                    name_matches = ref[
                        (ref["EmployeeTypeClean"] == employee_type_clean)
                        & (ref["ReferenceEmployeeNameClean"] == copy_from_name_clean)
                    ]
                    if not name_matches.empty:
                        matched = name_matches.copy()

        if matched.empty:
            return pd.DataFrame(columns=[
                "GroupNameClean",
                "GroupName",
                "InReferenceSheet",
                "ReferenceCategories",
            ])

        grouped = (
            matched.groupby("AccessNameClean", as_index=False)
            .agg(
                GroupName=("AccessName", "first"),
                ReferenceCategories=(
                    "AccessCategory",
                    lambda x: ", ".join(sorted(set(x.astype(str))))
                ),
            )
        )

        grouped = grouped.rename(columns={
            "AccessNameClean": "GroupNameClean"
        })

        grouped["InReferenceSheet"] = True

        return grouped
    def _get_ad_recommendations(
        self,
        comparison_cohort: pd.DataFrame,
    ) -> pd.DataFrame:
        if comparison_cohort.empty:
            return pd.DataFrame(columns=[
                "GroupName",
                "ADConfidence",
                "UserCountWithGroup",
                "TotalUsersInRole",
            ])

        total_users = len(comparison_cohort)
        counter = Counter()

        for groups in comparison_cohort["GroupsList"]:
            counter.update(groups)

        rows = []

        for group_name, count in counter.items():
            confidence = round(count / total_users, 3)

            if confidence < self.matrix_builder.min_confidence:
                continue

            rows.append({
                "GroupName": group_name,
                "ADConfidence": confidence,
                "UserCountWithGroup": count,
                "TotalUsersInRole": total_users,
            })

        recs = pd.DataFrame(rows)

        if recs.empty:
            return pd.DataFrame(columns=[
                "GroupName",
                "ADConfidence",
                "UserCountWithGroup",
                "TotalUsersInRole",
            ])

        return recs.sort_values(
            by=["ADConfidence", "UserCountWithGroup", "GroupName"],
            ascending=[False, False, True],
        )[
            [
                "GroupName",
                "ADConfidence",
                "UserCountWithGroup",
                "TotalUsersInRole",
            ]
        ]

    def _get_ml_recommendations(
        self,
        users_df: pd.DataFrame,
        new_hire_netid: str | None,
        department: str,
        comparison_cohort: pd.DataFrame,
    ) -> pd.DataFrame:

        ml = MLRecommender(users_df)

        if new_hire_netid is not None:
            recs = ml.recommend_for_user(
                sam_account_name=new_hire_netid,
                department=department,
                top_n_users=5,
                min_support=2,
                include_supervisors=False,
            )
            ml_mode = "target_user"
            ml_anchor_netid = new_hire_netid
        else:
            recs = ml.recommend_for_peer_cohort(
                cohort_df=comparison_cohort,
                min_support=2,
            )
            ml_mode = "peer_aggregate"
            ml_anchor_netid = ""

        if recs.empty:
            return pd.DataFrame(columns=[
                "GroupName",
                "MLConfidence",
                "MLSupportCount",
                "MLComparedUsers",
                "NearestUsers",
                "MLMode",
                "MLAnchorNetID",
            ])

        recs["MLMode"] = recs.get("MLMode", ml_mode)
        recs["MLAnchorNetID"] = ml_anchor_netid

        return recs[
            [
                "GroupName",
                "MLConfidence",
                "MLSupportCount",
                "MLComparedUsers",
                "NearestUsers",
                "MLMode",
                "MLAnchorNetID",
            ]
        ]

    def _get_copy_from_recommendations(
        self,
        users_df: pd.DataFrame,
        copy_from_netid: str | None,
    ) -> pd.DataFrame:

        if copy_from_netid is None:
            return pd.DataFrame(columns=[
                "GroupName",
                "CopyFromUserHasIt",
                "CopyFromNetID",
            ])

        user = users_df[users_df["SamAccountName"] == copy_from_netid]

        if user.empty:
            return pd.DataFrame(columns=[
                "GroupName",
                "CopyFromUserHasIt",
                "CopyFromNetID",
            ])

        rights = user.iloc[0]["GroupsList"]

        rows = []

        for right in rights:
            rows.append({
                "GroupName": right,
                "CopyFromUserHasIt": True,
                "CopyFromNetID": copy_from_netid,
            })

        return pd.DataFrame(rows)

    def _merge_all_sources(
            self,
            reference_recs: pd.DataFrame,
            ad_recs: pd.DataFrame,
            ml_recs: pd.DataFrame,
            copy_from_recs: pd.DataFrame,
    ) -> pd.DataFrame:

        def add_group_clean(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()

            if "GroupNameClean" not in df.columns:
                df["GroupNameClean"] = ""

            if "GroupName" not in df.columns:
                df["GroupName"] = ""

            if not df.empty:
                df["GroupNameClean"] = df["GroupName"].apply(self._normalize_group_name)

            return df

        reference_recs = add_group_clean(reference_recs)
        ad_recs = add_group_clean(ad_recs)
        ml_recs = add_group_clean(ml_recs)
        copy_from_recs = add_group_clean(copy_from_recs)



        all_group_names = set()

        for df in [reference_recs, ad_recs, ml_recs, copy_from_recs]:
            if not df.empty and "GroupNameClean" in df.columns:
                all_group_names.update(df["GroupNameClean"].dropna().astype(str))

        base = pd.DataFrame({"GroupNameClean": sorted(all_group_names)})

        merged = base.merge(reference_recs, on="GroupNameClean", how="left")
        merged = merged.merge(ad_recs, on="GroupNameClean", how="left", suffixes=("", "_AD"))
        merged = merged.merge(ml_recs, on="GroupNameClean", how="left", suffixes=("", "_ML"))
        merged = merged.merge(copy_from_recs, on="GroupNameClean", how="left", suffixes=("", "_Copy"))

        merged["GroupName"] = (
            merged["GroupName"]
            .fillna(merged.get("GroupName_AD"))
            .fillna(merged.get("GroupName_ML"))
            .fillna(merged.get("GroupName_Copy"))
        )


        merged["InReferenceSheet"] = merged["InReferenceSheet"].fillna(False)
        merged["CopyFromUserHasIt"] = merged["CopyFromUserHasIt"].fillna(False)

        merged["ADConfidence"] = merged["ADConfidence"].fillna(0)
        merged["MLConfidence"] = merged["MLConfidence"].fillna(0)

        merged["UserCountWithGroup"] = merged["UserCountWithGroup"].fillna(0).astype(int)
        merged["TotalUsersInRole"] = merged["TotalUsersInRole"].fillna(0).astype(int)

        merged["MLSupportCount"] = merged["MLSupportCount"].fillna(0).astype(int)
        merged["MLComparedUsers"] = merged["MLComparedUsers"].fillna(0).astype(int)

        merged["ReferenceCategories"] = merged["ReferenceCategories"].fillna("")
        merged["NearestUsers"] = merged["NearestUsers"].fillna("")
        merged["MLMode"] = merged["MLMode"].fillna("")
        merged["MLAnchorNetID"] = merged["MLAnchorNetID"].fillna("")
        merged["CopyFromNetID"] = merged["CopyFromNetID"].fillna("")

        merged["Score"] = merged["ADConfidence"]

        return merged

    def _score_row(self, row) -> float:
        score = 0

        if row["InReferenceSheet"]:
            score += 0.45

        if row["ADConfidence"] >= 0.8:
            score += 0.30
        elif row["ADConfidence"] >= 0.6:
            score += 0.20
        elif row["ADConfidence"] >= 0.4:
            score += 0.10

        if row["MLConfidence"] >= 0.8:
            score += 0.15
        elif row["MLConfidence"] >= 0.6:
            score += 0.10
        elif row["MLConfidence"] >= 0.4:
            score += 0.05

        if row["CopyFromUserHasIt"]:
            score += 0.10

        return round(score, 3)

    def _final_decision(self, row) -> str:
        if row["RiskLevel"] == "High":
            return "Manual Review"

        score = row["FinalScore"]

        if score >= 0.80:
            return "Auto Assign"

        if score >= 0.65:
            return "Strong Recommend"

        if score >= 0.45:
            return "Suggest"

        if row["CopyFromUserHasIt"] and not row["InReferenceSheet"]:
            return "Possible Extra Access"

        return "Ignore"

    def _reason(self, row) -> str:
        reasons = []

        if row["InReferenceSheet"]:
            reasons.append("listed in the access reference sheet")

        if row["ADConfidence"] > 0:
            reasons.append(
                f"found in {row['UserCountWithGroup']}/{row['TotalUsersInRole']} matching AD users"
            )

        if row["MLConfidence"] > 0:
            if row.get("MLMode") == "peer_aggregate":
                ml_reason = (
                    f"found by peer-aggregate ML in {row['MLSupportCount']}/{row['MLComparedUsers']} role peers"
                )
            else:
                ml_reason = (
                    f"found by ML similarity in {row['MLSupportCount']}/{row['MLComparedUsers']} nearest users"
                )

                if row.get("MLAnchorNetID"):
                    ml_reason += f" to {row['MLAnchorNetID']}"

            reasons.append(ml_reason)

        if row["CopyFromUserHasIt"]:
            reasons.append(f"present on copy-from user {row['CopyFromNetID']}")

        if not reasons:
            return "No strong evidence found."

        return "Recommended because it is " + ", and ".join(reasons) + "."

    def _select_ad_comparison_cohort(
        self,
        users_df: pd.DataFrame,
        title: str,
        department: str,
        reference_recs: pd.DataFrame,
    ) -> pd.DataFrame:
        users = users_df.copy()

        users["TitleClean"] = users["Title"].apply(self._normalize_role_text)
        users["DepartmentClean"] = users["Department"].apply(self._normalize_role_text)

        title_clean = self._normalize_role_text(title)
        department_clean = self._normalize_role_text(department)

        same_department = users[users["DepartmentClean"] == department_clean].copy()

        if same_department.empty:
            return users[
                (users["TitleClean"] == title_clean)
                & (users["DepartmentClean"] == department_clean)
            ].copy()

        reference_group_names = set()

        if not reference_recs.empty and "GroupNameClean" in reference_recs.columns:
            reference_group_names = set(
                reference_recs["GroupNameClean"].dropna().astype(str)
            )

        if reference_group_names:
            same_department["ReferenceOverlapCount"] = same_department["GroupsList"].apply(
                lambda groups: sum(
                    1
                    for group in groups
                    if self._normalize_group_name(group) in reference_group_names
                )
            )

            overlap_cohort = same_department[
                same_department["ReferenceOverlapCount"] >= 2
            ].copy()

            if overlap_cohort.empty:
                overlap_cohort = same_department[
                    same_department["ReferenceOverlapCount"] >= 1
                ].copy()

            if not overlap_cohort.empty:
                return overlap_cohort

        exact_title_cohort = same_department[
            same_department["TitleClean"] == title_clean
        ].copy()

        if not exact_title_cohort.empty:
            return exact_title_cohort

        return same_department

    @classmethod
    def _normalize_role_text(cls, value) -> str:
        text = str(value).lower().strip()

        for old, new in [("&", " and "), (",", " "), ("/", " "), ("-", " ")]:
            text = text.replace(old, new)

        return " ".join(text.split())

    @classmethod
    def _role_candidates(cls, title: str, department: str) -> set[tuple[str, str]]:
        base_key = (
            cls._normalize_role_text(title),
            cls._normalize_role_text(department),
        )

        return {base_key, *cls.ROLE_ALIASES.get(base_key, set())}

    @classmethod
    def _normalize_group_name(cls, value) -> str:
        text = str(value).lower().strip()

        for prefix in ("m.", "i."):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break

        return " ".join(text.split())
