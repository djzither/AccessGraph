import pandas as pd
from collections import Counter
import re

from DeterministicLayer.permission_filter import PermissionFilter
from DeterministicLayer.permission_matrix import PermissionMatrixBuilder
from DeterministicLayer.title_embed_matcher import TitleEmbedMatcher
from MLLayer.recommender import MLRecommender


class AccessRecommendationEngine:
    # Minimum cohort sizes for reliable frequency signals.
    MIN_COHORT_SIZE = 5       # below this we widen the cohort
    MIN_RELIABLE_COHORT = 10  # below this we down-weight AD/ML and up-weight reference
    AD_SMOOTHING_FACTOR = 5   # Laplace pseudo-count added to denominator

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

    def __init__(
        self,
        min_confidence: float = 0.5,
        title_matcher: TitleEmbedMatcher | None = None,
    ):
        self.matrix_builder = PermissionMatrixBuilder(min_confidence=min_confidence)
        self.permission_filter = PermissionFilter()
        self.title_matcher = title_matcher

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
            employee_type=employee_type,
            copy_from_netid=copy_from_netid,
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
        merged["EmployeeTypeClean"] = str(employee_type).lower().strip()
        merged["IsFSYRole"] = self._is_fsy_role(title=title, department=department)

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

        if matched.empty:
            employee_ref = ref[ref["EmployeeTypeClean"] == employee_type_clean].copy()
            if not employee_ref.empty:
                candidate_titles = employee_ref["JobTitle"].dropna().astype(str).unique().tolist()
                embed_matcher = self.title_matcher
                if embed_matcher is None:
                    try:
                        embed_matcher = TitleEmbedMatcher()
                    except Exception:
                        embed_matcher = None

                if embed_matcher is not None:
                    best_title, _ = embed_matcher.best_match(title, candidate_titles)
                    if best_title is not None:
                        matched = employee_ref[
                            employee_ref["JobTitle"].astype(str) == best_title
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
            # Laplace smoothing: add AD_SMOOTHING_FACTOR pseudo-observations so
            # a group seen by 2/3 users in a tiny cohort doesn't score 0.67 — it
            # scores 2/(3+5)=0.25, which better reflects how unreliable that is.
            smoothed = round(count / (total_users + self.AD_SMOOTHING_FACTOR), 3)

            if smoothed < self.matrix_builder.min_confidence:
                continue

            rows.append({
                "GroupName": group_name,
                "ADConfidence": smoothed,
                "ADRawConfidence": round(count / total_users, 3),
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
        employee_type_clean = str(row.get("EmployeeTypeClean", "")).lower().strip()
        is_fsy_role = bool(row.get("IsFSYRole", False))

        if is_fsy_role:
            reference_weight = 0.65
            ad_weights = (0.20, 0.15, 0.10)
            ml_weights = (0.10, 0.05, 0.03)
            copy_weight = 0.05
        elif employee_type_clean == "full time":
            reference_weight = 0.30
            ad_weights = (0.30, 0.20, 0.10)
            ml_weights = (0.25, 0.20, 0.10)
            copy_weight = 0.20
        else:
            # Students: trust the reference access list more.
            reference_weight = 0.60
            ad_weights = (0.20, 0.10, 0.05)
            ml_weights = (0.10, 0.05, 0.03)
            copy_weight = 0.05

        # Scale AD/ML weight by cohort reliability; boost reference when cohort is small.
        cohort_size = int(row.get("TotalUsersInRole", 0))
        cohort_factor = min(1.0, cohort_size / self.MIN_RELIABLE_COHORT)
        ad_weights = tuple(w * cohort_factor for w in ad_weights)
        ml_weights = tuple(w * cohort_factor for w in ml_weights)
        if cohort_factor < 1.0 and not is_fsy_role:
            reference_weight = min(0.80, reference_weight + (1 - cohort_factor) * 0.20)

        if row["InReferenceSheet"]:
            score += reference_weight

        if row["ADConfidence"] >= 0.8:
            score += ad_weights[0]
        elif row["ADConfidence"] >= 0.6:
            score += ad_weights[1]
        elif row["ADConfidence"] >= 0.4:
            score += ad_weights[2]

        if row["MLConfidence"] >= 0.8:
            score += ml_weights[0]
        elif row["MLConfidence"] >= 0.6:
            score += ml_weights[1]
        elif row["MLConfidence"] >= 0.4:
            score += ml_weights[2]

        if row["CopyFromUserHasIt"]:
            score += copy_weight

        return round(min(score, 1.0), 3)

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
        employee_type: str,
        copy_from_netid: str | None = None,
    ) -> pd.DataFrame:
        """
        Build the best comparison cohort using a 4-level fallback strategy.

        Level 1 — exact (title + dept): most precise; used when ≥ MIN_COHORT_SIZE.
        Level 2 — dept-only (+ copy-from fallback): wider pool; used when ≥ MIN_COHORT_SIZE.
        Level 3 — title cross-dept: catches titles that span departments.
        Level 4 — largest non-empty candidate: last resort to avoid empty cohort.

        Each level that clears the size bar is further refined by reference-sheet
        overlap so that users who "look like" this role float to the top.
        """
        users = users_df.copy()
        users["TitleClean"] = users["Title"].apply(self._normalize_role_text)
        users["DepartmentClean"] = users["Department"].apply(self._normalize_role_text)

        title_clean = self._normalize_role_text(title)
        department_clean = self._normalize_role_text(department)

        reference_group_names: set = set()
        if not reference_recs.empty and "GroupNameClean" in reference_recs.columns:
            reference_group_names = set(
                reference_recs["GroupNameClean"].dropna().astype(str)
            )

        # Level 1: exact title + department
        exact_cohort = users[
            (users["TitleClean"] == title_clean)
            & (users["DepartmentClean"] == department_clean)
        ].copy()
        if len(exact_cohort) >= self.MIN_COHORT_SIZE:
            return self._refine_by_reference_overlap(
                exact_cohort, reference_group_names, title_clean
            )

        # Level 2: department-only (copy-from fallback when dept not found)
        same_department = users[users["DepartmentClean"] == department_clean].copy()
        if same_department.empty and str(employee_type).lower().strip() == "full time" and copy_from_netid is not None:
            copy_user = users[users["SamAccountName"] == copy_from_netid]
            if not copy_user.empty:
                copy_dept_clean = copy_user.iloc[0]["DepartmentClean"]
                same_department = users[users["DepartmentClean"] == copy_dept_clean].copy()

        if len(same_department) >= self.MIN_COHORT_SIZE:
            return self._refine_by_reference_overlap(
                same_department, reference_group_names, title_clean
            )

        # Level 3: title cross-department
        cross_dept = users[users["TitleClean"] == title_clean].copy()
        if len(cross_dept) >= self.MIN_COHORT_SIZE:
            return cross_dept

        # Level 4: return the largest non-empty candidate we found
        candidates = [c for c in [exact_cohort, same_department, cross_dept] if not c.empty]
        if candidates:
            return max(candidates, key=len)
        return exact_cohort  # empty DataFrame

    def _refine_by_reference_overlap(
        self,
        cohort: pd.DataFrame,
        reference_group_names: set,
        title_clean: str,
    ) -> pd.DataFrame:
        """
        Within a cohort, prefer users whose current access overlaps the
        reference sheet — they are most similar to the target role.
        Falls back gracefully when no overlap is found.
        """
        if reference_group_names:
            cohort = cohort.copy()
            cohort["_overlap"] = cohort["GroupsList"].apply(
                lambda groups: sum(
                    1 for g in groups
                    if self._normalize_group_name(g) in reference_group_names
                )
            )
            for min_overlap in (2, 1):
                refined = cohort[cohort["_overlap"] >= min_overlap].drop(columns=["_overlap"])
                if not refined.empty:
                    return refined
            cohort = cohort.drop(columns=["_overlap"])

        # No reference signal — fall back to exact title match, then full cohort
        exact = cohort[cohort["TitleClean"] == title_clean] if "TitleClean" in cohort.columns else pd.DataFrame()
        if not exact.empty:
            return exact
        return cohort

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

        text = re.sub(r"[\s._-]+", "", text)
        return text

    @classmethod
    def _is_fsy_role(cls, title: str, department: str) -> bool:
        title_clean = cls._normalize_role_text(title)
        department_clean = cls._normalize_role_text(department)
        return "fsy" in title_clean or "fsy" in department_clean
