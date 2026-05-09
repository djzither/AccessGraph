import pandas as pd
from collections import Counter
import re

from DataLayer.access_exclusions import (
    filter_group_list,
    filter_recommendations_df,
    filter_reference_df,
    filter_user_groups_df,
)
from DeterministicLayer.permission_filter import PermissionFilter
from DeterministicLayer.permission_matrix import PermissionMatrixBuilder
from DeterministicLayer.title_embed_matcher import TitleEmbedMatcher
from MLLayer.recommender import MLRecommender


class AccessRecommendationEngine:
    # Minimum cohort sizes for reliable frequency signals.
    MIN_COHORT_SIZE = 5       # below this we widen the cohort
    MIN_RELIABLE_COHORT = 10  # below this we down-weight AD/ML and up-weight reference
    AD_SMOOTHING_FACTOR = 5   # Laplace pseudo-count added to denominator
    MAX_COMMON_GROUP_RATE = 0.80
    MIN_ML_SIMILARITY = 0.25

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
        self._global_group_rates = {}

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
        users_df = filter_user_groups_df(users_df)
        reference_df = filter_reference_df(reference_df)

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
        merged["CohortSize"] = len(comparison_cohort)
        merged["CohortReliability"] = min(1.0, len(comparison_cohort) / self.MIN_RELIABLE_COHORT)

        merged = self.permission_filter.filter_recommendations(merged)
        merged = self._apply_signal_filters(merged)

        merged["FinalScore"] = merged.apply(self._score_row, axis=1)
        merged["FinalDecision"] = merged.apply(self._final_decision, axis=1)
        merged["Reason"] = self._build_reason_series(merged)

        merged = filter_recommendations_df(merged)

        return merged.sort_values(
            by=["FinalScore", "GroupName"],
            ascending=[False, True],
        )

    def _get_reference_recommendations(
        self,
        reference_df: pd.DataFrame,
        title: str,
        department: str,
        employee_type: str | None,
        supervisor: str | None,
        users_df: pd.DataFrame,
        copy_from_netid: str | None,
    ) -> pd.DataFrame:

        ref = reference_df.copy()
        ref = filter_reference_df(ref)

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
        employee_type_clean = (
            str(employee_type).lower().strip()
            if employee_type is not None and str(employee_type).strip()
            else None
        )

        matched = ref[
            ref.apply(
                lambda row: (row["JobTitleClean"], row["DepartmentClean"]) in role_candidates,
                axis=1,
            )
        ].copy()

        if employee_type_clean is not None:
            matched = matched[matched["EmployeeTypeClean"] == employee_type_clean].copy()

        if matched.empty:
            if employee_type_clean is not None:
                employee_ref = ref[ref["EmployeeTypeClean"] == employee_type_clean].copy()
            else:
                employee_ref = ref.copy()
            if not employee_ref.empty:
                # Prevent cross-department leakage (e.g., Finance rights on Help Desk).
                dept_clean = self._normalize_role_text(department)
                dept_scoped = employee_ref[
                    employee_ref["DepartmentClean"] == dept_clean
                ].copy()
                if not dept_scoped.empty:
                    employee_ref = dept_scoped

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
                "ReferenceTemplateCount",
                "AmbiguousReferenceTemplate",
            ])

        # Ambiguity detection (mirrors ReferenceMatcher semantics):
        # Multiple template variants under the same role can remain when
        # employee_type and/or supervisor context is missing. In that case,
        # reference-sheet matches are valid signals but should be trusted less
        # to avoid over-recommending blended supervisor-specific templates.
        template_count = int(
            matched[["EmployeeTypeClean", "SupervisorClean"]]
            .drop_duplicates()
            .shape[0]
        )
        missing_employee_type = employee_type_clean is None
        missing_supervisor = supervisor is None or not str(supervisor).strip()
        ambiguous_template = bool(
            template_count > 1 and (missing_employee_type or missing_supervisor)
        )

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
        grouped["ReferenceTemplateCount"] = template_count
        grouped["AmbiguousReferenceTemplate"] = ambiguous_template

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
            counter.update(filter_group_list(groups))

        rows = []

        for group_name, count in counter.items():
            raw_confidence = round(count / total_users, 3)

            if raw_confidence < self.matrix_builder.min_confidence:
                continue

            rows.append({
                "GroupName": group_name,
                "ADConfidence": raw_confidence,
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

        recs = filter_recommendations_df(recs)

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

        rights = filter_group_list(user.iloc[0]["GroupsList"])

        rows = []

        for right in rights:
            rows.append({
                "GroupName": right,
                "CopyFromUserHasIt": True,
                "CopyFromNetID": copy_from_netid,
            })

        return filter_recommendations_df(pd.DataFrame(rows))

    def _merge_all_sources(
            self,
            reference_recs: pd.DataFrame,
            ad_recs: pd.DataFrame,
            ml_recs: pd.DataFrame,
            copy_from_recs: pd.DataFrame,
    ) -> pd.DataFrame:

        def _ensure_group_key_dtype(df: pd.DataFrame) -> pd.DataFrame:
            """Force merge keys to object dtype even for empty frames."""
            out = df.copy()
            if "GroupNameClean" not in out.columns:
                out["GroupNameClean"] = ""
            if "GroupName" not in out.columns:
                out["GroupName"] = ""

            out["GroupNameClean"] = out["GroupNameClean"].fillna("").astype(str)
            out["GroupName"] = out["GroupName"].fillna("").astype(str)
            return out

        def add_group_clean(df: pd.DataFrame) -> pd.DataFrame:
            df = _ensure_group_key_dtype(df)

            if not df.empty:
                df["GroupNameClean"] = df["GroupName"].apply(self._normalize_group_name)

            return _ensure_group_key_dtype(df)

        reference_recs = add_group_clean(reference_recs)
        ad_recs = add_group_clean(ad_recs)
        ml_recs = add_group_clean(ml_recs)
        copy_from_recs = add_group_clean(copy_from_recs)



        all_group_names = set()

        for df in [reference_recs, ad_recs, ml_recs, copy_from_recs]:
            if not df.empty and "GroupNameClean" in df.columns:
                all_group_names.update(df["GroupNameClean"].dropna().astype(str))

        base = pd.DataFrame({
            "GroupNameClean": pd.Series(sorted(all_group_names), dtype="object")
        })
        base = _ensure_group_key_dtype(base)

        merged = base.merge(reference_recs, on="GroupNameClean", how="left")
        merged = merged.merge(ad_recs, on="GroupNameClean", how="left", suffixes=("", "_AD"))
        merged = merged.merge(ml_recs, on="GroupNameClean", how="left", suffixes=("", "_ML"))
        merged = merged.merge(copy_from_recs, on="GroupNameClean", how="left", suffixes=("", "_Copy"))

        def choose_group_name(row) -> str:
            ref_name = row.get("GroupName")
            ad_name = row.get("GroupName_AD")
            ml_name = row.get("GroupName_ML")
            copy_name = row.get("GroupName_Copy")

            for candidate in (ad_name, ml_name, copy_name, ref_name):
                if pd.notna(candidate) and str(candidate).strip():
                    fallback = str(candidate)
                    break
            else:
                return ""

            if not (pd.notna(ref_name) and str(ref_name).strip()):
                return fallback

            ref_text = str(ref_name).strip()
            if pd.notna(ad_name) and str(ad_name).strip():
                ad_text = str(ad_name).strip()
                ad_lower = ad_text.lower()
                ref_lower = ref_text.lower()

                # Keep AD label when reference carries a domain prefix variant.
                if ref_lower.startswith("dce.") or ref_lower.startswith("dce-") or ref_lower.startswith("dce "):
                    return ad_text

                # Keep reference label when AD uses technical transport prefixes.
                if ad_lower.startswith("m.") or ad_lower.startswith("i."):
                    return ref_text

                # Prefer AD/copy label when it is the same normalized right but
                # human formatting differs (e.g., spaces vs hyphens).
                if self._normalize_group_name(ad_text) == self._normalize_group_name(ref_text):
                    if " " in ad_text and "-" in ref_text:
                        return ad_text

            return ref_text

        merged["GroupName"] = merged.apply(choose_group_name, axis=1)


        merged["InReferenceSheet"] = merged["InReferenceSheet"].fillna(False)
        merged["CopyFromUserHasIt"] = merged["CopyFromUserHasIt"].fillna(False)

        merged["ADConfidence"] = merged["ADConfidence"].fillna(0)
        merged["MLConfidence"] = merged["MLConfidence"].fillna(0)

        merged["UserCountWithGroup"] = merged["UserCountWithGroup"].fillna(0).astype(int)
        merged["TotalUsersInRole"] = merged["TotalUsersInRole"].fillna(0).astype(int)

        merged["MLSupportCount"] = merged["MLSupportCount"].fillna(0).astype(int)
        merged["MLComparedUsers"] = merged["MLComparedUsers"].fillna(0).astype(int)

        merged["ReferenceCategories"] = merged["ReferenceCategories"].fillna("")
        merged["ReferenceTemplateCount"] = merged["ReferenceTemplateCount"].fillna(0).astype(int)
        merged["AmbiguousReferenceTemplate"] = merged["AmbiguousReferenceTemplate"].fillna(False)
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

        is_ambiguous_ref = bool(row.get("AmbiguousReferenceTemplate", False))

        if row["InReferenceSheet"] and not is_ambiguous_ref:
            score += reference_weight
        elif row["InReferenceSheet"] and is_ambiguous_ref:
            # Conservative fallback for blended templates: keep some reference
            # influence, but avoid giving it full "strong structured signal" weight.
            score += reference_weight * 0.5

        cohort_reliability = float(row.get("CohortReliability", 0.0))
        if row["InReferenceSheet"] or row["CopyFromUserHasIt"]:
            cohort_reliability = max(cohort_reliability, 0.8)
        else:
            cohort_reliability = max(cohort_reliability, 0.5)
        global_rate = float(row.get("GlobalGroupRate", 0.0))
        commonality_penalty = max(0.0, 1.0 - max(0.0, global_rate - 0.5))
        ad_signal = float(row["ADConfidence"]) * cohort_reliability * commonality_penalty
        ml_signal = float(row["MLConfidence"]) * cohort_reliability * commonality_penalty
        support_ratio = float(row.get("SupportRatio", 0.0))

        if ad_signal >= 0.8:
            score += ad_weights[0]
        elif ad_signal >= 0.6:
            score += ad_weights[1]
        elif ad_signal >= 0.4:
            score += ad_weights[2]

        if ml_signal >= 0.8 and support_ratio >= 0.6:
            score += ml_weights[0]
        elif ml_signal >= 0.6 and support_ratio >= 0.5:
            score += ml_weights[1]
        elif ml_signal >= 0.4 and support_ratio >= 0.4:
            score += ml_weights[2]

        if row["CopyFromUserHasIt"]:
            score += copy_weight

        if row["InReferenceSheet"] and not is_ambiguous_ref:
            score = max(score, reference_weight)

        if row["RiskLevel"] == "High" and not row["InReferenceSheet"]:
            score *= 0.5

        return round(min(score, 1.0), 3)

    def _final_decision(self, row) -> str:
        if row["RiskLevel"] == "High":
            return "Manual Review"

        score = row["FinalScore"]

        if score >= 0.85:
            return "Auto Assign"

        if score >= 0.70:
            return "Strong Recommend"

        if score >= 0.50:
            return "Suggest"

        if row["CopyFromUserHasIt"] and not row["InReferenceSheet"]:
            return "Possible Extra Access"

        return "Ignore"

    def _reason(self, row) -> str:
        reasons = []

        if row["InReferenceSheet"]:
            reasons.append("listed in the access reference sheet")
            if bool(row.get("AmbiguousReferenceTemplate", False)):
                reasons.append(
                    "reference template is ambiguous "
                    f"({int(row.get('ReferenceTemplateCount', 0))} templates)"
                )

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
        confidence_bits = (
            f"score={row.get('FinalScore', 0):.2f}; "
            f"ad={row.get('ADConfidence', 0):.2f}; "
            f"ml={row.get('MLConfidence', 0):.2f}; "
            f"support={int(row.get('MLSupportCount', 0))}/{int(row.get('MLComparedUsers', 0))}; "
            f"cohort={int(row.get('CohortSize', 0))}; "
            f"global_rate={row.get('GlobalGroupRate', 0):.2f}"
        )
        return "Recommended because it is " + ", and ".join(reasons) + f". Confidence: {confidence_bits}."

    def _build_reason_series(self, merged: pd.DataFrame) -> pd.Series:
        """
        Build a stable 1-D reason series even if row-wise apply expands to a DataFrame.
        """
        raw_reasons = merged.apply(self._reason, axis=1)

        def _clean_reason(value: object) -> str:
            text = "" if value is None else str(value).strip()
            if not text or text.lower() == "nan":
                return "No strong evidence found."
            return text

        if isinstance(raw_reasons, pd.DataFrame):
            # Defensive fallback: pick first non-empty reason-like value per row.
            return raw_reasons.apply(
                lambda row: next(
                    (
                        normalized
                        for value in row.tolist()
                        for normalized in [_clean_reason(value)]
                        if normalized != "No strong evidence found."
                    ),
                    "No strong evidence found.",
                ),
                axis=1,
            )

        return raw_reasons.apply(_clean_reason)

    def _apply_signal_filters(self, merged: pd.DataFrame) -> pd.DataFrame:
        if merged.empty:
            return merged
        df = merged.copy()
        df["GroupNameNorm"] = df["GroupName"].apply(self._normalize_group_name)
        df["GlobalGroupRate"] = df["GroupNameNorm"].map(self._global_group_rates).fillna(0.0)
        df["SupportRatio"] = (
            df["MLSupportCount"] / df["MLComparedUsers"].replace(0, 1)
        ).fillna(0.0)
        df["IsVeryCommon"] = df["GlobalGroupRate"] >= self.MAX_COMMON_GROUP_RATE
        df["IsLowSignalML"] = (df["MLConfidence"] > 0) & (
            (df["SupportRatio"] < 0.4) | (df["MLComparedUsers"] < 3)
        )
        keep = (
            df["InReferenceSheet"]
            | (df["ADConfidence"] >= 0.5)
            | ((df["MLConfidence"] >= 0.6) & (~df["IsLowSignalML"]))
            | (df["CopyFromUserHasIt"])
        )
        # Suppress very-common groups unless there is structured enterprise signal.
        keep = keep & (
            ~df["IsVeryCommon"]
            | df["InReferenceSheet"]
            | (df["ADConfidence"] >= 0.6)
            | df["CopyFromUserHasIt"]
        )
        return filter_recommendations_df(df[keep].copy())

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
        self._global_group_rates = self._compute_global_group_rates(users)
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

        refined_same_department = self._refine_by_reference_overlap(
            same_department, reference_group_names, title_clean
        )
        if len(same_department) >= self.MIN_COHORT_SIZE or len(refined_same_department) >= 3:
            return refined_same_department

        # Level 3: title cross-department
        cross_dept = users[users["TitleClean"] == title_clean].copy()
        if len(cross_dept) >= self.MIN_COHORT_SIZE:
            return cross_dept

        # Level 4: return the largest non-empty candidate we found
        candidates = [c for c in [exact_cohort, same_department, cross_dept] if not c.empty]
        if candidates:
            return max(candidates, key=len)
        return exact_cohort  # empty DataFrame

    def _compute_global_group_rates(self, users: pd.DataFrame) -> dict[str, float]:
        if users.empty:
            return {}
        total = max(len(users), 1)
        counts = Counter()
        for groups in users["GroupsList"]:
            normalized = {self._normalize_group_name(g) for g in filter_group_list(groups)}
            counts.update(normalized)
        return {k: v / total for k, v in counts.items()}

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
                    1 for g in filter_group_list(groups)
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

        for prefix in ("m.", "i.", "dce.", "dce-", "dce "):
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
