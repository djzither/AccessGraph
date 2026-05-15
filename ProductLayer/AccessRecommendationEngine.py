import logging
import re
from collections import Counter

import pandas as pd

logger = logging.getLogger(__name__)

from DataLayer.access_exclusions import (
    filter_group_list,
    filter_recommendations_df,
    filter_reference_df,
    filter_user_groups_df,
)
from DataLayer.permission_normalization import (
    canonical_permission_id,
    normalize_single_permission,
)
from DataLayer.workforce_type import (
    canonical_from_ui_label,
    reference_match_value,
)
from DataLayer.peer_cohort import (
    build_peer_pool_from_anchor,
    build_target_user_row,
    contamination_stats_for_group,
    infer_workforce_type,
    is_supervisor_like,
    median_permission_count,
)
from DataLayer.subgroup_detection import analyze_recommendation_subgroups
from DeterministicLayer.access_pattern_labels import apply_access_pattern_columns
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
    ML_SCOPE_CURRENT = "current"
    ML_SCOPE_AD_COHORT = "ad_cohort"
    ML_SCOPE_HYBRID = "hybrid"
    ML_SCOPES = frozenset({ML_SCOPE_CURRENT, ML_SCOPE_AD_COHORT, ML_SCOPE_HYBRID})
    ML_EVIDENCE_STRONG_INSIDE_AD = "strong_inside_ad"
    ML_EVIDENCE_USABLE_INSIDE_AD = "usable_inside_ad"
    ML_EVIDENCE_EXTERNAL_ONLY = "external_only"
    ML_EVIDENCE_NO_ML_SUPPORT = "no_ml_support"
    ML_EVIDENCE_COHORT_TOO_SMALL = "cohort_too_small"

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
                "information technology",
            ),
        },
    }

    def __init__(
        self,
        min_confidence: float = 0.5,
        title_matcher: TitleEmbedMatcher | None = None,
        ml_scope: str = "current",
    ):
        if ml_scope not in self.ML_SCOPES:
            allowed = ", ".join(sorted(self.ML_SCOPES))
            raise ValueError(f"ml_scope must be one of: {allowed}")
        self.matrix_builder = PermissionMatrixBuilder(min_confidence=min_confidence)
        self.permission_filter = PermissionFilter()
        self.title_matcher = title_matcher
        self.ml_scope = ml_scope
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
        cohort_diagnostics: bool = False,
        recommendation_debug: bool = False,
    ) -> pd.DataFrame:
        users_df = filter_user_groups_df(users_df)
        reference_df = filter_reference_df(reference_df)
        target_canonical = canonical_from_ui_label(employee_type)

        debug = bool(cohort_diagnostics or recommendation_debug)

        reference_recs = self._get_reference_recommendations(
            reference_df=reference_df,
            title=title,
            department=department,
            employee_type=employee_type,
            supervisor=supervisor,
            users_df=users_df,
            copy_from_netid=copy_from_netid,
            reference_debug=debug,
        )
        reference_diagnostics = dict(reference_recs.attrs.get("reference_diagnostics", {}))

        target_user_row = build_target_user_row(
            title=title,
            department=department,
            employee_type=employee_type,
            sam_account_name=new_hire_netid or "",
        )
        peer_pool_metadata: dict[str, object] = {}
        comparison_cohort = self._select_ad_comparison_cohort(
            users_df=users_df,
            title=title,
            department=department,
            reference_recs=reference_recs,
            employee_type=employee_type,
            copy_from_netid=copy_from_netid,
            target_user_row=target_user_row,
            peer_pool_metadata=peer_pool_metadata,
            cohort_diagnostics=debug,
        )

        ad_recs = self._get_ad_recommendations(
            comparison_cohort=comparison_cohort,
        )

        ml_recs, ml_audit = self._get_ml_recommendations(
            users_df=users_df,
            new_hire_netid=new_hire_netid,
            department=department,
            comparison_cohort=comparison_cohort,
            workforce_segment=target_canonical,
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
        merged.attrs["reference_diagnostics"] = reference_diagnostics
        if debug:
            merged.attrs["cohort_filter_diagnostics"] = getattr(
                comparison_cohort, "attrs", {}
            ).get("cohort_filter_diagnostics")
            med = median_permission_count(users_df)
            audits: list[dict[str, object]] = []
            if not comparison_cohort.empty:
                for _, srow in comparison_cohort.iterrows():
                    notes: list[str] = []
                    is_supervisor_like(
                        srow,
                        users_df=users_df,
                        cohort_median_group_count=med,
                        target_workforce_type=infer_workforce_type(srow),
                        decision_notes=notes,
                    )
                    audits.append(
                        {
                            "SamAccountName": str(srow.get("SamAccountName", "")),
                            "decision_notes": list(notes),
                        }
                    )
            merged.attrs["supervisor_decision_audits"] = audits
            merged.attrs["recommendation_debug"] = True
        elif cohort_diagnostics:
            merged.attrs["cohort_filter_diagnostics"] = getattr(
                comparison_cohort, "attrs", {}
            ).get("cohort_filter_diagnostics")
        merged = self._apply_ml_scope_diagnostics(merged, ml_audit)
        wattrs = getattr(comparison_cohort, "attrs", None) or {}
        mix = wattrs.get("workforce_mix") or {}
        mix_str = ", ".join(f"{k}={v}" for k, v in sorted(mix.items()))

        merged["EmployeeTypeClean"] = str(employee_type).lower().strip()
        merged["WorkforceSegmentTarget"] = target_canonical
        merged["CohortWorkforceTarget"] = wattrs.get("workforce_target", target_canonical)
        merged["CohortWorkforceFallback"] = bool(wattrs.get("workforce_fallback", False))
        merged["CohortEmployeeTypeMix"] = mix_str
        merged["CohortFallbackLevel"] = wattrs.get("cohort_fallback_level", "")
        merged["CohortUsedForScoring"] = wattrs.get("cohort_used_for_scoring", "")
        merged["CohortUsedMix"] = mix_str
        merged["IsFSYRole"] = self._is_fsy_role(title=title, department=department)
        merged["CohortSize"] = len(comparison_cohort)
        merged["CohortReliability"] = min(1.0, len(comparison_cohort) / self.MIN_RELIABLE_COHORT)
        for meta_key, default in {
            "AnchorUserName": "",
            "AnchorUserTitle": "",
            "AnchorUserType": "",
            "PeerPoolSize": len(comparison_cohort),
            "SupervisorUsersExcluded": "",
            "OutlierUsersExcluded": "",
            "PeerPoolComposition": "",
            "PeerUsers": "",
            "TargetWorkforceType": "",
            "AnchorWorkforceType": "",
            "AnchorMismatchFlag": False,
            "ManagerNetId": "",
            "FullTimeExcludedForStudentTarget": "",
            "StudentsExcludedForFullTimeTarget": "",
            "ManagerOfOthersExcluded": "",
            "FallbackReason": "",
        }.items():
            merged[meta_key] = peer_pool_metadata.get(meta_key, default)

        merged = self.permission_filter.filter_recommendations(merged)
        merged = self._apply_signal_filters(merged)
        merged = self._apply_peer_contamination_metadata(
            merged=merged,
            comparison_cohort=comparison_cohort,
            peer_pool_metadata=peer_pool_metadata,
            target_user_row=target_user_row,
            users_df=users_df,
        )
        merged = self._apply_reference_governance_metadata(merged)

        merged["FinalScore"] = merged.apply(self._score_row, axis=1)
        merged["FinalDecision"] = merged.apply(self._final_decision, axis=1)
        merged["Reason"] = self._build_reason_series(merged)

        sub_df = analyze_recommendation_subgroups(
            comparison_cohort=comparison_cohort,
            recommendations_df=merged,
        )
        merged = apply_access_pattern_columns(merged, sub_df)

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
        *,
        reference_debug: bool = False,
    ) -> pd.DataFrame:

        ref = reference_df.copy()
        ref = filter_reference_df(ref)

        empty_reference_columns = [
            "GroupNameClean",
            "GroupName",
            "InReferenceSheet",
            "ReferenceCategories",
            "ReferenceTemplateCount",
            "AmbiguousReferenceTemplate",
        ]
        if ref.empty:
            empty = pd.DataFrame(columns=empty_reference_columns)
            rd: dict[str, object] = {
                "reference_match_path": "no_reference_match",
                "target_title": str(title).strip(),
                "target_department": str(department).strip(),
                "target_title_clean": self._normalize_role_text(title),
                "target_department_clean": self._normalize_role_text(department),
                "fallback_matched_title": None,
                "fallback_candidate_departments": [],
                "fallback_rows_after_department_guard": 0,
                "fallback_empty_due_to_department_mismatch": False,
            }
            if reference_debug:
                rd["match_stages"] = {
                    "raw_target": {
                        "title": str(title).strip(),
                        "department": str(department).strip(),
                    },
                    "reason": "reference_frame_empty_after_filters",
                }
            empty.attrs["reference_diagnostics"] = rd
            return empty

        for column, default in (
            ("JobTitle", ""),
            ("Department", ""),
            ("EmployeeType", ""),
            ("Supervisor", ""),
            ("AccessName", ""),
        ):
            if column not in ref.columns:
                ref[column] = default

        ref["JobTitleClean"] = ref["JobTitle"].apply(self._normalize_role_text)
        ref["DepartmentClean"] = ref["Department"].apply(self._normalize_role_text)
        if "EmployeeTypeClean" in ref.columns:
            ref["EmployeeTypeClean"] = ref["EmployeeTypeClean"].astype(str).str.lower().str.strip()
        elif "EmployeeType" in ref.columns:
            ref["EmployeeTypeClean"] = ref["EmployeeType"].astype(str).str.lower().str.strip()
        else:
            ref["EmployeeTypeClean"] = ""
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
            reference_match_value(canonical_from_ui_label(employee_type))
            if employee_type is not None and str(employee_type).strip()
            else None
        )

        dept_clean = self._normalize_role_text(department)
        title_clean_norm = self._normalize_role_text(title)
        diag: dict[str, object] = {
            "reference_match_path": "no_reference_match",
            "target_title": str(title).strip(),
            "target_department": str(department).strip(),
            "target_title_clean": title_clean_norm,
            "target_department_clean": dept_clean,
            "fallback_matched_title": None,
            "fallback_candidate_departments": [],
            "fallback_rows_after_department_guard": 0,
            "fallback_empty_due_to_department_mismatch": False,
        }

        match_stages: dict[str, object] | None = {} if reference_debug else None

        def _ref_stage(key: str, value: object) -> None:
            if match_stages is not None:
                match_stages[key] = value

        if reference_debug:
            _ref_stage(
                "raw_target",
                {
                    "title": str(title).strip(),
                    "department": str(department).strip(),
                    "employee_type": None
                    if employee_type is None
                    else str(employee_type).strip(),
                    "supervisor": None
                    if supervisor is None
                    else str(supervisor).strip(),
                    "copy_from_netid": copy_from_netid,
                },
            )
            _ref_stage(
                "canonical_target",
                {
                    "role_candidates": sorted(
                        [list(t) for t in role_candidates],
                        key=lambda x: (x[0], x[1]),
                    ),
                    "employee_type_clean": employee_type_clean,
                    "title_clean": title_clean_norm,
                    "department_clean": dept_clean,
                },
            )
            _ref_stage("reference_row_count_after_column_normalization", int(len(ref)))

        matched = ref[
            ref.apply(
                lambda row: (row["JobTitleClean"], row["DepartmentClean"]) in role_candidates,
                axis=1,
            )
        ].copy()
        if reference_debug:
            _ref_stage("candidate_rows_after_exact_title_department_join", int(len(matched)))

        if "EmployeeTypeClean" not in matched.columns:
            if "EmployeeType" in matched.columns:
                matched["EmployeeTypeClean"] = matched["EmployeeType"].astype(str).str.lower().str.strip()
            else:
                matched["EmployeeTypeClean"] = ""

        if employee_type_clean is not None:
            matched = matched[matched["EmployeeTypeClean"] == employee_type_clean].copy()
        if reference_debug:
            _ref_stage("candidate_rows_after_employee_type_filter", int(len(matched)))

        if not matched.empty:
            diag["reference_match_path"] = "exact_title_dept"
        else:
            if employee_type_clean is not None:
                employee_ref = ref[ref["EmployeeTypeClean"] == employee_type_clean].copy()
            else:
                employee_ref = ref.copy()

            if not employee_ref.empty:
                diag["fallback_candidate_departments"] = sorted(
                    employee_ref["DepartmentClean"].dropna().astype(str).unique().tolist()
                )

                dept_scoped = employee_ref[
                    employee_ref["DepartmentClean"] == dept_clean
                ].copy()
                diag["fallback_rows_after_department_guard"] = int(len(dept_scoped))
                if reference_debug:
                    _ref_stage("fallback_employee_ref_row_count", int(len(employee_ref)))
                    _ref_stage("fallback_dept_scoped_row_count", int(len(dept_scoped)))

                if dept_scoped.empty:
                    diag["fallback_empty_due_to_department_mismatch"] = bool(len(employee_ref) > 0)
                    matched = ref.iloc[0:0].copy()
                else:
                    candidate_titles = dept_scoped["JobTitle"].dropna().astype(str).unique().tolist()
                    if reference_debug:
                        _ref_stage("fallback_embed_candidate_title_values", list(candidate_titles))
                    embed_matcher = self.title_matcher
                    if embed_matcher is None:
                        try:
                            embed_matcher = TitleEmbedMatcher()
                        except Exception:
                            embed_matcher = None

                    if embed_matcher is not None and candidate_titles:
                        best_title, _ = embed_matcher.best_match(title, candidate_titles)
                        if best_title is not None:
                            matched = dept_scoped[
                                dept_scoped["JobTitle"].astype(str) == best_title
                            ].copy()
                            if not matched.empty:
                                diag["reference_match_path"] = "fallback_title_same_department"
                                diag["fallback_matched_title"] = str(best_title)
                                logger.debug(
                                    "reference title fallback (department-scoped): "
                                    "target_title=%r target_department_clean=%r "
                                    "matched_title=%r dept_rows=%s",
                                    title,
                                    dept_clean,
                                    best_title,
                                    len(dept_scoped),
                                )
                        else:
                            matched = ref.iloc[0:0].copy()
                    else:
                        matched = ref.iloc[0:0].copy()

        if employee_type_clean == "full time" and supervisor is not None:
            supervisor_clean = str(supervisor).lower().strip()

            supervisor_matches = matched[
                matched["SupervisorClean"] == supervisor_clean
            ]

            if not supervisor_matches.empty:
                matched = supervisor_matches
        if reference_debug:
            _ref_stage("candidate_rows_after_supervisor_narrowing_if_applicable", int(len(matched)))

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
                        & (ref["DepartmentClean"] == dept_clean)
                    ]
                    if not name_matches.empty:
                        matched = name_matches.copy()
                        diag["reference_match_path"] = "copy_from_reference_name"
        if reference_debug:
            _ref_stage("candidate_rows_after_copy_from_name_if_applicable", int(len(matched)))

        if matched.empty:
            if match_stages is not None:
                match_stages["final_matched_source_rows"] = 0
                match_stages["final_matched_access_names"] = []
                match_stages["final_recommended_group_name_clean"] = []
                diag["match_stages"] = match_stages
            empty = pd.DataFrame(columns=[
                "GroupNameClean",
                "GroupName",
                "InReferenceSheet",
                "ReferenceCategories",
                "ReferenceTemplateCount",
                "AmbiguousReferenceTemplate",
            ])
            empty.attrs["reference_diagnostics"] = diag
            return empty

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

        if match_stages is not None:
            match_stages["final_matched_source_rows"] = int(len(matched))
            match_stages["final_matched_access_names"] = sorted(
                matched["AccessName"].dropna().astype(str).unique().tolist()
            )
            match_stages["final_recommended_group_name_clean"] = sorted(
                grouped["GroupNameClean"].astype(str).tolist()
            )
            diag["match_stages"] = match_stages

        grouped.attrs["reference_diagnostics"] = diag
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

    @staticmethod
    def _dataframe_netids(df: pd.DataFrame) -> set[str]:
        if df.empty or "SamAccountName" not in df.columns:
            return set()
        return {
            str(netid).strip()
            for netid in df["SamAccountName"].astype(str)
            if str(netid).strip()
        }

    @staticmethod
    def _empty_ml_recommendations() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "GroupName",
                "MLConfidence",
                "MLSupportCount",
                "MLComparedUsers",
                "NearestUsers",
                "MLMode",
                "MLAnchorNetID",
                "MLWorkforcePoolFallback",
            ]
        )

    def _harmonize_ml_recommendations(self, recs: pd.DataFrame) -> pd.DataFrame:
        if recs is None or recs.empty:
            return self._empty_ml_recommendations()

        out = recs.copy()
        out["MLMode"] = out.get("MLMode", "")
        out["MLAnchorNetID"] = out.get("MLAnchorNetID", "")
        if "MLWorkforcePoolFallback" not in out.columns:
            out["MLWorkforcePoolFallback"] = False
        return out[
            [
                "GroupName",
                "MLConfidence",
                "MLSupportCount",
                "MLComparedUsers",
                "NearestUsers",
                "MLMode",
                "MLAnchorNetID",
                "MLWorkforcePoolFallback",
            ]
        ]

    def _select_ml_recs_for_scoring(
        self,
        current_recs: pd.DataFrame,
        ad_scoped_recs: pd.DataFrame,
    ) -> pd.DataFrame:
        current = self._harmonize_ml_recommendations(current_recs)
        ad_scoped = self._harmonize_ml_recommendations(ad_scoped_recs)

        if self.ml_scope == self.ML_SCOPE_CURRENT:
            return current
        if self.ml_scope == self.ML_SCOPE_AD_COHORT:
            return ad_scoped
        if current.empty:
            return ad_scoped
        if ad_scoped.empty:
            return current

        current = current.copy()
        ad_scoped = ad_scoped.copy()
        current["GroupNameClean"] = current["GroupName"].apply(self._normalize_group_name)
        ad_scoped["GroupNameClean"] = ad_scoped["GroupName"].apply(self._normalize_group_name)
        ad_by_key = ad_scoped.set_index("GroupNameClean", drop=False)
        selected_rows: list[pd.Series] = []
        seen_keys: set[str] = set()

        for _, row in current.iterrows():
            key = str(row["GroupNameClean"])
            seen_keys.add(key)
            if key in ad_by_key.index and float(ad_by_key.loc[key, "MLConfidence"]) > 0:
                selected_rows.append(ad_by_key.loc[key])
            else:
                selected_rows.append(row)

        for key, ad_row in ad_by_key.iterrows():
            if key in seen_keys:
                continue
            selected_rows.append(ad_row)

        if not selected_rows:
            return self._empty_ml_recommendations()

        combined = pd.DataFrame(selected_rows)
        return self._harmonize_ml_recommendations(combined)

    def _ml_peer_aggregate_recommendations(
        self,
        ml: MLRecommender,
        comparison_cohort: pd.DataFrame,
        workforce_segment: str,
    ) -> pd.DataFrame:
        return ml.recommend_for_peer_cohort(
            cohort_df=comparison_cohort,
            min_support=2,
            workforce_segment=workforce_segment,
            peer_aggregate_fallback=getattr(comparison_cohort, "attrs", {}).get(
                "workforce_fallback", False
            ),
            respect_anchor_pool=getattr(comparison_cohort, "attrs", {}).get(
                "peer_pool_locked", False
            ),
        )

    def _build_ml_scope_audit(
        self,
        *,
        current_recs: pd.DataFrame,
        ad_scoped_recs: pd.DataFrame,
        current_ml_pool: pd.DataFrame,
        comparison_cohort: pd.DataFrame,
    ) -> dict[str, object]:
        current_ids = self._dataframe_netids(current_ml_pool)
        ad_ids = self._dataframe_netids(comparison_cohort)
        intersection_ids = current_ids & ad_ids
        ml_only_ids = current_ids - ad_ids
        ad_only_ids = ad_ids - current_ids

        current_by_group = self._harmonize_ml_recommendations(current_recs).copy()
        ad_scoped_by_group = self._harmonize_ml_recommendations(ad_scoped_recs).copy()
        if not current_by_group.empty:
            current_by_group["GroupNameClean"] = current_by_group["GroupName"].apply(
                self._normalize_group_name
            )
        if not ad_scoped_by_group.empty:
            ad_scoped_by_group["GroupNameClean"] = ad_scoped_by_group["GroupName"].apply(
                self._normalize_group_name
            )

        return {
            "ml_scope": self.ml_scope,
            "ad_cohort_size": len(ad_ids),
            "current_ml_pool_size": len(current_ids),
            "ml_ad_intersection_size": len(intersection_ids),
            "ml_only_count": len(ml_only_ids),
            "ad_only_count": len(ad_only_ids),
            "current_by_group": current_by_group,
            "ad_scoped_by_group": ad_scoped_by_group,
        }

    @classmethod
    def derive_ml_evidence_quality(
        cls,
        *,
        has_current_ml_support: bool,
        has_ad_scoped_ml_support: bool,
        ad_scoped_ml_compared_users: int,
        ml_ad_cohort_size: int,
    ) -> str:
        if has_ad_scoped_ml_support and ad_scoped_ml_compared_users >= 5:
            return cls.ML_EVIDENCE_STRONG_INSIDE_AD
        if has_ad_scoped_ml_support and ad_scoped_ml_compared_users < 5:
            return cls.ML_EVIDENCE_USABLE_INSIDE_AD
        if has_current_ml_support and not has_ad_scoped_ml_support:
            return cls.ML_EVIDENCE_EXTERNAL_ONLY
        if ml_ad_cohort_size > 0 and ad_scoped_ml_compared_users < 2:
            return cls.ML_EVIDENCE_COHORT_TOO_SMALL
        return cls.ML_EVIDENCE_NO_ML_SUPPORT

    def _apply_ml_scope_diagnostics(
        self,
        merged: pd.DataFrame,
        ml_audit: dict[str, object],
    ) -> pd.DataFrame:
        if merged.empty:
            return merged

        df = merged.copy()
        df["MLScope"] = str(ml_audit.get("ml_scope", self.ml_scope))
        df["MLCurrentPoolSize"] = int(ml_audit.get("current_ml_pool_size", 0) or 0)
        df["MLAdCohortSize"] = int(ml_audit.get("ad_cohort_size", 0) or 0)
        df["MLAdIntersectionSize"] = int(ml_audit.get("ml_ad_intersection_size", 0) or 0)
        df["MLOnlyCount"] = int(ml_audit.get("ml_only_count", 0) or 0)
        df["ADOnlyCount"] = int(ml_audit.get("ad_only_count", 0) or 0)

        current_by_group = ml_audit.get("current_by_group")
        ad_scoped_by_group = ml_audit.get("ad_scoped_by_group")
        if not isinstance(current_by_group, pd.DataFrame):
            current_by_group = self._empty_ml_recommendations()
        if not isinstance(ad_scoped_by_group, pd.DataFrame):
            ad_scoped_by_group = self._empty_ml_recommendations()

        current_lookup: dict[str, float] = {}
        current_compared_lookup: dict[str, int] = {}
        if not current_by_group.empty and "GroupNameClean" in current_by_group.columns:
            for _, row in current_by_group.iterrows():
                key = str(row["GroupNameClean"])
                current_lookup[key] = float(row.get("MLConfidence", 0) or 0)
                current_compared_lookup[key] = int(row.get("MLComparedUsers", 0) or 0)
        ad_scoped_lookup: dict[str, float] = {}
        ad_scoped_compared_lookup: dict[str, int] = {}
        if not ad_scoped_by_group.empty and "GroupNameClean" in ad_scoped_by_group.columns:
            for _, row in ad_scoped_by_group.iterrows():
                key = str(row["GroupNameClean"])
                ad_scoped_lookup[key] = float(row.get("MLConfidence", 0) or 0)
                ad_scoped_compared_lookup[key] = int(row.get("MLComparedUsers", 0) or 0)

        if "GroupNameClean" not in df.columns:
            df["GroupNameClean"] = df["GroupName"].apply(self._normalize_group_name)

        current_confidences: list[float] = []
        ad_scoped_confidences: list[float] = []
        current_compared_users: list[int] = []
        ad_scoped_compared_users: list[int] = []
        has_current_support: list[bool] = []
        has_ad_scoped_support: list[bool] = []
        would_lose_support: list[bool] = []
        ml_evidence_quality: list[str] = []
        ml_ad_cohort_size = int(ml_audit.get("ad_cohort_size", 0) or 0)

        for _, row in df.iterrows():
            key = str(row.get("GroupNameClean", ""))
            current_confidence = current_lookup.get(key, 0.0)
            ad_scoped_confidence = ad_scoped_lookup.get(key, 0.0)
            current_compared = current_compared_lookup.get(key, 0)
            ad_scoped_compared = ad_scoped_compared_lookup.get(key, 0)
            current_confidences.append(current_confidence)
            ad_scoped_confidences.append(ad_scoped_confidence)
            current_compared_users.append(current_compared)
            ad_scoped_compared_users.append(ad_scoped_compared)
            has_current = current_confidence > 0
            has_ad_scoped = ad_scoped_confidence > 0
            has_current_support.append(has_current)
            has_ad_scoped_support.append(has_ad_scoped)
            would_lose_support.append(has_current and not has_ad_scoped)
            ml_evidence_quality.append(
                self.derive_ml_evidence_quality(
                    has_current_ml_support=has_current,
                    has_ad_scoped_ml_support=has_ad_scoped,
                    ad_scoped_ml_compared_users=ad_scoped_compared,
                    ml_ad_cohort_size=ml_ad_cohort_size,
                )
            )

        df["CurrentMLConfidence"] = current_confidences
        df["AdScopedMLConfidence"] = ad_scoped_confidences
        df["CurrentMLComparedUsers"] = current_compared_users
        df["AdScopedMLComparedUsers"] = ad_scoped_compared_users
        df["HasCurrentMLSupport"] = has_current_support
        df["HasAdScopedMLSupport"] = has_ad_scoped_support
        df["WouldLoseMLSupportIfScopedToAD"] = would_lose_support
        df["MLEvidenceQuality"] = ml_evidence_quality
        return df

    def _get_ml_recommendations(
        self,
        users_df: pd.DataFrame,
        new_hire_netid: str | None,
        department: str,
        comparison_cohort: pd.DataFrame,
        workforce_segment: str,
    ) -> tuple[pd.DataFrame, dict[str, object]]:

        ml = MLRecommender(users_df)

        if new_hire_netid is not None:
            target_user = users_df[users_df["SamAccountName"] == new_hire_netid]
            segment = workforce_segment
            if not target_user.empty and "EmployeeType" in target_user.columns:
                segment = str(target_user.iloc[0]["EmployeeType"])
            target_title = (
                str(target_user.iloc[0]["Title"])
                if not target_user.empty and "Title" in target_user.columns
                else ""
            )
            target_department = (
                str(target_user.iloc[0]["Department"])
                if not target_user.empty and "Department" in target_user.columns
                else department
            )
            current_recs = ml.recommend_for_user(
                sam_account_name=new_hire_netid,
                department=department,
                top_n_users=5,
                min_support=2,
                include_supervisors=False,
                workforce_segment=segment,
            )
            current_pool, _ = ml.similarity_pool_for_user(
                title=target_title,
                department=target_department,
                include_supervisors=False,
                workforce_segment=segment,
            )
            ad_scoped_recs = ml.recommend_for_similarity_pool(
                new_hire_netid,
                comparison_cohort,
                top_n_users=5,
                min_support=2,
                pool_wf_fallback=bool(
                    getattr(comparison_cohort, "attrs", {}).get("workforce_fallback", False)
                ),
                ml_mode="ad_cohort",
                ml_anchor_netid=new_hire_netid,
            )
        else:
            current_recs = self._ml_peer_aggregate_recommendations(
                ml,
                comparison_cohort,
                workforce_segment,
            )
            current_pool = comparison_cohort
            ad_scoped_recs = self._ml_peer_aggregate_recommendations(
                ml,
                comparison_cohort,
                workforce_segment,
            )

        ml_audit = self._build_ml_scope_audit(
            current_recs=current_recs,
            ad_scoped_recs=ad_scoped_recs,
            current_ml_pool=current_pool,
            comparison_cohort=comparison_cohort,
        )
        scoring_recs = self._select_ml_recs_for_scoring(current_recs, ad_scoped_recs)
        return scoring_recs, ml_audit

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
        base["GroupNameClean"] = base["GroupNameClean"].fillna("").astype(str)

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
        if "MLWorkforcePoolFallback" not in merged.columns:
            merged["MLWorkforcePoolFallback"] = False
        else:
            merged["MLWorkforcePoolFallback"] = merged["MLWorkforcePoolFallback"].fillna(False)
        merged["CopyFromNetID"] = merged["CopyFromNetID"].fillna("")

        merged["Score"] = merged["ADConfidence"]

        return merged

    @staticmethod
    def _student_peer_support_count(row) -> int:
        return int(row.get("PeerStudentSupportCount", 0) or 0)

    @classmethod
    def _reference_support_count(cls, row) -> int:
        if not bool(row.get("InReferenceSheet", False)):
            return 0
        count = int(row.get("ReferenceTemplateCount", 0) or 0)
        return count if count > 0 else 1

    @classmethod
    def _student_reference_contamination_conflict(cls, row) -> bool:
        if str(row.get("EmployeeTypeClean", "")).lower().strip() != "student":
            return False
        if not bool(row.get("InReferenceSheet", False)):
            return False
        return bool(row.get("SupervisorContaminationFlag", False))

    def _apply_reference_governance_metadata(self, merged: pd.DataFrame) -> pd.DataFrame:
        if merged.empty:
            return merged

        df = merged.copy()
        for idx, row in df.iterrows():
            df.at[idx, "ReferenceSupportCount"] = self._reference_support_count(row)
            df.at[idx, "PeerStudentSupportCount"] = self._student_peer_support_count(row)
            df.at[idx, "SupervisorSupportCount"] = int(row.get("SupervisorSupportCount", 0) or 0)
            df.at[idx, "ReferenceContaminationConflict"] = (
                self._student_reference_contamination_conflict(row)
            )
        return df

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
        copy_anchor = row["CopyFromUserHasIt"] and not bool(row.get("AnchorMismatchFlag", False))
        if row["InReferenceSheet"] or copy_anchor:
            cohort_reliability = max(cohort_reliability, 0.8)
        else:
            cohort_reliability = max(cohort_reliability, 0.5)
        global_rate = float(row.get("GlobalGroupRate", 0.0))
        commonality_penalty = max(0.0, 1.0 - max(0.0, global_rate - 0.5))
        workforce_penalty = 0.85 if (
            bool(row.get("CohortWorkforceFallback", False))
            or bool(row.get("MLWorkforcePoolFallback", False))
        ) else 1.0
        contamination_penalty = 0.35 if bool(row.get("SupervisorContaminationFlag", False)) else 1.0
        if contamination_penalty < 1.0 and employee_type_clean == "student":
            workforce_penalty *= contamination_penalty
        if bool(row.get("AnchorMismatchFlag", False)) and employee_type_clean == "student":
            workforce_penalty *= 0.5
        ad_signal = (
            float(row["ADConfidence"]) * cohort_reliability * commonality_penalty * workforce_penalty
        )
        ml_signal = (
            float(row["MLConfidence"]) * cohort_reliability * commonality_penalty * workforce_penalty
        )
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
            if not bool(row.get("AnchorMismatchFlag", False)):
                score += copy_weight
            if employee_type_clean == "student" and bool(row.get("SupervisorContaminationFlag", False)):
                score = min(score, copy_weight + 0.05)

        peer_student_support = int(row.get("PeerStudentSupportCount", 0) or 0)
        full_time_support = int(row.get("FullTimeSupportCount", 0) or 0)
        if employee_type_clean == "student" and full_time_support > peer_student_support:
            score = min(score, 0.45)
        if (
            employee_type_clean == "student"
            and peer_student_support >= 2
            and float(row["ADConfidence"]) >= 0.99
            and not bool(row.get("SupervisorContaminationFlag", False))
            and not bool(row.get("AnchorMismatchFlag", False))
        ):
            score = max(score, 0.55)

        # Reference templates may raise visibility, but contaminated student rows
        # must not be promoted to Suggest/Strong Recommend on reference floors alone.
        conflict = self._student_reference_contamination_conflict(row)
        if row["InReferenceSheet"] and not is_ambiguous_ref:
            if not conflict or peer_student_support >= 2:
                score = max(score, reference_weight)
        if conflict and peer_student_support < 2:
            score = min(score, 0.49)

        if row["RiskLevel"] == "High" and not row["InReferenceSheet"]:
            score *= 0.5

        return round(min(score, 1.0), 3)

    def _final_decision(self, row) -> str:
        if row["RiskLevel"] == "High":
            return "Manual Review"

        employee_type_clean = str(row.get("EmployeeTypeClean", "")).lower().strip()

        # Reference-backed student rows still require student peer support when
        # supervisor/full-time contamination is present.
        if (
            employee_type_clean == "student"
            and self._student_reference_contamination_conflict(row)
            and self._student_peer_support_count(row) < 2
        ):
            return "Manual Review"

        if bool(row.get("SupervisorContaminationFlag", False)):
            if employee_type_clean == "student" and not bool(row.get("InReferenceSheet", False)):
                return "Manual Review"

        if bool(row.get("AnchorMismatchFlag", False)):
            if (
                employee_type_clean == "student"
                and row["CopyFromUserHasIt"]
                and not bool(row.get("InReferenceSheet", False))
                and self._student_peer_support_count(row) < 2
            ):
                return "Manual Review"

        score = row["FinalScore"]

        if score >= 0.85:
            if employee_type_clean == "student" and (
                bool(row.get("SupervisorContaminationFlag", False))
                or bool(row.get("AnchorMismatchFlag", False))
            ):
                return "Manual Review"
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

        if bool(row.get("CohortWorkforceFallback", False)) or bool(
            row.get("MLWorkforcePoolFallback", False)
        ):
            reasons.append(
                "compared across mixed workforce segments (reduced confidence)"
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

        anchor_name = str(row.get("AnchorUserName", "")).strip()
        if anchor_name:
            reasons.append(f"peer baseline built from users similar to {anchor_name}")

        peer_users = str(row.get("PeerUsers", "")).strip()
        if peer_users:
            reasons.append(f"peer users used: {peer_users}")

        supervisors_excluded = str(row.get("SupervisorUsersExcluded", "")).strip()
        if supervisors_excluded:
            reasons.append(f"supervisors excluded: {supervisors_excluded}")

        outliers_excluded = str(row.get("OutlierUsersExcluded", "")).strip()
        if outliers_excluded:
            reasons.append(f"outliers excluded: {outliers_excluded}")

        if bool(row.get("SupervisorContaminationFlag", False)):
            reasons.append(
                "recommendation confidence reduced due to supervisor contamination risk"
            )
            reasons.append(
                "peer evidence="
                f"{int(row.get('PeerStudentSupportCount', 0))}; "
                f"supervisor evidence={int(row.get('SupervisorSupportCount', 0))}"
            )

        if bool(row.get("ReferenceContaminationConflict", False)):
            reasons.append(
                "Reference template includes this access, but supporting peer evidence "
                "comes primarily from supervisor/full-time users."
            )

        review_reason = str(row.get("ReviewReason", "")).strip()
        if review_reason:
            reasons.append(review_reason)

        ml_evidence_quality = str(row.get("MLEvidenceQuality", "")).strip()
        if ml_evidence_quality:
            reasons.append(f"ML evidence quality is {ml_evidence_quality.replace('_', ' ')}")

        if not reasons:
            return "No strong evidence found."
        confidence_bits = (
            f"score={row.get('FinalScore', 0):.2f}; "
            f"ad={row.get('ADConfidence', 0):.2f}; "
            f"ml={row.get('MLConfidence', 0):.2f}; "
            f"support={int(row.get('MLSupportCount', 0))}/{int(row.get('MLComparedUsers', 0))}; "
            f"cohort={int(row.get('CohortSize', 0))}; "
            f"global_rate={row.get('GlobalGroupRate', 0):.2f}; "
            f"ml_evidence={ml_evidence_quality or 'no_ml_support'}"
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

    def _cohort_with_workforce(
        self,
        cohort: pd.DataFrame,
        target_canonical: str,
    ) -> pd.DataFrame:
        """Prefer users with the same WorkforceSegment; widen with attrs when too small."""
        meta = {
            "workforce_target": target_canonical,
            "workforce_fallback": False,
            "workforce_mix": {},
        }
        if cohort.empty:
            out = cohort.copy()
            out.attrs = meta
            return out
        if "EmployeeType" not in cohort.columns:
            out = cohort.copy()
            out.attrs = meta
            return out
        mix = {str(k): int(v) for k, v in cohort["EmployeeType"].value_counts().items()}
        meta["workforce_mix"] = mix
        strict = cohort[
            cohort["EmployeeType"].apply(canonical_from_ui_label) == target_canonical
        ].copy()
        if len(strict) >= self.MIN_COHORT_SIZE:
            strict.attrs = {**meta, "workforce_fallback": False}
            return strict
        if strict.empty:
            out = cohort.copy()
            out.attrs = {**meta, "workforce_fallback": True}
            return out
        strict.attrs = {**meta, "workforce_fallback": False}
        return strict

    def _apply_peer_contamination_metadata(
        self,
        merged: pd.DataFrame,
        comparison_cohort: pd.DataFrame,
        peer_pool_metadata: dict[str, object],
        target_user_row: dict[str, object],
        users_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if merged.empty:
            return merged

        df = merged.copy()
        target_is_student = str(target_user_row.get("EmployeeType", "")).lower().strip() == "student"
        anchor_mismatch = bool(peer_pool_metadata.get("AnchorMismatchFlag", False))
        for idx, row in df.iterrows():
            stats = contamination_stats_for_group(
                comparison_cohort,
                str(row["GroupName"]),
                normalizer=self._normalize_group_name,
                target_row=target_user_row,
                users_df=users_df,
            )
            row_meta = stats.as_row_metadata()
            row_meta["AnchorMismatchFlag"] = anchor_mismatch
            if not target_is_student:
                row_meta["SupervisorContaminationFlag"] = False
                row_meta["ReviewReason"] = ""
            for key, value in row_meta.items():
                df.at[idx, key] = value

        for key, value in peer_pool_metadata.items():
            if key not in df.columns:
                df[key] = value
        return df

    @staticmethod
    def _stamp_cohort_diagnostics(
        cohort: pd.DataFrame,
        *,
        fallback_level: object,
        used_for_scoring: str,
    ) -> pd.DataFrame:
        attrs = {
            **(getattr(cohort, "attrs", None) or {}),
            "cohort_fallback_level": fallback_level,
            "cohort_used_for_scoring": used_for_scoring,
        }
        stamped = cohort.copy()
        stamped.attrs = attrs
        return stamped

    def _select_ad_comparison_cohort(
        self,
        users_df: pd.DataFrame,
        title: str,
        department: str,
        reference_recs: pd.DataFrame,
        employee_type: str,
        copy_from_netid: str | None = None,
        target_user_row: dict[str, object] | None = None,
        peer_pool_metadata: dict[str, object] | None = None,
        cohort_diagnostics: bool = False,
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
        target_canonical = canonical_from_ui_label(employee_type)

        reference_group_names: set = set()
        if not reference_recs.empty and "GroupNameClean" in reference_recs.columns:
            reference_group_names = set(
                reference_recs["GroupNameClean"].dropna().astype(str)
            )

        if copy_from_netid is not None:
            anchor_rows = users[users["SamAccountName"] == copy_from_netid]
            if not anchor_rows.empty:
                anchor_row = anchor_rows.iloc[0]
                peer_result = build_peer_pool_from_anchor(
                    users_df=users,
                    anchor_user_row=anchor_row,
                    target_user_row=target_user_row or build_target_user_row(
                        title=title,
                        department=department,
                        employee_type=employee_type,
                    ),
                    cohort_diagnostics=cohort_diagnostics,
                )
                if peer_pool_metadata is not None:
                    peer_pool_metadata.update(peer_result.as_metadata())
                anchor_pool = peer_result.peer_pool
                if not anchor_pool.empty:
                    anchor_workforce = self._cohort_with_workforce(anchor_pool, target_canonical)
                    if len(anchor_workforce) >= 2:
                        refined_anchor = self._refine_by_reference_overlap(
                            anchor_workforce,
                            reference_group_names,
                            title_clean,
                        )
                        attrs = {
                            **getattr(anchor_workforce, "attrs", {}),
                            "peer_pool_locked": True,
                        }
                        if cohort_diagnostics and peer_result.cohort_filter_diagnostics is not None:
                            attrs["cohort_filter_diagnostics"] = peer_result.cohort_filter_diagnostics
                        refined_anchor.attrs = attrs
                        return self._stamp_cohort_diagnostics(
                            refined_anchor,
                            fallback_level=0,
                            used_for_scoring="anchor_peer_pool",
                        )

        # Level 1: exact title + department
        exact_cohort = users[
            (users["TitleClean"] == title_clean)
            & (users["DepartmentClean"] == department_clean)
        ].copy()
        exact_workforce = self._cohort_with_workforce(exact_cohort, target_canonical)
        if len(exact_workforce) >= self.MIN_COHORT_SIZE:
            refined = self._refine_by_reference_overlap(
                exact_workforce, reference_group_names, title_clean
            )
            refined.attrs = getattr(exact_workforce, "attrs", {}).copy()
            return self._stamp_cohort_diagnostics(
                refined,
                fallback_level=1,
                used_for_scoring="title_department",
            )

        # Level 2: department-only (copy-from fallback when dept not found)
        same_department = users[users["DepartmentClean"] == department_clean].copy()
        if same_department.empty and str(employee_type).lower().strip() == "full time" and copy_from_netid is not None:
            copy_user = users[users["SamAccountName"] == copy_from_netid]
            if not copy_user.empty:
                copy_dept_clean = copy_user.iloc[0]["DepartmentClean"]
                same_department = users[users["DepartmentClean"] == copy_dept_clean].copy()

        dept_workforce = self._cohort_with_workforce(same_department, target_canonical)
        refined_same_department = self._refine_by_reference_overlap(
            dept_workforce, reference_group_names, title_clean
        )
        if len(same_department) >= self.MIN_COHORT_SIZE or len(refined_same_department) >= 3:
            refined_same_department.attrs = getattr(dept_workforce, "attrs", {}).copy()
            return self._stamp_cohort_diagnostics(
                refined_same_department,
                fallback_level=2,
                used_for_scoring="department",
            )

        # Level 3: title cross-department
        cross_dept = users[users["TitleClean"] == title_clean].copy()
        cross_workforce = self._cohort_with_workforce(cross_dept, target_canonical)
        if len(cross_workforce) >= self.MIN_COHORT_SIZE:
            refined_cross = self._refine_by_reference_overlap(
                cross_workforce, reference_group_names, title_clean
            )
            refined_cross.attrs = getattr(cross_workforce, "attrs", {}).copy()
            return self._stamp_cohort_diagnostics(
                refined_cross,
                fallback_level=3,
                used_for_scoring="title_cross_department",
            )

        # Level 4: return the largest non-empty candidate we found
        wf_exact = self._cohort_with_workforce(exact_cohort, target_canonical)
        wf_dept = self._cohort_with_workforce(same_department, target_canonical)
        wf_cross = self._cohort_with_workforce(cross_dept, target_canonical)
        candidates = [c for c in [wf_exact, wf_dept, wf_cross] if not c.empty]
        if candidates:
            best = max(candidates, key=len)
            return self._stamp_cohort_diagnostics(
                best,
                fallback_level=4,
                used_for_scoring="best_available",
            )
        out = exact_cohort.copy()
        out.attrs = getattr(wf_exact, "attrs", {}).copy()
        return self._stamp_cohort_diagnostics(
            out,
            fallback_level=4,
            used_for_scoring="title_department_empty",
        )

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
        return canonical_permission_id(value)

    @classmethod
    def _is_fsy_role(cls, title: str, department: str) -> bool:
        title_clean = cls._normalize_role_text(title)
        department_clean = cls._normalize_role_text(department)
        return "fsy" in title_clean or "fsy" in department_clean
