import logging

import pandas as pd

from DataLayer.access_exclusions import filter_group_list, filter_recommendations_df, filter_user_groups_df
from DataLayer.workforce_type import canonical_from_ui_label
from MLLayer.similarity_model import SimilarityModel

logger = logging.getLogger(__name__)


class MLRecommender:
    # Fall back to the full user dataset when a department pool is smaller
    # than this — a tiny pool produces unreliable cosine similarities.
    MIN_POOL_SIZE = 10

    def __init__(self, users_df: pd.DataFrame):
        self.users_df = filter_user_groups_df(users_df)

    @staticmethod
    def _empty_similarity_recommendations() -> pd.DataFrame:
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

    @staticmethod
    def _normalize_text(value: object) -> str:
        return str(value).lower().strip()

    @staticmethod
    def _canonical_workforce_segment(workforce_segment: str | None) -> str | None:
        if workforce_segment is None:
            return None
        return canonical_from_ui_label(workforce_segment)

    def _role_title_department_pool(
        self,
        *,
        title: str,
        department: str,
        include_supervisors: bool = False,
    ) -> pd.DataFrame:
        """
        Build an exact Title + Department cohort.

        This is intentionally strict (normalized equality) to reduce noise from
        loosely-related department peers and to make duplicate job titles more
        role-aware.
        """
        if "Title" not in self.users_df.columns or "Department" not in self.users_df.columns:
            return pd.DataFrame(columns=self.users_df.columns)

        title_clean = self._normalize_text(title)
        department_clean = self._normalize_text(department)

        pool = self.users_df[
            self.users_df["Title"].astype(str).str.lower().str.strip().eq(title_clean)
            & self.users_df["Department"].astype(str).str.lower().str.strip().eq(department_clean)
        ].copy()

        if not include_supervisors and "IsSupervisor" in pool.columns:
            pool = pool[pool["IsSupervisor"] == False]

        return pool

    def _same_department_pool(
        self,
        department: str,
        include_supervisors: bool = False,
        allow_global_fallback: bool = True,
    ) -> pd.DataFrame:

        department_clean = str(department).lower().strip()

        pool = self.users_df[
            self.users_df["Department"]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.contains(department_clean, na=False)
        ].copy()

        if not include_supervisors and "IsSupervisor" in pool.columns:
            pool = pool[pool["IsSupervisor"] == False]

        # If the department is too small for a meaningful similarity search,
        # widen to the full dataset so the model has enough variance to work with.
        # (This is the historical/default behavior; some callers may disable it
        # to explicitly enforce a role->dept->global fallback order.)
        if allow_global_fallback and len(pool) < self.MIN_POOL_SIZE:
            pool = self.users_df.copy()
            if not include_supervisors and "IsSupervisor" in pool.columns:
                pool = pool[pool["IsSupervisor"] == False]

        return pool

    def _restrict_workforce(
        self,
        pool: pd.DataFrame,
        workforce_segment: str | None,
    ) -> tuple[pd.DataFrame, bool]:
        if (
            workforce_segment is None
            or pool.empty
            or "EmployeeType" not in pool.columns
        ):
            return pool.copy(), False

        target_canonical = self._canonical_workforce_segment(workforce_segment)
        pool_types = pool["EmployeeType"].dropna().astype(str).unique().tolist()
        strict = pool[
            pool["EmployeeType"].apply(canonical_from_ui_label) == target_canonical
        ].copy()
        matched_count = len(strict)
        if matched_count >= self.MIN_POOL_SIZE:
            logger.debug(
                "ML workforce restriction applied: segment=%r canonical=%r "
                "pool_types=%r matched=%d wf_fallback=False",
                workforce_segment,
                target_canonical,
                pool_types,
                matched_count,
            )
            return strict, False

        wf_fallback = True
        logger.debug(
            "ML workforce restriction fallback: segment=%r canonical=%r "
            "pool_types=%r matched=%d pool_size=%d wf_fallback=True",
            workforce_segment,
            target_canonical,
            pool_types,
            matched_count,
            len(pool),
        )
        if strict.empty:
            return pool.copy(), wf_fallback
        return pool.copy(), wf_fallback

    def _similarity_pool_for_user(
        self,
        *,
        title: str,
        department: str,
        include_supervisors: bool = False,
        workforce_segment: str | None = None,
    ) -> tuple[pd.DataFrame, bool]:
        # Fallback order (preserves MIN_POOL_SIZE behavior):
        # 1) Exact normalized Title + Department cohort (most role-aware)
        # 2) Department-only cohort (current behavior)
        # 3) Global cohort (current MIN_POOL_SIZE fallback)
        role_pool = self._role_title_department_pool(
            title=title,
            department=department,
            include_supervisors=include_supervisors,
        )
        role_pool, wf_fb = self._restrict_workforce(role_pool, workforce_segment)
        if len(role_pool) >= self.MIN_POOL_SIZE:
            return role_pool, wf_fb

        dept_pool = self._same_department_pool(
            department=department,
            include_supervisors=include_supervisors,
            allow_global_fallback=False,
        )
        dept_pool, wf_fb2 = self._restrict_workforce(dept_pool, workforce_segment)
        wf_fb = wf_fb or wf_fb2
        if len(dept_pool) >= self.MIN_POOL_SIZE:
            return dept_pool, wf_fb

        # Final fallback: preserve existing behavior by using the full dataset.
        pool = self.users_df.copy()
        if not include_supervisors and "IsSupervisor" in pool.columns:
            pool = pool[pool["IsSupervisor"] == False]
        pool, wf_fb3 = self._restrict_workforce(pool, workforce_segment)
        return pool, wf_fb or wf_fb3

    def similarity_pool_for_user(
        self,
        *,
        title: str,
        department: str,
        include_supervisors: bool = False,
        workforce_segment: str | None = None,
    ) -> tuple[pd.DataFrame, bool]:
        return self._similarity_pool_for_user(
            title=title,
            department=department,
            include_supervisors=include_supervisors,
            workforce_segment=workforce_segment,
        )

    def _recommend_similarity_within_pool(
        self,
        *,
        sam_account_name: str,
        pool: pd.DataFrame,
        top_n_users: int,
        min_support: int,
        pool_wf_fallback: bool = False,
        ml_mode: str = "target_user",
        ml_anchor_netid: str = "",
    ) -> pd.DataFrame:
        target_user = self.users_df[self.users_df["SamAccountName"] == sam_account_name]
        if target_user.empty:
            raise ValueError(f"{sam_account_name} not found in full user data")

        scoped_pool = filter_user_groups_df(pool).copy()
        scoped_pool = pd.concat([scoped_pool, target_user], ignore_index=True)
        scoped_pool = scoped_pool.drop_duplicates(subset=["SamAccountName"])
        if len(scoped_pool) < 2:
            return self._empty_similarity_recommendations()

        model = SimilarityModel().fit(scoped_pool)
        similar_users = model.similar_users(
            sam_account_name=sam_account_name,
            top_n=top_n_users,
        )
        if similar_users.empty:
            return self._empty_similarity_recommendations()

        users_by_id = scoped_pool.set_index("SamAccountName")
        target_rights = set(filter_group_list(users_by_id.loc[sam_account_name, "GroupsList"]))
        candidate_counts: dict[str, int] = {}
        supporter_users: dict[str, list[str]] = {}

        for similar_user in similar_users["SamAccountName"]:
            rights = filter_group_list(users_by_id.loc[similar_user, "GroupsList"])
            for right in rights:
                if right in target_rights:
                    continue
                candidate_counts[right] = candidate_counts.get(right, 0) + 1
                supporter_users.setdefault(right, []).append(str(similar_user))

        rows: list[dict[str, object]] = []
        for right, count in candidate_counts.items():
            if count >= min_support:
                rows.append(
                    {
                        "GroupName": right,
                        "MLSupportCount": count,
                        "MLComparedUsers": len(similar_users),
                        "MLConfidence": count / len(similar_users),
                        "NearestUsers": ", ".join(supporter_users.get(right, [])),
                    }
                )

        if not rows:
            return self._empty_similarity_recommendations()

        out = filter_recommendations_df(pd.DataFrame(rows)).sort_values(
            ["MLConfidence", "MLSupportCount"],
            ascending=False,
        )
        out["MLMode"] = ml_mode
        out["MLAnchorNetID"] = ml_anchor_netid
        out["MLWorkforcePoolFallback"] = pool_wf_fallback
        return out

    def recommend_for_similarity_pool(
        self,
        sam_account_name: str,
        pool: pd.DataFrame,
        *,
        top_n_users: int = 5,
        min_support: int = 3,
        pool_wf_fallback: bool = False,
        ml_mode: str = "ad_cohort",
        ml_anchor_netid: str | None = None,
    ) -> pd.DataFrame:
        return self._recommend_similarity_within_pool(
            sam_account_name=sam_account_name,
            pool=pool,
            top_n_users=top_n_users,
            min_support=min_support,
            pool_wf_fallback=pool_wf_fallback,
            ml_mode=ml_mode,
            ml_anchor_netid=ml_anchor_netid or sam_account_name,
        )

    def recommend_for_user(
        self,
        sam_account_name: str,
        department: str,
        top_n_users: int = 5,
        min_support: int = 3,
        include_supervisors: bool = False,
        workforce_segment: str | None = None,
    ) -> pd.DataFrame:
        target_user = self.users_df[
            self.users_df["SamAccountName"] == sam_account_name
        ]

        if target_user.empty:
            raise ValueError(f"{sam_account_name} not found in full user data")

        # Build similarity pool with strict role-first fallback order:
        # Title+Department -> Department-only -> Global
        target_title = (
            target_user["Title"].iloc[0]
            if "Title" in target_user.columns and len(target_user["Title"]) > 0
            else ""
        )
        target_department = (
            target_user["Department"].iloc[0]
            if "Department" in target_user.columns and len(target_user["Department"]) > 0
            else department
        )
        pool, pool_wf_fallback = self._similarity_pool_for_user(
            title=target_title,
            department=target_department,
            include_supervisors=include_supervisors,
            workforce_segment=workforce_segment,
        )

        return self._recommend_similarity_within_pool(
            sam_account_name=sam_account_name,
            pool=pool,
            top_n_users=top_n_users,
            min_support=min_support,
            pool_wf_fallback=pool_wf_fallback,
            ml_mode="target_user",
            ml_anchor_netid=sam_account_name,
        )

    def recommend_for_role_peers(
        self,
        title: str,
        department: str,
        top_n_users: int = 5,
        min_support: int = 2,
        include_supervisors: bool = False,
    ) -> pd.DataFrame:

        pool = self._same_department_pool(
            department=department,
            include_supervisors=include_supervisors,
        )

        title_clean = str(title).lower().strip()
        department_clean = str(department).lower().strip()

        if "Title" not in pool.columns:
            return pd.DataFrame()

        role_peers = pool[
            pool["Title"].astype(str).str.lower().str.strip().eq(title_clean)
            & pool["Department"].astype(str).str.lower().str.strip().eq(department_clean)
        ].copy()

        if role_peers.empty:
            return pd.DataFrame()

        role_peers = role_peers.sort_values("SamAccountName").head(top_n_users)

        if len(role_peers) < 2:
            return pd.DataFrame()

        candidate_counts = {}

        for rights in role_peers["GroupsList"]:
            for right in filter_group_list(rights):
                candidate_counts[right] = candidate_counts.get(right, 0) + 1

        rows = []

        for right, count in candidate_counts.items():
            if count >= min_support:
                rows.append({
                    "GroupName": right,
                    "MLSupportCount": count,
                    "MLComparedUsers": len(role_peers),
                    "MLConfidence": count / len(role_peers),
                    "NearestUsers": ", ".join(role_peers["SamAccountName"]),
                    "MLMode": "peer_aggregate",
                })

        if not rows:
            return pd.DataFrame()

        return filter_recommendations_df(pd.DataFrame(rows)).sort_values(
            ["MLConfidence", "MLSupportCount"],
            ascending=False,
        )

    def recommend_for_peer_cohort(
        self,
        cohort_df: pd.DataFrame,
        min_support: int = 2,
        workforce_segment: str | None = None,
        peer_aggregate_fallback: bool = False,
        respect_anchor_pool: bool = False,
    ) -> pd.DataFrame:

        if cohort_df.empty:
            return pd.DataFrame()

        role_peers = filter_user_groups_df(cohort_df)
        ml_wf_fb = bool(peer_aggregate_fallback)
        target_canonical = self._canonical_workforce_segment(workforce_segment)
        if (
            not respect_anchor_pool
            and target_canonical is not None
            and "EmployeeType" in role_peers.columns
        ):
            pool_types = role_peers["EmployeeType"].dropna().astype(str).unique().tolist()
            strict = role_peers[
                role_peers["EmployeeType"].apply(canonical_from_ui_label) == target_canonical
            ].copy()
            logger.debug(
                "ML peer cohort workforce filter: segment=%r canonical=%r "
                "pool_types=%r matched=%d respect_anchor_pool=%s",
                workforce_segment,
                target_canonical,
                pool_types,
                len(strict),
                respect_anchor_pool,
            )
            if len(strict) >= 2:
                role_peers = strict
            elif strict.empty and len(role_peers) > 0:
                ml_wf_fb = True

        if "SamAccountName" in role_peers.columns:
            role_peers = role_peers.sort_values("SamAccountName")

        if len(role_peers) < 2:
            return pd.DataFrame()

        candidate_counts = {}

        for rights in role_peers["GroupsList"]:
            for right in filter_group_list(rights):
                candidate_counts[right] = candidate_counts.get(right, 0) + 1

        rows = []

        for right, count in candidate_counts.items():
            if count >= min_support:
                rows.append({
                    "GroupName": right,
                    "MLSupportCount": count,
                    "MLComparedUsers": len(role_peers),
                    "MLConfidence": count / len(role_peers),
                    "NearestUsers": ", ".join(role_peers["SamAccountName"]),
                    "MLMode": "peer_aggregate",
                })

        if not rows:
            return pd.DataFrame()

        out = filter_recommendations_df(pd.DataFrame(rows)).sort_values(
            ["MLConfidence", "MLSupportCount"],
            ascending=False,
        )
        out["MLWorkforcePoolFallback"] = ml_wf_fb
        return out
