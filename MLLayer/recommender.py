import pandas as pd

from MLLayer.similarity_model import SimilarityModel


class MLRecommender:
    def __init__(self, users_df: pd.DataFrame):
        self.users_df = users_df.copy()

    def _same_department_pool(
        self,
        department: str,
        include_supervisors: bool = False,
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

        return pool

    def recommend_for_user(
        self,
        sam_account_name: str,
        department: str,
        top_n_users: int = 5,
        min_support: int = 3,
        include_supervisors: bool = False,
    ) -> pd.DataFrame:

        pool = self._same_department_pool(
            department=department,
            include_supervisors=include_supervisors,
        )

        target_user = self.users_df[
            self.users_df["SamAccountName"] == sam_account_name
        ]

        if target_user.empty:
            raise ValueError(f"{sam_account_name} not found in full user data")

        pool = pd.concat([pool, target_user], ignore_index=True)
        pool = pool.drop_duplicates(subset=["SamAccountName"])

        if len(pool) < 2:
            return pd.DataFrame()

        model = SimilarityModel().fit(pool)

        similar_users = model.similar_users(
            sam_account_name=sam_account_name,
            top_n=top_n_users,
        )

        users_by_id = pool.set_index("SamAccountName")

        target_rights = set(
            users_by_id.loc[sam_account_name, "GroupsList"]
        )

        candidate_counts = {}

        for similar_user in similar_users["SamAccountName"]:
            rights = users_by_id.loc[similar_user, "GroupsList"]

            for right in rights:
                if right not in target_rights:
                    candidate_counts[right] = candidate_counts.get(right, 0) + 1

        rows = []

        for right, count in candidate_counts.items():
            if count >= min_support:
                rows.append({
                    "GroupName": right,
                    "MLSupportCount": count,
                    "MLComparedUsers": len(similar_users),
                    "MLConfidence": count / len(similar_users),
                    "NearestUsers": ", ".join(similar_users["SamAccountName"]),
                })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values(
            ["MLConfidence", "MLSupportCount"],
            ascending=False,
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
            for right in rights:
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

        return pd.DataFrame(rows).sort_values(
            ["MLConfidence", "MLSupportCount"],
            ascending=False,
        )

    def recommend_for_peer_cohort(
        self,
        cohort_df: pd.DataFrame,
        min_support: int = 2,
    ) -> pd.DataFrame:

        if cohort_df.empty:
            return pd.DataFrame()

        role_peers = cohort_df.copy()

        if "SamAccountName" in role_peers.columns:
            role_peers = role_peers.sort_values("SamAccountName")

        if len(role_peers) < 2:
            return pd.DataFrame()

        candidate_counts = {}

        for rights in role_peers["GroupsList"]:
            for right in rights:
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

        return pd.DataFrame(rows).sort_values(
            ["MLConfidence", "MLSupportCount"],
            ascending=False,
        )
