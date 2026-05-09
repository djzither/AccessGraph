import pandas as pd

from DataLayer.access_exclusions import filter_group_list, filter_recommendations_df, filter_user_groups_df


class AccessPatternAnalyzer:
    def add_access_patterns(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        recommendations = filter_recommendations_df(recommendations)

        recommendations["AccessPattern"] = recommendations.apply(
            self._classify_pattern,
            axis=1,
        )

        return recommendations

    def _classify_pattern(self, row) -> str:
        count = row["UserCountWithGroup"]
        total = row["TotalUsersInRole"]

        if count == total:
            return "Baseline Access"

        if count == 1:
            return "Unique Access"

        if count == 2:
            return "Rare Access"

        return "Common Access"

    def find_orphaned_access(
        self,
        users_df: pd.DataFrame,
        scope: str = "department",
        min_peer_count: int = 3,
    ) -> pd.DataFrame:
        """
        Scan the full user dataset for groups that are unique or rare within
        a user's cohort — a strong indicator of orphaned or drifted access.

        Parameters
        ----------
        scope : "department" groups by Department only;
                "role" groups by (Title, Department).
        min_peer_count : cohorts smaller than this are skipped to avoid
                         false positives from tiny teams.

        Returns
        -------
        DataFrame with one row per flagged user, sorted by UniqueGroupCount desc.
        Columns: SamAccountName, Title, Department, CohortSize,
                 UniqueGroupCount, RareGroupCount, UniqueGroups, RareGroups.
        """
        df = filter_user_groups_df(users_df)
        group_keys = ["Department"] if scope == "department" else ["Title", "Department"]
        rows = []

        for _, cohort in df.groupby(group_keys):
            cohort_size = len(cohort)
            if cohort_size < min_peer_count:
                continue

            cohort_member_ids = sorted(cohort["SamAccountName"].astype(str).tolist())

            # Count how many cohort members hold each group
            group_counts: dict[str, int] = {}
            for groups in cohort["GroupsList"]:
                for g in filter_group_list(groups):
                    group_counts[g] = group_counts.get(g, 0) + 1

            rare_threshold = max(2, int(cohort_size * 0.10))

            for _, user_row in cohort.iterrows():
                unique_groups = [
                    g for g in filter_group_list(user_row["GroupsList"])
                    if group_counts.get(g, 0) == 1
                ]
                rare_groups = [
                    g for g in filter_group_list(user_row["GroupsList"])
                    if 1 < group_counts.get(g, 0) <= rare_threshold
                ]

                if not unique_groups and not rare_groups:
                    continue

                this_user = str(user_row.get("SamAccountName", ""))
                rows.append({
                    "SamAccountName":   this_user,
                    "Title":            str(user_row.get("Title", "")),
                    "Department":       str(user_row.get("Department", "")),
                    "CohortSize":       cohort_size,
                    "UniqueGroupCount": len(unique_groups),
                    "RareGroupCount":   len(rare_groups),
                    "UniqueGroups":     ", ".join(sorted(unique_groups)),
                    "RareGroups":       ", ".join(sorted(rare_groups)),
                    "CohortMembers":    ", ".join(
                        m for m in cohort_member_ids if m != this_user
                    ),
                })

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .sort_values(
                ["UniqueGroupCount", "RareGroupCount"],
                ascending=False,
            )
            .reset_index(drop=True)
        )
