import pandas as pd

from DataLayer.access_exclusions import filter_group_list, filter_user_groups_df


class PrivilegeAuditAnalyzer:
    """
    Identifies users whose AD group count significantly exceeds their
    role-peer median, which is a strong signal of privilege creep or
    unreviewed historical access accumulation.
    """

    def __init__(self, threshold_multiplier: float = 1.5):
        self.threshold_multiplier = threshold_multiplier

    def analyze(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich users_df with audit columns:
          - GroupCount           : number of clean AD groups
          - RoleMedian           : median GroupCount for the (Title, Department) cohort
          - RolePeerCount        : how many peers share the same role
          - OverprivilegeRatio   : GroupCount / RoleMedian
          - IsOverprivileged     : True when ratio >= threshold_multiplier
          - ExtraGroupCount      : groups above the role median (floored at 0)
        """
        df = filter_user_groups_df(users_df)
        df["GroupCount"] = df["GroupsList"].apply(len)

        role_stats = (
            df.groupby(["Title", "Department"])["GroupCount"]
            .agg(RoleMedian="median", RolePeerCount="count")
            .reset_index()
        )

        df = df.merge(role_stats, on=["Title", "Department"], how="left")
        df["RoleMedian"] = df["RoleMedian"].fillna(0)
        df["RolePeerCount"] = df["RolePeerCount"].fillna(0).astype(int)

        # Avoid division-by-zero; also ignore singleton roles (no real peer comparison)
        safe_median = df["RoleMedian"].clip(lower=1)
        df["OverprivilegeRatio"] = (df["GroupCount"] / safe_median).round(2)
        df["IsOverprivileged"] = (
            (df["OverprivilegeRatio"] >= self.threshold_multiplier)
            & (df["RolePeerCount"] >= 2)
        )
        df["ExtraGroupCount"] = (
            (df["GroupCount"] - df["RoleMedian"]).clip(lower=0).astype(int)
        )

        return df

    def get_flagged_users(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Return flagged users sorted by overprivilege ratio (worst first)."""
        analyzed = self.analyze(users_df)
        flagged = analyzed[analyzed["IsOverprivileged"]].copy()

        output_cols = [
            "SamAccountName", "Title", "Department",
            "GroupCount", "RoleMedian", "ExtraGroupCount",
            "OverprivilegeRatio", "RolePeerCount",
        ]
        output_cols = [c for c in output_cols if c in flagged.columns]

        return (
            flagged[output_cols]
            .sort_values("OverprivilegeRatio", ascending=False)
            .reset_index(drop=True)
        )

    def get_role_summary(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Per-(Title, Department) group-count statistics."""
        df = filter_user_groups_df(users_df)
        df["GroupCount"] = df["GroupsList"].apply(len)

        return (
            df.groupby(["Title", "Department"])
            .agg(
                UserCount=("SamAccountName", "count"),
                MedianGroups=("GroupCount", "median"),
                MaxGroups=("GroupCount", "max"),
                MinGroups=("GroupCount", "min"),
            )
            .reset_index()
            .sort_values("MedianGroups", ascending=False)
            .reset_index(drop=True)
        )
