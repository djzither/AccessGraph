import pandas as pd
from collections import Counter

from DataLayer.access_exclusions import filter_group_list, filter_recommendations_df


class PermissionMatrixBuilder:
    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def build_by_title_department(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        grouped = df.groupby(["Title", "Department"], dropna=False)

        for (title, department), group in grouped:
            total_users = len(group)
            counter = Counter()

            for groups in group["GroupsList"]:
                counter.update(filter_group_list(groups))

            for group_name, count in counter.items():
                confidence = count / total_users

                rows.append({
                    "Title": title,
                    "Department": department,
                    "GroupName": group_name,
                    "UserCountWithGroup": count,
                    "TotalUsersInRole": total_users,
                    "Confidence": round(confidence, 3),
                })

        matrix = pd.DataFrame(rows)

        if matrix.empty:
            return matrix

        matrix = filter_recommendations_df(matrix)

        return matrix.sort_values(
            by=["Title", "Department", "Confidence"],
            ascending=[True, True, False],
        )

    def recommend_for_role(
        self,
        matrix: pd.DataFrame,
        title: str,
        department: str
    ) -> pd.DataFrame:
        matches = matrix[
            (matrix["Title"] == title) &
            (matrix["Department"] == department) &
            (matrix["Confidence"] >= self.min_confidence)
        ].copy()

        matches = filter_recommendations_df(matches)
        return matches.sort_values(by="Confidence", ascending=False)
