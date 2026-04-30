import pandas as pd

from DeterministicLayer.permission_matrix import PermissionMatrixBuilder
from DeterministicLayer.permission_filter import PermissionFilter

class RulesRecommender:
    def __init__(self, min_confidence: float = 0.6):
        self.matrix_builder = PermissionMatrixBuilder(min_confidence=min_confidence)
        self.permission_filter = PermissionFilter()

    def recommend_for_new_user(
            self,
            users_df: pd.DataFrame,
            title: str,
            department: str
    ) -> pd.DataFrame:
        matrix = self.matrix_builder.build_by_title_department(users_df)

        recommendations = self.matrix_builder.recommend_for_role(
            matrix=matrix,
            title=title,
            department=department
        )

        recommendations = self.permission_filter.filter_recommendations(
            recommendations
        )

        recommendations["Source"] = "Title + Department Match"

        return recommendations[
            [
                "GroupName",
                "Source",
                "Confidence",
                "RiskLevel",
                "Decision",
                "UserCountWithGroup",
                "TotalUsersInRole",
            ]
        ]