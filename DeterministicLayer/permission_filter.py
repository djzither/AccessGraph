import pandas as pd


class PermissionFilter:
    def __init__(self):
        self.sensitive_keywords = [
            "admin",
            "payroll",
            "finance",
            "hr",
            "superuser",
            "domain",
            "security",
            "privileged",
            "owner",
        ]

        self.ignore_keywords = [
            "cannot find an object",
            "deprecated",
            "disabled",
        ]

    def is_sensitive(self, group_name: str) -> bool:
        group_lower = str(group_name).lower()
        return any(keyword in group_lower for keyword in self.sensitive_keywords)

    def should_ignore(self, group_name: str) -> bool:
        group_lower = str(group_name).lower()
        return any(keyword in group_lower for keyword in self.ignore_keywords)

    def filter_recommendations(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        recommendations = recommendations.copy()

        if recommendations.empty:
            recommendations["RiskLevel"] = []
            recommendations["Decision"] = []
            return recommendations

        # Remove junk groups
        recommendations = recommendations[
            ~recommendations["GroupName"].apply(self.should_ignore)
        ].copy()

        # Risk level
        recommendations["RiskLevel"] = recommendations["GroupName"].apply(
            lambda group: "High" if self.is_sensitive(group) else "Low"
        )

        # ✅ Compute score if not already present
        if "Score" not in recommendations.columns:
            recommendations["Score"] = (
                    recommendations["UserCountWithGroup"]
                    / recommendations["TotalUsersInRole"]
            ).round(3)

        # 🔥 New decision logic (this fixes your issue)
        def assign_decision(row):
            score = row["Score"]
            risk = row["RiskLevel"]

            if risk == "High":
                return "Manual Review"

            if score == 1.0:
                return "Auto Assign"
            elif score >= 0.8:
                return "Strong Recommend"
            elif score >= 0.6:
                return "Suggest"
            elif score >= 0.4:
                return "Low Confidence"
            else:
                return "This is probably an extra right"

        recommendations["Decision"] = recommendations.apply(assign_decision, axis=1)

        return recommendations