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
        group_lower = group_name.lower()
        return any(keyword in group_lower for keyword in self.sensitive_keywords)

    def should_ignore(self, group_name: str) -> bool:
        group_lower = group_name.lower()
        return any(keyword in group_lower for keyword in self.ignore_keywords)

    def filter_recommendations(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        if recommendations.empty:
            return recommendations

        recommendations = recommendations.copy()

        recommendations = recommendations[
            ~recommendations["GroupName"].apply(self.should_ignore)
        ]

        recommendations["RiskLevel"] = recommendations["GroupName"].apply(
            lambda group: "High" if self.is_sensitive(group) else "Low"
        )

        recommendations["Decision"] = recommendations["RiskLevel"].apply(
            lambda risk: "Manual Review" if risk == "High" else "Recommend"
        )

        return recommendations