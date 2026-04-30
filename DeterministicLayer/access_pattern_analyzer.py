import pandas as pd


class AccessPatternAnalyzer:
    def add_access_patterns(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        recommendations = recommendations.copy()

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