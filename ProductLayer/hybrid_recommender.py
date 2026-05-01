import pandas as pd

from DeterministicLayer.rules_recommender import RulesRecommender
from MLLayer.recommender import MLRecommender


class HybridRecommender:
    def __init__(self, min_rules_confidence: float = 0.6):
        self.rules_recommender = RulesRecommender(
            min_confidence=min_rules_confidence
        )

    def recommend(
        self,
        users_df: pd.DataFrame,
        sam_account_name: str,
        title: str,
        department: str,
        top_n_users: int = 5,
        min_ml_support: int = 2,
        include_supervisors: bool = False,
    ) -> pd.DataFrame:

        rules_results = self.rules_recommender.recommend_for_new_user(
            users_df=users_df,
            title=title,
            department=department,
        )

        ml_results = MLRecommender(users_df).recommend_for_user(
            sam_account_name=sam_account_name,
            department=department,
            top_n_users=top_n_users,
            min_support=min_ml_support,
            include_supervisors=include_supervisors,
        )

        final = self._combine_results(rules_results, ml_results)
        final = self._assign_final_decisions(final)

        return final.sort_values(
            by=["FinalPriority", "RulesConfidence", "MLConfidence"],
            ascending=[True, False, False],
        )

    def _combine_results(
        self,
        rules_results: pd.DataFrame,
        ml_results: pd.DataFrame,
    ) -> pd.DataFrame:

        rules = rules_results.copy()
        ml = ml_results.copy()

        if rules.empty:
            rules = pd.DataFrame(columns=[
                "GroupName",
                "Source",
                "Confidence",
                "RiskLevel",
                "Decision",
                "UserCountWithGroup",
                "TotalUsersInRole",
            ])

        if ml.empty:
            ml = pd.DataFrame(columns=[
                "GroupName",
                "MLSupportCount",
                "MLComparedUsers",
                "MLConfidence",
                "NearestUsers",
            ])

        rules = rules.rename(columns={
            "Confidence": "RulesConfidence",
            "Decision": "RulesDecision",
            "Source": "RulesSource",
        })

        combined = pd.merge(
            rules,
            ml,
            on="GroupName",
            how="outer",
        )

        combined["RulesConfidence"] = combined["RulesConfidence"].fillna(0)
        combined["MLConfidence"] = combined["MLConfidence"].fillna(0)
        combined["MLSupportCount"] = combined["MLSupportCount"].fillna(0)
        combined["MLComparedUsers"] = combined["MLComparedUsers"].fillna(0)

        combined["RiskLevel"] = combined["RiskLevel"].fillna("Low")
        combined["RulesDecision"] = combined["RulesDecision"].fillna("Not Found By Rules")
        combined["RulesSource"] = combined["RulesSource"].fillna("ML Similar Users")

        return combined

    def _assign_final_decisions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["FoundByRules"] = df["RulesConfidence"] > 0
        df["FoundByML"] = df["MLConfidence"] > 0

        df["HybridScore"] = (
            0.65 * df["RulesConfidence"]
            + 0.35 * df["MLConfidence"]
        ).round(3)

        df["FinalDecision"] = df.apply(self._decision_logic, axis=1)
        df["FinalPriority"] = df["FinalDecision"].map({
            "Auto Assign": 1,
            "Strong Recommend": 2,
            "Suggest": 3,
            "Manual Review": 4,
            "Low Confidence": 5,
            "Ignore": 6,
        }).fillna(99)

        df["Reason"] = df.apply(self._reason, axis=1)

        return df

    def _decision_logic(self, row) -> str:
        risk = row["RiskLevel"]
        rules_conf = row["RulesConfidence"]
        ml_conf = row["MLConfidence"]
        found_by_rules = row["FoundByRules"]
        found_by_ml = row["FoundByML"]

        if risk == "High":
            return "Manual Review"

        if found_by_rules and found_by_ml:
            if rules_conf >= 0.8 and ml_conf >= 0.6:
                return "Strong Recommend"
            return "Suggest"

        if found_by_rules:
            if rules_conf == 1.0:
                return "Auto Assign"
            if rules_conf >= 0.8:
                return "Strong Recommend"
            if rules_conf >= 0.6:
                return "Suggest"

        if found_by_ml:
            if ml_conf >= 0.8:
                return "Suggest"
            if ml_conf >= 0.5:
                return "Low Confidence"

        return "Ignore"

    def _reason(self, row) -> str:
        rules_conf = row["RulesConfidence"]
        ml_conf = row["MLConfidence"]
        support = int(row["MLSupportCount"])
        compared = int(row["MLComparedUsers"])

        if row["RiskLevel"] == "High":
            return "Sensitive permission. Requires manual approval."

        if row["FoundByRules"] and row["FoundByML"]:
            return (
                f"Found by title/department rules with {rules_conf:.0%} confidence "
                f"and supported by ML similarity in {support}/{compared} nearest users."
            )

        if row["FoundByRules"]:
            return f"Found by title/department rules with {rules_conf:.0%} confidence."

        if row["FoundByML"]:
            return f"Not found by rules, but appeared in {support}/{compared} similar users."

        return "No strong evidence."