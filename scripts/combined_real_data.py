from pathlib import Path
import pandas as pd

from DataLayer.cleaner import DataCleaner
from ProductLayer.hybrid_recommender import HybridRecommender


def main():
    project_root = Path(__file__).resolve().parents[1]
    cleaned_path = project_root / "data" / "processed" / "clean_users.parquet"

    cleaner = DataCleaner(processed_path=str(cleaned_path))
    users_df = cleaner.load_cleaned()

    print("Loaded users:", len(users_df))
    print(users_df[["SamAccountName", "DisplayName", "Title", "Department"]].head())

    target_user = "snelson8"

    target_row = users_df[users_df["SamAccountName"] == target_user]

    if target_row.empty:
        raise ValueError(f"{target_user} not found")

    title = target_row.iloc[0]["Title"]
    department = target_row.iloc[0]["Department"]

    print("\nTesting user:")
    print("SamAccountName:", target_user)
    print("Title:", title)
    print("Department:", department)

    # Simulate this user as a new hire with no current rights
    users_df_test = users_df.copy()

    users_df_test.loc[
        users_df_test["SamAccountName"] == target_user,
        "GroupsList"
    ] = users_df_test.loc[
        users_df_test["SamAccountName"] == target_user,
        "GroupsList"
    ].apply(lambda _: [])

    recommender = HybridRecommender(min_rules_confidence=0.6)

    results = recommender.recommend(
        users_df=users_df_test,
        sam_account_name=target_user,
        title=title,
        department=department,
        top_n_users=5,
        min_ml_support=1,
        include_supervisors=False,
    )
    actual_groups = set(
        users_df.loc[
            users_df["SamAccountName"] == target_user,
            "GroupsList"
        ].iloc[0]
    )

    recommended_groups = set(results["GroupName"])

    correct = actual_groups.intersection(recommended_groups)

    missing = recommended_groups - actual_groups
    extra = actual_groups - recommended_groups

    print("\n=== MODEL-BASED ACCESS AUDIT ===")
    print(f"Actual groups user has: {len(actual_groups)}")
    print(f"Groups model recommends: {len(recommended_groups)}")
    print(f"Correct / agreed groups: {len(correct)}")
    print(f"Missing according to model: {len(missing)}")
    print(f"Extra according to model: {len(extra)}")

    print("\nMissing access according to model:")
    print(sorted(list(missing))[:30])

    print("\nExtra access according to model:")
    print(sorted(list(extra))[:30])

    print("\nTop recommendations:")
    print(
        results[
            [
                "GroupName",
                "FinalDecision",
                "HybridScore",
                "RulesConfidence",
                "MLConfidence",
                "RiskLevel",
                "Reason",
            ]
        ].head(50).to_string(index=False)
    )


if __name__ == "__main__":
    main()