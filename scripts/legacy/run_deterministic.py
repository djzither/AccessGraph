"""DEPRECATED: legacy raw-data deterministic pipeline.

Use `python -m DataLayer.build_clean_users` and parquet-based product flows instead.
"""

from pathlib import Path

import pandas as pd

from DataLayer.cleaner import DataCleaner
from DeterministicLayer.rules_recommender import RulesRecommender
from DataLayer.rights_sheets_loader import RightsSheetsLoader
from DeterministicLayer.reference_matcher import ReferenceMatcher
from DeterministicLayer.access_pattern_analyzer import AccessPatternAnalyzer


def get_users_with_group(group_name, cohort_df):
    return cohort_df[
        cohort_df["GroupsList"].apply(lambda groups: group_name in groups)
    ][["SamAccountName", "DisplayName", "Title", "Department"]]


def get_users_without_group(group_name, cohort_df):
    return cohort_df[
        ~cohort_df["GroupsList"].apply(lambda groups: group_name in groups)
    ][["SamAccountName", "DisplayName", "Title", "Department"]]


def print_group_breakdown(results_subset, cohort, title):
    print(f"\n{title}")

    if results_subset.empty:
        print("None found.")
        return

    for _, result_row in results_subset.iterrows():
        group_name = result_row["GroupName"]
        count = result_row["UserCountWithGroup"]
        total = result_row["TotalUsersInRole"]
        score = result_row["Score"]
        score_pct = result_row["ScorePct"]
        decision = result_row.get("FinalDecision", result_row["Decision"])
        access_pattern = result_row.get("AccessPattern", "")
        reason = result_row.get("Reason", "")
        sheet_match = result_row.get("ReferenceSheetMatch", False)

        users_with = get_users_with_group(group_name, cohort)
        users_without = get_users_without_group(group_name, cohort)

        print("\n" + "=" * 80)
        print(f"Group: {group_name}")
        print(f"Match: {count}/{total}")
        print(f"Score: {score} ({score_pct})")
        print(f"Decision: {decision}")
        print(f"Access Pattern: {access_pattern}")
        print(f"Reference Sheet Match: {sheet_match}")

        if reason:
            print(f"Reason: {reason}")

        print("\nUsers WITH this group:")
        if users_with.empty:
            print("None")
        else:
            print(users_with[["SamAccountName", "DisplayName"]].to_string(index=False))

        print("\nUsers WITHOUT this group:")
        if users_without.empty:
            print("None")
        else:
            print(users_without[["SamAccountName", "DisplayName"]].to_string(index=False))


def main():
    print("=== DEPRECATED: run_deterministic.py (legacy raw-data path) ===")
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "raw"

    cleaner = DataCleaner()

    df = pd.read_excel(data_path / "ce_ad_user_rights_all.xlsx")
    df = cleaner.clean_groups(df)

    row = df.dropna(subset=["Title", "Department"]).iloc[0]

    title = row["Title"]
    department = row["Department"]

    print("Using:")
    print(f"Title: {title}")
    print(f"Department: {department}")
    print()

    cohort = df[
        (df["Title"] == title)
        & (df["Department"] == department)
    ]

    print("Users being compared:")
    print(
        cohort[
            ["SamAccountName", "DisplayName", "Title", "Department"]
        ].to_string(index=False)
    )

    recommender = RulesRecommender(min_confidence=0.0)

    results = recommender.recommend_for_new_user(
        users_df=df,
        title=title,
        department=department,
    )

    if results.empty:
        print("\nNo groups found.")
        return

    results["Score"] = (
        results["UserCountWithGroup"] / results["TotalUsersInRole"]
    ).round(3)

    results["ScorePct"] = (
        (results["Score"] * 100).round(1).astype(str) + "%"
    )

    pattern_analyzer = AccessPatternAnalyzer()
    results = pattern_analyzer.add_access_patterns(results)

    rights_loader = RightsSheetsLoader(raw_path=data_path)
    reference_df = rights_loader.load_reference_sheets()

    matcher = ReferenceMatcher(reference_df)

    results = matcher.match_recommendations(
        recommendations=results,
        title=title,
        department=department,
        employee_type="Full Time",
        supervisor=None,
    )

    display_cols = [
        "GroupName",
        "UserCountWithGroup",
        "TotalUsersInRole",
        "Score",
        "ScorePct",
        "AccessPattern",
        "Confidence",
        "RiskLevel",
        "Decision",
        "ReferenceSheetMatch",
        "ReferenceCategories",
        "FinalDecision",
        "Reason",
    ]

    print("\nTop groups:")
    print(results[display_cols].head(30).to_string(index=False))

    print("\nSummary of match counts:")
    print(results["UserCountWithGroup"].value_counts().sort_index(ascending=False))

    exact_4s_3s = results[
        (results["UserCountWithGroup"].isin([3, 4]))
        & (results["TotalUsersInRole"] >= 5)
    ].sort_values(
        by=["UserCountWithGroup", "Score"],
        ascending=[False, False],
    )

    print("\nExact 4s and 3s:")
    if exact_4s_3s.empty:
        print("None found.")
    else:
        print(exact_4s_3s[display_cols].head(50).to_string(index=False))

    low_1s_2s = results[
        results["UserCountWithGroup"].isin([1, 2])
    ].sort_values(
        by=["UserCountWithGroup", "Score"],
        ascending=[False, False],
    )

    print("\nExact 1s and 2s:")
    if low_1s_2s.empty:
        print("None found.")
    else:
        print(low_1s_2s[display_cols].head(50).to_string(index=False))

    print_group_breakdown(
        results_subset=exact_4s_3s.head(25),
        cohort=cohort,
        title="Detailed breakdown for 4s and 3s",
    )

    print_group_breakdown(
        results_subset=low_1s_2s.head(25),
        cohort=cohort,
        title="Detailed breakdown for 1s and 2s",
    )

    print("\nFull breakdown by match count:")
    for count in sorted(results["UserCountWithGroup"].unique(), reverse=True):
        subset = results[results["UserCountWithGroup"] == count]

        print("\n" + "-" * 80)
        print(f"Groups found in {count}/{subset['TotalUsersInRole'].iloc[0]} users")
        print(subset[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
