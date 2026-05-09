from __future__ import annotations

import argparse
from pathlib import Path

from DataLayer.access_exclusions import filter_reference_df, filter_user_groups_df
from DataLayer.cleaner import DataCleaner
from DataLayer.rights_sheets_loader import RightsSheetsLoader
from DataLayer.subgroup_detection import analyze_recommendation_subgroups
from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


def _load_users(path: Path):
    users = DataCleaner(processed_path=str(path)).load_cleaned()
    return filter_user_groups_df(users)


def _load_reference(raw_dir: Path):
    return filter_reference_df(RightsSheetsLoader(raw_path=raw_dir).load_reference_sheets())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect permission-based subgroups inside an engine-selected cohort."
    )
    parser.add_argument("--title", required=True, help="Target title.")
    parser.add_argument("--department", required=True, help="Target department.")
    parser.add_argument("--employee-type", default="Full Time", help="Employee type passed to engine.")
    parser.add_argument("--supervisor", default=None, help="Supervisor value passed to engine.")
    parser.add_argument("--copy-from", dest="copy_from", default=None, help="Copy-from NetID.")
    parser.add_argument("--new-hire", dest="new_hire", default=None, help="New-hire NetID.")
    parser.add_argument("--users", default="data/processed/clean_users.parquet", help="Users parquet path.")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw directory containing reference sheets.")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="Engine AD confidence floor.")
    args = parser.parse_args()

    users_df = _load_users(Path(args.users))
    reference_df = _load_reference(Path(args.raw_dir))
    engine = AccessRecommendationEngine(min_confidence=float(args.min_confidence))

    reference_recs = engine._get_reference_recommendations(
        reference_df=reference_df,
        title=args.title,
        department=args.department,
        employee_type=args.employee_type,
        supervisor=args.supervisor,
        users_df=users_df,
        copy_from_netid=args.copy_from,
    )
    comparison_cohort = engine._select_ad_comparison_cohort(
        users_df=users_df,
        title=args.title,
        department=args.department,
        reference_recs=reference_recs,
        employee_type=args.employee_type,
        copy_from_netid=args.copy_from,
    )

    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title=args.title,
        department=args.department,
        employee_type=args.employee_type,
        supervisor=args.supervisor,
        copy_from_netid=args.copy_from,
        new_hire_netid=args.new_hire,
    )

    report = analyze_recommendation_subgroups(
        comparison_cohort=comparison_cohort,
        recommendations_df=recommendations,
    )

    print(f"title={args.title}")
    print(f"department={args.department}")
    print(f"broad_cohort_size={len(comparison_cohort)}")
    print()
    if report.empty:
        print("No recommendation permissions found for subgroup detection.")
        return

    cols = [
        "permission",
        "broad_cohort_size",
        "users_with_permission",
        "users_without_permission",
        "with_shared_permissions",
        "without_shared_permissions",
        "strongest_subgroup_indicators",
        "subgroup_assessment",
    ]
    print(report[cols].to_string(index=False))


if __name__ == "__main__":
    main()
