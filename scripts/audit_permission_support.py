from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from DataLayer.access_exclusions import filter_group_list, filter_reference_df, filter_user_groups_df
from DataLayer.cleaner import DataCleaner
from DataLayer.rights_sheets_loader import RightsSheetsLoader
from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


def _norm(value: object) -> str:
    return str(value).strip().lower()


def _contains_keyword(groups: list[str], keyword: str) -> list[str]:
    key = _norm(keyword)
    return [g for g in groups if key in _norm(g)]


def _print_section(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def _load_users(path: Path) -> pd.DataFrame:
    users = DataCleaner(processed_path=str(path)).load_cleaned()
    return filter_user_groups_df(users)


def _load_reference(raw_dir: Path) -> pd.DataFrame:
    return filter_reference_df(RightsSheetsLoader(raw_path=raw_dir).load_reference_sheets())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit permission support (AD/ML/copy-from/reference) for one target permission."
    )
    parser.add_argument("--permission", required=True, help="Target permission name (exact text).")
    parser.add_argument("--copy-from", dest="copy_from", default=None, help="Copy-from NetID.")
    parser.add_argument("--new-hire", dest="new_hire", default=None, help="New-hire NetID (optional).")
    parser.add_argument("--title", default=None, help="Target title. Defaults to copy-from user's title.")
    parser.add_argument("--department", default=None, help="Target department. Defaults to copy-from user's department.")
    parser.add_argument("--employee-type", default="Full Time", help="Employee type passed to engine.")
    parser.add_argument("--supervisor", default=None, help="Supervisor value passed to engine.")
    parser.add_argument("--users", default="data/processed/clean_users.parquet", help="Users parquet path.")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data dir for reference sheets.")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="Engine AD confidence floor.")
    parser.add_argument("--keyword", default=None, help="Optional keyword for substring diagnostics.")
    args = parser.parse_args()

    users_df = _load_users(Path(args.users))
    reference_df = _load_reference(Path(args.raw_dir))
    engine = AccessRecommendationEngine(min_confidence=float(args.min_confidence))

    copy_from = args.copy_from
    target_user = None
    if copy_from:
        hit = users_df[users_df["SamAccountName"] == copy_from]
        if not hit.empty:
            target_user = hit.iloc[0]

    title = args.title or (str(target_user["Title"]) if target_user is not None else "")
    department = args.department or (str(target_user["Department"]) if target_user is not None else "")
    if not title or not department:
        raise ValueError("Could not infer title/department. Provide --title and --department.")

    ui_exact = users_df[
        users_df["Title"].astype(str).str.strip().eq(str(title).strip())
        & users_df["Department"].astype(str).str.strip().eq(str(department).strip())
    ].copy()

    reference_recs = engine._get_reference_recommendations(
        reference_df=reference_df,
        title=title,
        department=department,
        employee_type=args.employee_type,
        supervisor=args.supervisor,
        users_df=users_df,
        copy_from_netid=copy_from,
    )
    comparison_cohort = engine._select_ad_comparison_cohort(
        users_df=users_df,
        title=title,
        department=department,
        reference_recs=reference_recs,
        employee_type=args.employee_type,
        copy_from_netid=copy_from,
    )
    ad_recs = engine._get_ad_recommendations(comparison_cohort=comparison_cohort)
    ml_recs = engine._get_ml_recommendations(
        users_df=users_df,
        new_hire_netid=args.new_hire,
        department=department,
        comparison_cohort=comparison_cohort,
    )
    copy_recs = engine._get_copy_from_recommendations(users_df=users_df, copy_from_netid=copy_from)
    final = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title=title,
        department=department,
        employee_type=args.employee_type,
        supervisor=args.supervisor,
        copy_from_netid=copy_from,
        new_hire_netid=args.new_hire,
    )

    permission = args.permission
    keyword = args.keyword or permission
    permission_norm = engine._normalize_group_name(permission)

    _print_section("Request Context")
    print(f"permission={permission}")
    print(f"permission_norm={permission_norm}")
    print(f"keyword={keyword}")
    print(f"title={title}")
    print(f"department={department}")
    print(f"employee_type={args.employee_type}")
    print(f"supervisor={args.supervisor}")
    print(f"copy_from={copy_from}")
    print(f"new_hire={args.new_hire}")

    _print_section("Cohort Comparison")
    print(f"ui_exact_title_department_count={len(ui_exact)}")
    print(f"engine_comparison_cohort_count={len(comparison_cohort)}")
    print("engine_cohort_users=" + ", ".join(comparison_cohort["SamAccountName"].astype(str).tolist()))

    _print_section("Peer Group Diagnostics")
    ad_numerator = 0
    for _, peer in comparison_cohort.iterrows():
        raw_groups = peer.get("GroupsList")
        parsed_groups = filter_group_list(raw_groups)
        keyword_hits = _contains_keyword(parsed_groups, keyword)
        exact_match = permission in parsed_groups
        if exact_match:
            ad_numerator += 1
        print(f"- peer={peer.get('SamAccountName','')}")
        print(f"  raw_groups={raw_groups}")
        print(f"  parsed_keyword_hits={keyword_hits}")
        print(f"  exact_match={exact_match}")
    print(f"ad_support_manual={ad_numerator}/{len(comparison_cohort)}")

    _print_section("Signal Rows For Permission")
    ad_hit = ad_recs[ad_recs["GroupName"].apply(lambda x: engine._normalize_group_name(x) == permission_norm)]
    ml_hit = ml_recs[ml_recs["GroupName"].apply(lambda x: engine._normalize_group_name(x) == permission_norm)]
    copy_hit = copy_recs[copy_recs["GroupName"].apply(lambda x: engine._normalize_group_name(x) == permission_norm)]
    ref_hit = reference_recs[reference_recs["GroupName"].apply(lambda x: engine._normalize_group_name(x) == permission_norm)] if not reference_recs.empty else reference_recs

    print("reference_match_rows=" + str(len(ref_hit)))
    print(ref_hit.to_string(index=False) if not ref_hit.empty else "(none)")
    print("ad_rows=" + str(len(ad_hit)))
    print(ad_hit.to_string(index=False) if not ad_hit.empty else "(none)")
    print("ml_rows=" + str(len(ml_hit)))
    print(ml_hit.to_string(index=False) if not ml_hit.empty else "(none)")
    print("copy_from_rows=" + str(len(copy_hit)))
    print(copy_hit.to_string(index=False) if not copy_hit.empty else "(none)")

    _print_section("Final Recommendation Row(s)")
    final_hit = final[final["GroupName"].apply(lambda x: engine._normalize_group_name(x) == permission_norm)]
    if final_hit.empty:
        print("No final recommendation row matched normalized permission.")
        keyword_rows = final[final["GroupName"].astype(str).str.contains(keyword, case=False, na=False)]
        print("Keyword-matching final rows:")
        print(keyword_rows.to_string(index=False) if not keyword_rows.empty else "(none)")
    else:
        cols = [
            c for c in [
                "GroupName",
                "InReferenceSheet",
                "UserCountWithGroup",
                "TotalUsersInRole",
                "ADConfidence",
                "MLSupportCount",
                "MLComparedUsers",
                "MLConfidence",
                "CopyFromUserHasIt",
                "CopyFromNetID",
                "FinalScore",
                "FinalDecision",
                "Reason",
            ]
            if c in final_hit.columns
        ]
        print(final_hit[cols].to_string(index=False))


if __name__ == "__main__":
    main()

