from __future__ import annotations

import argparse
from pathlib import Path

from DataLayer.access_exclusions import filter_user_groups_df
from DataLayer.cleaner import DataCleaner
from DataLayer.permission_cooccurrence import cooccurrence_with_target


def _load_users(path: Path):
    cleaner = DataCleaner(processed_path=str(path))
    return filter_user_groups_df(cleaner.load_cleaned())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze AD permission co-occurrence from cleaned users (diagnostic only)."
    )
    parser.add_argument(
        "--permission",
        required=True,
        help="Target permission / group name (exact string as in AD).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top co-occurring permissions to show.",
    )
    parser.add_argument(
        "--users",
        default="data/processed/clean_users.parquet",
        help="Path to clean_users.parquet.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Max example NetIDs per row in the overlap column.",
    )
    args = parser.parse_args()

    users_df = _load_users(Path(args.users))
    if users_df.empty:
        print("No users loaded.")
        return

    result = cooccurrence_with_target(
        users_df,
        args.permission,
        top_n=int(args.top),
        max_example_users=int(args.examples),
    )

    print(f"target_permission={args.permission}")
    print(f"users_in_dataset={len(users_df)}")
    if result.empty:
        print("No holders of the target permission found (after filters), or no co-occurring groups.")
        return

    holders = int(result.iloc[0]["users_with_target"])
    print(f"users_with_target={holders}")
    print()
    display = result.rename(
        columns={
            "co_permission": "co_permission",
            "users_with_b": "users_with_B",
            "users_with_both": "users_with_both",
            "p_b_given_a": "P(B|A)",
            "p_a_given_b": "P(A|B)",
            "jaccard": "jaccard",
            "lift": "lift",
            "overlap_pct": "overlap_pct",
            "example_users_overlap": "example_users",
        }
    )
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
