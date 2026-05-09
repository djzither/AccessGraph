import argparse
from pathlib import Path
import pandas as pd

from DataLayer.access_exclusions import filter_group_list


def audit_clean_users(data_path: Path) -> None:
    df = pd.read_parquet(data_path)
    if "GroupsList" not in df.columns:
        raise ValueError("GroupsList missing from dataset.")

    groups = df["GroupsList"].apply(
        filter_group_list
    )
    group_counts = groups.apply(len)
    all_groups = pd.Series([g for lst in groups for g in lst], dtype="object")
    top = all_groups.value_counts().head(25)

    zero = int((group_counts == 0).sum())
    total = len(df)
    print("AccessGraph clean-user audit")
    print(f"Data path: {data_path}")
    print(f"Total users: {total:,}")
    print(f"Zero-group users: {zero:,} ({(zero / max(total, 1)) * 100:.2f}%)")
    print(f"Missing Title: {(df['Title'].astype(str).str.strip() == '').sum():,}")
    print(f"Missing Department: {(df['Department'].astype(str).str.strip() == '').sum():,}")
    print("\nGroup count distribution:")
    print(group_counts.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    print("\nTop 25 groups:")
    print(top.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit clean_users parquet quality.")
    parser.add_argument("--data", default="data/processed/clean_users.parquet")
    args = parser.parse_args()
    audit_clean_users(Path(args.data))


if __name__ == "__main__":
    main()
