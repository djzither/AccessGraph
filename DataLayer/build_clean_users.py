from pathlib import Path
import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from DataLayer.access_exclusions import (
    count_excluded_group_entries,
    count_excluded_reference_rows,
    filter_user_groups_df,
    is_excluded_permission,
)
from DataLayer.cleaner import DataCleaner
from DataLayer.loader import DataLoader
from DataLayer.permission_normalization import normalize_groups_input, summarize_column_values
from DataLayer.rights_sheets_loader import RightsSheetsLoader

logger = logging.getLogger("accessgraph.permissions")


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_RAW_FILE = "ce_ad_user_rights_all.xlsx"
DEFAULT_OUT = Path("data/processed/clean_users.parquet")
DEFAULT_REFERENCE_OUT = Path("data/processed/access_reference.parquet")

REQUIRED_RAW_COLUMNS = {
    "SamAccountName",
    "DisplayName",
    "Title",
    "Department",
    "Manager",
    "Groups",
}

REQUIRED_OUTPUT_COLUMNS = {
    "SamAccountName",
    "DisplayName",
    "Title",
    "Department",
    "Manager",
    "GroupsList",
    "EmployeeType",
}


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["SamAccountName", "DisplayName", "Title", "Department", "Manager"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": "", "None": ""})
            )
    return df


def build_clean_users(
    raw_dir: Path = DEFAULT_RAW_DIR,
    raw_file: str = DEFAULT_RAW_FILE,
    output_path: Path = DEFAULT_OUT,
    reference_output_path: Path = DEFAULT_REFERENCE_OUT,
) -> Path:
    loader = DataLoader(str(raw_dir))
    cleaner = DataCleaner(processed_path=str(output_path))

    raw_df = loader.load_file(raw_file, sheet_name=0)
    raw_rows = len(raw_df)
    if raw_rows == 0:
        raise ValueError("Raw AD export is empty. Refusing to write processed parquet.")

    cleaner.validate_required_columns(raw_df, REQUIRED_RAW_COLUMNS)

    cleaned = cleaner.clean_groups(raw_df)
    cleaned = _normalize_text_columns(cleaned)
    cleaned = cleaned[cleaned["SamAccountName"] != ""].copy()
    excluded_user_group_entries = int(
        sum(
            1
            for groups in raw_df["Groups"].fillna("")
            for group in normalize_groups_input(groups)
            if is_excluded_permission(group)
        )
    )
    perm_stats_raw = summarize_column_values(
        raw_df["Groups"].tolist(),
        context="build_clean_users.raw_AD_Groups",
    )
    groupslist_token_total = int(
        sum(len(x) for x in cleaned["GroupsList"] if isinstance(x, list))
    )
    print(
        "Permission normalization (raw AD Groups): "
        f"raw_segments={perm_stats_raw.total_raw_segments:,} "
        f"dropped_empty_invalid={perm_stats_raw.total_dropped:,} "
        f"rows={perm_stats_raw.rows_processed:,}"
    )
    print(
        "Permission tokens after AD cleaner rules (GroupsList): "
        f"count={groupslist_token_total:,}"
    )
    logger.info(
        "build_clean_users raw_segments=%s dropped_invalid=%s groupslist_tokens=%s",
        perm_stats_raw.total_raw_segments,
        perm_stats_raw.total_dropped,
        groupslist_token_total,
    )
    cleaned = filter_user_groups_df(cleaned)

    cleaner.validate_required_columns(cleaned, REQUIRED_OUTPUT_COLUMNS)

    if cleaned.empty:
        raise ValueError("Cleaned dataframe is empty. Refusing to overwrite parquet.")

    unique_groups = len(
        {
            g
            for groups in cleaned["GroupsList"]
            for g in (groups if isinstance(groups, list) else [])
        }
    )
    users = cleaned["SamAccountName"].nunique()
    zero_group_users = int((cleaned["GroupsList"].apply(len) == 0).sum())
    zero_group_pct = round((zero_group_users / len(cleaned)) * 100, 2)

    cleaner.save_cleaned(cleaned)

    reference_loader = RightsSheetsLoader(raw_path=raw_dir)
    reference_df = reference_loader.load_reference_sheets()
    if reference_df.empty:
        raise ValueError("Parsed access reference dataframe is empty. Refusing to write reference parquet.")
    remaining_reference_crm_rows = count_excluded_reference_rows(reference_df)
    remaining_user_crm_entries = count_excluded_group_entries(cleaned)
    if remaining_reference_crm_rows or remaining_user_crm_entries:
        raise ValueError(
            "CRM exclusion failed: "
            f"{remaining_reference_crm_rows} reference rows and "
            f"{remaining_user_crm_entries} user group entries remain."
        )
    reference_output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_parquet(reference_output_path, index=False)

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "reference_output_path": str(reference_output_path),
        "source_files": [
            str((raw_dir / raw_file).resolve()),
            str((raw_dir / "full_time_employee_access.xlsx").resolve()),
            str((raw_dir / "student_employee_access.xlsx").resolve()),
        ],
        "raw_row_count": int(raw_rows),
        "cleaned_row_count": int(len(cleaned)),
        "reference_row_count": int(len(reference_df)),
        "reference_row_count_by_source": {
            str(key): int(value)
            for key, value in reference_df["SourceFile"].value_counts().items()
        },
        "reference_validation": reference_loader.validation,
        "excluded_crm_user_group_entries": excluded_user_group_entries,
        "remaining_crm_user_group_entries": remaining_user_crm_entries,
        "remaining_crm_reference_rows": remaining_reference_crm_rows,
        "user_count": int(users),
        "unique_group_count": int(unique_groups),
        "zero_group_users": zero_group_users,
        "zero_group_percentage": zero_group_pct,
        "schema_columns": list(cleaned.columns),
        "permission_normalization": {
            "raw_column": "Groups",
            "raw_segments_total": perm_stats_raw.total_raw_segments,
            "dropped_empty_invalid_total": perm_stats_raw.total_dropped,
            "rows_processed": perm_stats_raw.rows_processed,
            "groupslist_token_total_after_cleaner": groupslist_token_total,
        },
        "employee_type_distribution": {
            str(k): int(v)
            for k, v in cleaned["EmployeeType"].value_counts().items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("AccessGraph clean-user build complete")
    print(f"Raw rows: {raw_rows:,}")
    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Users: {users:,}")
    print(f"Unique groups: {unique_groups:,}")
    print(f"Zero-group users: {zero_group_users:,} ({zero_group_pct}%)")
    print(f"Reference rows: {len(reference_df):,}")
    print(f"Excluded CRM user group entries: {excluded_user_group_entries:,}")
    print(f"Remaining CRM user group entries: {remaining_user_crm_entries:,}")
    print(f"Remaining CRM reference rows: {remaining_reference_crm_rows:,}")
    print(f"Wrote: {output_path}")
    print(f"Reference: {reference_output_path}")
    print(f"Metadata: {metadata_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/processed/clean_users.parquet from raw AD export.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory containing raw AD export.")
    parser.add_argument("--raw-file", default=DEFAULT_RAW_FILE, help="Raw AD export filename.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output parquet path.")
    parser.add_argument(
        "--reference-out",
        default=str(DEFAULT_REFERENCE_OUT),
        help="Output parquet path for parsed access reference docs.",
    )
    args = parser.parse_args()

    build_clean_users(
        raw_dir=Path(args.raw_dir),
        raw_file=args.raw_file,
        output_path=Path(args.out),
        reference_output_path=Path(args.reference_out),
    )


if __name__ == "__main__":
    main()
