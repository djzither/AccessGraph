from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DataLayer.build_clean_users import build_clean_users


def main() -> None:
    build_clean_users(
        raw_dir=Path("data/raw"),
        raw_file="ce_ad_user_rights_all.xlsx",
        output_path=Path("data/processed/clean_users.parquet"),
    )


if __name__ == "__main__":
    main()

