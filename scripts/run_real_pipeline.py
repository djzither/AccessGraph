from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DataLayer.build_clean_users import build_clean_users
from DataLayer.rights_sheets_loader import RightsSheetsLoader


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent

    # ── Step 1: AD user export → clean_users.parquet ─────────────────────
    print("── Step 1: AD user pipeline ──")
    build_clean_users(
        raw_dir=base_dir / "data" / "raw",
        raw_file="ce_ad_user_rights_all.xlsx",
        output_path=base_dir / "data" / "processed" / "clean_users.parquet",
    )

    # ── Step 2: Reference access sheets → reference_access.parquet ───────
    print("\n── Step 2: Reference sheet pipeline ──")
    loader = RightsSheetsLoader(raw_path=str(base_dir / "data" / "raw"))
    ref_df = loader.load_reference_sheets()

    out_path = base_dir / "data" / "processed" / "reference_access.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_df.to_parquet(out_path, index=False)
    print(f"\n[reference] Saved {len(ref_df):,} rows → {out_path}")


if __name__ == "__main__":
    main()
