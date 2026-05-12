"""
scripts/build_demo_ticket_dataset.py
─────────────────────────────────────────────────────────────────────────────
Build ``data/processed/demo_servicenow_tickets.parquet`` from exported CE
ticket PDFs (temporary stand-in for ServiceNow Table API ingestion).

Layers (mirror future API pipeline):
    PDF files → pdf_ticket_parser (extract + parse)
             → identity normalization (sanitized demo users + mapping CSVs)
             → optional signal columns (door/onboarding flags)
             → Parquet

Usage:
    python -m scripts.build_demo_ticket_dataset
    python -m scripts.build_demo_ticket_dataset --analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataLayer.pdf_ticket_parser import (  # noqa: E402
    DEFAULT_DEMO_PDF_DIR,
    DEFAULT_MAPPING_DIR,
    DEFAULT_SANITIZED_USERS_PATH,
    analyze_demo_tickets_example_report,
    analyze_demo_tickets_for_access_signals,
    build_ticket_dataframe,
    log_validation_summary,
    validate_ticket_dataframe,
)

DEFAULT_OUT = Path("data/processed/demo_servicenow_tickets.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse CE ticket PDFs into a demo ServiceNow-equivalent Parquet.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_DEMO_PDF_DIR,
        help="Directory containing exported *.pdf CE tickets",
    )
    parser.add_argument(
        "--users",
        type=Path,
        default=DEFAULT_SANITIZED_USERS_PATH,
        help="Sanitized demo users parquet for identity resolution",
    )
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=DEFAULT_MAPPING_DIR,
        help="Directory with user_map.csv / person_name_map.csv (from build_demo_dataset)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output Parquet path",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Print the demo access/onboarding analysis report",
    )
    parser.add_argument(
        "--no-signal-columns",
        action="store_true",
        help="Omit _demo_* exploratory columns from the Parquet",
    )
    args = parser.parse_args()

    df = build_ticket_dataframe(
        args.pdf_dir,
        sanitized_users_path=args.users,
        mapping_dir=args.mapping_dir,
    )

    out_df = df if args.no_signal_columns else analyze_demo_tickets_for_access_signals(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    stats = validate_ticket_dataframe(df)
    log_validation_summary(stats)

    print(f"Output written: {args.out.resolve()}")
    print(f"Rows: {len(out_df)}")

    if args.analysis:
        analyze_demo_tickets_example_report(df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
