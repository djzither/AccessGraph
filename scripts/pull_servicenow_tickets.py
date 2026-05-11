"""
scripts/pull_servicenow_tickets.py
─────────────────────────────────────────────────────────────────────────────
Pull "New Employee" CE tickets from ServiceNow and save a local Parquet for
AccessGraph demos.

Environment (PowerShell example):
    $env:SN_USER = "my_username"
    $env:SN_PASS = "my_password"

Optional: ``python-dotenv`` loads a ``.env`` file from the repo root if present.

Usage:
    python -m scripts.pull_servicenow_tickets
    python -m scripts.pull_servicenow_tickets --page-size 200 --out data/raw/custom.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

from DataLayer.servicenow_loader import (
    DEFAULT_SYSPARM_FIELDS,
    DEFAULT_TABLE_API_URL,
    ServiceNowAuthError,
    ServiceNowAPIError,
    pull_new_employee_tickets_normalized,
    ServiceNowTableClient,
)


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


DEFAULT_OUT = Path("data/raw/servicenow_new_employee_tickets.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull New Employee CE tickets from ServiceNow.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_TABLE_API_URL,
        help="Table API URL (default: BYU test CE ticket table)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output Parquet path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="sysparm_limit per request (default: 500)",
    )
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Do not load a .env file even if python-dotenv is installed",
    )
    args = parser.parse_args()

    if not args.no_dotenv:
        _try_load_dotenv()

    try:
        client = ServiceNowTableClient.from_env(table_api_url=args.base_url)
    except ServiceNowAuthError as exc:
        raise SystemExit(str(exc)) from exc

    print("Pulling New Employee CE tickets from ServiceNow...")
    try:
        df = pull_new_employee_tickets_normalized(
            client,
            sysparm_fields=DEFAULT_SYSPARM_FIELDS,
            page_size=args.page_size,
            progress_log=True,
        )
    except (ServiceNowAuthError, ServiceNowAPIError) as exc:
        raise SystemExit(str(exc)) from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    n = len(df)
    print(f"Records pulled: {n}")
    print(f"Output written: {args.out.resolve()}")


if __name__ == "__main__":
    main()
