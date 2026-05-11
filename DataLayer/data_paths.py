"""
DataLayer/data_paths.py
─────────────────────────────────────────────────────────────────────────────
DEMO_MODE-aware path resolution for AccessGraph.

Set ACCESSGRAPH_DEMO_MODE=1 (or true/yes/on) in the environment to make the
app load the sanitized demo dataset (data/demo_processed/...) instead of the
real processed dataset (data/processed/...). No code edits required — just
toggle the environment variable, e.g.:

    PowerShell:   $env:ACCESSGRAPH_DEMO_MODE = "1"; streamlit run ProductLayer/app.py
    bash:         ACCESSGRAPH_DEMO_MODE=1 streamlit run ProductLayer/app.py

The sanitized parquets are produced by:
    python -m scripts.build_demo_dataset
"""
from __future__ import annotations

import os
from pathlib import Path

DEMO_ENV_VAR = "ACCESSGRAPH_DEMO_MODE"

REAL_CLEAN_USERS_PATH = Path("data/processed/clean_users.parquet")
REAL_ACCESS_REFERENCE_PATH = Path("data/processed/access_reference.parquet")
REAL_RAW_DATA_DIR = Path("data/raw")

DEMO_CLEAN_USERS_PATH = Path("data/demo_processed/sanitized_clean_users.parquet")
DEMO_ACCESS_REFERENCE_PATH = Path("data/demo_processed/sanitized_access_reference.parquet")


def is_demo_mode() -> bool:
    """True when the ACCESSGRAPH_DEMO_MODE env var is set to a truthy value."""
    return os.environ.get(DEMO_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def clean_users_path() -> Path:
    """Path to the clean users parquet — sanitized when in demo mode."""
    return DEMO_CLEAN_USERS_PATH if is_demo_mode() else REAL_CLEAN_USERS_PATH


def access_reference_path() -> Path:
    """Path to the access reference parquet — sanitized when in demo mode."""
    return DEMO_ACCESS_REFERENCE_PATH if is_demo_mode() else REAL_ACCESS_REFERENCE_PATH


def raw_data_dir() -> Path:
    """
    Raw data directory. In demo mode the raw xlsx sheets are not available
    (and should not be shipped); callers should use access_reference_path()
    instead of re-parsing raw sheets.
    """
    return REAL_RAW_DATA_DIR


def mode_label() -> str:
    """Short human-readable label for the active mode (used by the UI)."""
    return "demo" if is_demo_mode() else "real"
