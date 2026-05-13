"""
Workforce / employee-type classification for AD users and reference rows.

Canonical values are stable for parquet and engine logic; UI and spreadsheets may use labels.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Only org-specific permission used for classification (presence in normalized AD groups).
FULL_TIME_STAFF_AD_GROUP = "a.FULL TIME STAFF"

FULL_TIME = "FULL_TIME"
STUDENT = "STUDENT"
UNKNOWN = "UNKNOWN"

# Future types: add constants and extend _SHEET_LABEL_TO_CANONICAL / UI_MAP


def classify_from_normalized_groups(groups: list[str]) -> str:
    """Return FULL_TIME if the staff marker group is present; otherwise STUDENT."""
    if FULL_TIME_STAFF_AD_GROUP in groups:
        return FULL_TIME
    return STUDENT


def canonical_from_ui_label(label: object) -> str:
    """Map Streamlit / API labels to canonical workforce segment.

    Returns UNKNOWN for any label that does not resolve to a known type.
    Callers that need a fallback default must handle UNKNOWN explicitly —
    this function intentionally does not silently widen cohorts.
    """
    text = "" if label is None else str(label).strip().lower()
    if text in {"full time", "fulltime", "full_time", "staff", "fte"}:
        return FULL_TIME
    if text in {"student", "stu"}:
        return STUDENT
    logger.debug("canonical_from_ui_label: unrecognized label=%r", label)
    return UNKNOWN


def canonical_from_reference_employee_type(value: object) -> str:
    """Map reference-sheet EmployeeType cell to canonical (Full Time / Student / …)."""
    if value is None:
        return UNKNOWN
    text = str(value).strip().lower()
    if not text or text == "nan":
        return UNKNOWN
    if "full" in text and "time" in text:
        return FULL_TIME
    if "student" in text:
        return STUDENT
    return UNKNOWN


def reference_match_value(canonical: str) -> str:
    """
    Value to compare against reference_df['EmployeeTypeClean'] (lowercase stripped labels).
    """
    if canonical == FULL_TIME:
        return "full time"
    if canonical == STUDENT:
        return "student"
    return ""
