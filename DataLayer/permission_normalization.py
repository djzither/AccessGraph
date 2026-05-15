"""
Centralized permission string normalization for AccessGraph.

All permission tokens (AD groups, reference access names, exploded rows) pass
through this module so empty/whitespace/NaN-like values and noisy delimiters
are handled consistently.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

# NaN-like string tokens (after strip / lower)
_INVALID_STRING_TOKENS = frozenset(
    {"nan", "none", "null", "#n/a", "n/a", "nat", ""}
)


def normalize_single_permission(value: object) -> str | None:
    """
    Normalize a single permission token: strip ends, drop NaN-like / empty.

    Does not change inner spelling or case (downstream matchers own case folding).
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = value.strip() if isinstance(value, str) else str(value).strip()
    if not text or text.lower() in _INVALID_STRING_TOKENS:
        return None
    return text


def _normalize_delimited_string(value: str) -> list[str]:
    import re
    collapsed = re.sub(r";+", ";", value.strip())
    if not collapsed or collapsed.lower() in _INVALID_STRING_TOKENS:
        return []
    return [t for p in collapsed.split(";") if (t := normalize_single_permission(p))]


def _normalize_from_iterable(items: Iterable[object]) -> list[str]:
    return [t for item in items if (t := normalize_single_permission(item))]


def normalize_groups_input(value: object) -> list[str]:
    """
    Normalize a GroupsList cell value: semicolon string, list/tuple/set, or numpy array.

    Removes empty segments from duplicate delimiters (e.g. 'a;;b' -> ['a', 'b']).
    """
    if value is None:
        return []

    if isinstance(value, str):
        return _normalize_delimited_string(value)

    if isinstance(value, (list, tuple, set)):
        return _normalize_from_iterable(value)

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            as_list = value.tolist()
            if isinstance(as_list, list):
                return _normalize_from_iterable(as_list)
        except Exception:
            pass

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    return _normalize_delimited_string(str(value))


_PERMISSION_PREFIXES = ("m.", "i.", "dce.", "dce-", "dce ")


def canonical_permission_id(value: object) -> str:
    """
    Stable comparison key for AD groups and reference access names.

    Collapses variant prefixes and punctuation so e.g. CMP.AllUsers and
    DCE.CMP.Allusers compare equal.
    """
    base = normalize_single_permission(value)
    text = str(base).lower().strip() if base else ""
    for prefix in _PERMISSION_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return re.sub(r"[\s._-]+", "", text)


@dataclass(frozen=True)
class PermissionCanonical:
    raw_permission_name: str
    canonical_permission_id: str
    source: str = "unknown"


def canonicalize_permission(value: object, *, source: str = "unknown") -> PermissionCanonical:
    raw = normalize_single_permission(value) or ""
    return PermissionCanonical(
        raw_permission_name=raw,
        canonical_permission_id=canonical_permission_id(value),
        source=source,
    )
