"""
Centralized permission string normalization for AccessGraph.

All permission tokens (AD groups, reference access names, exploded rows) pass
through this module so empty/whitespace/NaN-like values and noisy delimiters
are handled consistently.
"""

from __future__ import annotations

from collections.abc import Iterable

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
