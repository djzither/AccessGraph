"""
Centralized permission string normalization for AccessGraph.

All permission tokens (AD groups, reference access names, exploded rows) should pass
through this module so empty/whitespace/NaN-like values and noisy delimiters are
handled consistently.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger("accessgraph.permissions")

# NaN-like string tokens (after strip / lower)
_INVALID_STRING_TOKENS = frozenset(
    {"nan", "none", "null", "#n/a", "n/a", "nat", ""}
)


@dataclass
class PermissionNormalizationStats:
    """Counters for one or more normalization calls (use merge() to aggregate)."""

    raw_segments: int = 0
    dropped_blank_or_invalid: int = 0
    output_tokens: int = 0

    def merge(self, other: PermissionNormalizationStats) -> None:
        self.raw_segments += other.raw_segments
        self.dropped_blank_or_invalid += other.dropped_blank_or_invalid
        self.output_tokens += other.output_tokens


@dataclass
class BatchPermissionStats:
    """Aggregates stats across many values (e.g. a dataframe column)."""

    total_raw_segments: int = 0
    total_dropped: int = 0
    total_output_tokens: int = 0
    rows_processed: int = 0

    def add_row(self, stats: PermissionNormalizationStats) -> None:
        self.total_raw_segments += stats.raw_segments
        self.total_dropped += stats.dropped_blank_or_invalid
        self.total_output_tokens += stats.output_tokens
        self.rows_processed += 1


def normalize_single_permission(value: object) -> str | None:
    """
    Normalize a single permission token: strip ends, drop NaN-like / empty / whitespace-only.

    Does not change inner spelling or case (downstream matchers own case folding).
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            if isinstance(value, float) and pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in _INVALID_STRING_TOKENS:
        return None
    return text


def _collapse_semicolon_runs(s: str) -> str:
    return re.sub(r";+", ";", s.strip())


def _normalize_delimited_string(value: str, stats: PermissionNormalizationStats | None) -> list[str]:
    collapsed = _collapse_semicolon_runs(value)
    if not collapsed or collapsed.lower() in _INVALID_STRING_TOKENS:
        return []
    parts = collapsed.split(";")
    out: list[str] = []
    for p in parts:
        if stats is not None:
            stats.raw_segments += 1
        token = normalize_single_permission(p)
        if not token:
            if stats is not None:
                stats.dropped_blank_or_invalid += 1
            continue
        out.append(token)
    if stats is not None:
        stats.output_tokens = len(out)
    return out


def _normalize_from_iterable(items: Iterable[object], stats: PermissionNormalizationStats | None) -> list[str]:
    out: list[str] = []
    for item in items:
        if stats is not None:
            stats.raw_segments += 1
        token = normalize_single_permission(item)
        if not token:
            if stats is not None:
                stats.dropped_blank_or_invalid += 1
            continue
        out.append(token)
    if stats is not None:
        stats.output_tokens = len(out)
    return out


def normalize_groups_input(
    value: object,
    *,
    stats: PermissionNormalizationStats | None = None,
) -> list[str]:
    """
    Normalize a GroupsList cell value: semicolon string, list/tuple/set, or numpy array.

    Removes empty segments from duplicate delimiters (e.g. 'a;;b' -> ['a', 'b']).
    """
    inner = PermissionNormalizationStats() if stats is not None else None

    if value is None:
        return []

    if isinstance(value, str):
        result = _normalize_delimited_string(value, inner)
    elif isinstance(value, (list, tuple, set)):
        result = _normalize_from_iterable(value, inner)
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            as_list = value.tolist()
        except Exception:
            as_list = None
        if isinstance(as_list, list):
            result = _normalize_from_iterable(as_list, inner)
        else:
            try:
                result = [] if pd.isna(value) else _normalize_delimited_string(str(value), inner)
            except (TypeError, ValueError):
                result = _normalize_delimited_string(str(value), inner)
    else:
        try:
            result = [] if pd.isna(value) else _normalize_delimited_string(str(value), inner)
        except (TypeError, ValueError):
            result = _normalize_delimited_string(str(value), inner)

    if stats is not None:
        stats.merge(inner)
    return result


def log_normalization_batch(logger_: logging.Logger, batch: BatchPermissionStats, *, context: str) -> None:
    """Emit a single INFO line summarizing normalization across a batch."""
    logger_.info(
        "[%s] permission_normalization raw_segments=%s dropped_empty_invalid=%s output_tokens=%s rows=%s",
        context,
        batch.total_raw_segments,
        batch.total_dropped,
        batch.total_output_tokens,
        batch.rows_processed,
    )


def summarize_column_values(values: Iterable[object], *, context: str) -> BatchPermissionStats:
    """Run normalize_groups_input on each value and return aggregate stats."""
    batch = BatchPermissionStats()
    for v in values:
        st = PermissionNormalizationStats()
        normalize_groups_input(v, stats=st)
        batch.add_row(st)
    log_normalization_batch(logger, batch, context=context)
    return batch
