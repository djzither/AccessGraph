from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from DataLayer.access_exclusions import filter_group_list


@dataclass(frozen=True)
class SubgroupDetectionConfig:
    min_common_rate: float = 0.60
    min_indicator_lift: float = 0.40
    min_indicator_count: int = 2
    top_n_common: int = 10
    top_n_indicators: int = 5


def _normalize_group(value: object) -> str:
    return str(value).strip()


def _user_groups_map(users_df: pd.DataFrame) -> dict[str, set[str]]:
    user_map: dict[str, set[str]] = {}
    for _, row in users_df.iterrows():
        netid = str(row.get("SamAccountName", "")).strip()
        if not netid:
            continue
        groups = {_normalize_group(g) for g in filter_group_list(row.get("GroupsList")) if _normalize_group(g)}
        user_map[netid] = groups
    return user_map


def _slice_common_permissions(
    netids: list[str],
    user_groups: dict[str, set[str]],
    exclude_group: str,
    min_rate: float,
    top_n: int,
) -> list[str]:
    if not netids:
        return []
    counter = Counter()
    for netid in netids:
        counter.update(user_groups.get(netid, set()))
    threshold = max(1, int(round(len(netids) * min_rate)))
    rows = []
    for group, count in counter.items():
        if group == exclude_group:
            continue
        if count >= threshold:
            rows.append((group, count))
    rows.sort(key=lambda x: (-x[1], x[0].lower()))
    return [name for name, _ in rows[:top_n]]


def analyze_permission_subgroup(
    comparison_cohort: pd.DataFrame,
    permission: str,
    config: SubgroupDetectionConfig | None = None,
) -> dict[str, object]:
    cfg = config or SubgroupDetectionConfig()
    permission = _normalize_group(permission)
    if comparison_cohort.empty:
        return {
            "permission": permission,
            "broad_cohort_size": 0,
            "users_with_permission": [],
            "users_without_permission": [],
            "with_shared_permissions": [],
            "without_shared_permissions": [],
            "strongest_subgroup_indicators": [],
            "subgroup_assessment": "Rare Access",
        }

    user_groups = _user_groups_map(comparison_cohort)
    netids = sorted(user_groups.keys())
    with_perm = [n for n in netids if permission in user_groups.get(n, set())]
    without_perm = [n for n in netids if permission not in user_groups.get(n, set())]

    with_shared = _slice_common_permissions(
        netids=with_perm,
        user_groups=user_groups,
        exclude_group=permission,
        min_rate=cfg.min_common_rate,
        top_n=cfg.top_n_common,
    )
    without_shared = _slice_common_permissions(
        netids=without_perm,
        user_groups=user_groups,
        exclude_group=permission,
        min_rate=cfg.min_common_rate,
        top_n=cfg.top_n_common,
    )

    with_counts = Counter()
    without_counts = Counter()
    for n in with_perm:
        with_counts.update(user_groups.get(n, set()))
    for n in without_perm:
        without_counts.update(user_groups.get(n, set()))

    indicator_rows: list[dict[str, object]] = []
    with_total = max(1, len(with_perm))
    without_total = max(1, len(without_perm))
    all_candidates = sorted(set(with_counts.keys()) | set(without_counts.keys()))
    for group in all_candidates:
        if group == permission:
            continue
        c_with = with_counts.get(group, 0)
        c_without = without_counts.get(group, 0)
        with_rate = c_with / with_total
        without_rate = c_without / without_total if without_perm else 0.0
        lift = with_rate - without_rate
        if c_with >= cfg.min_indicator_count and with_rate >= cfg.min_common_rate and lift >= cfg.min_indicator_lift:
            indicator_rows.append(
                {
                    "permission": group,
                    "with_count": int(c_with),
                    "without_count": int(c_without),
                    "with_rate": round(with_rate, 3),
                    "without_rate": round(without_rate, 3),
                    "lift": round(lift, 3),
                }
            )

    indicator_rows.sort(key=lambda r: (-float(r["lift"]), -int(r["with_count"]), str(r["permission"]).lower()))
    indicator_rows = indicator_rows[: cfg.top_n_indicators]

    broad_size = len(netids)
    target_rate = len(with_perm) / max(1, broad_size)
    is_subrole = len(with_perm) >= 2 and len(indicator_rows) > 0 and target_rate < 0.90
    subgroup_assessment = "Subrole Access" if is_subrole else "Rare Access"

    return {
        "permission": permission,
        "broad_cohort_size": broad_size,
        "users_with_permission": with_perm,
        "users_without_permission": without_perm,
        "with_shared_permissions": with_shared,
        "without_shared_permissions": without_shared,
        "strongest_subgroup_indicators": indicator_rows,
        "subgroup_assessment": subgroup_assessment,
    }


def analyze_recommendation_subgroups(
    comparison_cohort: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    config: SubgroupDetectionConfig | None = None,
) -> pd.DataFrame:
    permissions = (
        recommendations_df.get("GroupName", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.strip()
    )
    rows = [
        analyze_permission_subgroup(comparison_cohort=comparison_cohort, permission=permission, config=config)
        for permission in sorted(set(permissions))
        if permission
    ]
    if not rows:
        return pd.DataFrame(
            columns=[
                "permission",
                "broad_cohort_size",
                "users_with_permission",
                "users_without_permission",
                "with_shared_permissions",
                "without_shared_permissions",
                "strongest_subgroup_indicators",
                "subgroup_assessment",
            ]
        )
    return pd.DataFrame(rows)
