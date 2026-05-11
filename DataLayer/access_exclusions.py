from __future__ import annotations

import pandas as pd

from DataLayer.permission_normalization import normalize_groups_input, normalize_single_permission


# Temporary CRM disable switch. Re-enable CRM by clearing these sets.
EXCLUDED_ACCESS_CATEGORIES = {"CRM Access"}
EXCLUDED_KEYWORDS = {"crm", "salesforce"}


def is_excluded_access_category(value: object) -> bool:
    text = "" if pd.isna(value) else str(value).strip().lower()
    return text in {category.lower() for category in EXCLUDED_ACCESS_CATEGORIES}


def is_excluded_permission(value: object) -> bool:
    text = normalize_single_permission(value)
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in EXCLUDED_KEYWORDS)


def is_excluded_access(category: object = None, permission: object = None) -> bool:
    return is_excluded_access_category(category) or is_excluded_permission(permission)


def filter_group_list(groups: object) -> list[str]:
    values = normalize_groups_input(groups)
    return [group for group in values if not is_excluded_permission(group)]


def count_excluded_group_entries(df: pd.DataFrame, column: str = "GroupsList") -> int:
    if column not in df.columns:
        return 0
    return int(
        sum(
            1
            for groups in df[column]
            for group in normalize_groups_input(groups)
            if is_excluded_permission(group)
        )
    )


def filter_user_groups_df(df: pd.DataFrame, column: str = "GroupsList") -> pd.DataFrame:
    if column not in df.columns:
        return df.copy()
    filtered = df.copy()
    filtered[column] = filtered[column].apply(filter_group_list)
    return filtered


def filter_reference_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    filtered = df.copy()
    mask = pd.Series(False, index=filtered.index)
    if "AccessCategory" in filtered.columns:
        mask = mask | filtered["AccessCategory"].apply(is_excluded_access_category)
    if "AccessName" in filtered.columns:
        mask = mask | filtered["AccessName"].apply(is_excluded_permission)
    return filtered[~mask].copy()


def count_excluded_reference_rows(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(len(df) - len(filter_reference_df(df)))


def filter_recommendations_df(df: pd.DataFrame, column: str = "GroupName") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df.copy()
    out = df[~df[column].apply(is_excluded_permission)].copy()
    out = out[out[column].apply(lambda v: normalize_single_permission(v) is not None)].copy()
    return out
