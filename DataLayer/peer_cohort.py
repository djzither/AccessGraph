from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from DataLayer.access_exclusions import filter_group_list
from DataLayer.workforce_type import FULL_TIME, STUDENT, canonical_from_ui_label

DEFAULT_SUPERVISOR_TITLE_KEYWORDS: tuple[str, ...] = (
    "manager",
    "supervisor",
    "director",
    "lead",
    "coordinator",
    "admin",
    "administrator",
    "assistant director",
    "dean",
    "principal",
    "owner",
    "chair",
    "faculty",
    "full time",
    "staff",
)

SENSITIVE_GROUP_KEYWORDS: tuple[str, ...] = (
    "admin",
    "owner",
    "privileged",
    "superuser",
    "domain",
)


def _normalize_text(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return ""
    return text


def _row_value(row: Any, key: str, default: object = "") -> object:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    if isinstance(row, dict):
        return row.get(key, default)
    return default


def _truthy_flag(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if value is True:
        return True
    if value is False:
        return False
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "false", "0", "no"}:
        return False
    return text in {"true", "1", "yes"}


def _permission_count(row: Any) -> int:
    return len(filter_group_list(_row_value(row, "GroupsList", [])))


def _employee_type_canonical(row: Any) -> str:
    raw = _row_value(row, "EmployeeType", "")
    if raw:
        return canonical_from_ui_label(raw)
    title = _normalize_text(_row_value(row, "Title", ""))
    if any(keyword in title for keyword in ("student", "intern")):
        return STUDENT
    if any(keyword in title for keyword in DEFAULT_SUPERVISOR_TITLE_KEYWORDS):
        return FULL_TIME
    return STUDENT


def _title_matches_keywords(title: str, keywords: tuple[str, ...]) -> bool:
    if not title:
        return False
    return any(keyword in title for keyword in keywords)


def _owns_sensitive_groups(row: Any) -> bool:
    for group in filter_group_list(_row_value(row, "GroupsList", [])):
        lowered = str(group).lower()
        if any(keyword in lowered for keyword in SENSITIVE_GROUP_KEYWORDS):
            return True
    return False


def is_supervisor_like(
    row: Any,
    *,
    title_keywords: tuple[str, ...] = DEFAULT_SUPERVISOR_TITLE_KEYWORDS,
    manager_of_count: int = 0,
    cohort_median_group_count: float | None = None,
) -> bool:
    if _truthy_flag(_row_value(row, "IsSupervisor", False)):
        return True

    title = _normalize_text(_row_value(row, "Title", ""))
    if _title_matches_keywords(title, title_keywords):
        return True

    employee_type = _employee_type_canonical(row)
    if employee_type == FULL_TIME and any(
        token in _normalize_text(_row_value(row, "EmployeeType", ""))
        for token in ("staff", "faculty", "full")
    ):
        return True

    if manager_of_count > 0:
        return True

    group_count = _permission_count(row)
    if cohort_median_group_count is not None and cohort_median_group_count > 0:
        if group_count >= max(cohort_median_group_count * 1.75, cohort_median_group_count + 8):
            return True

    if _owns_sensitive_groups(row):
        return True

    return False


def _target_accepts_supervisor_peers(target_row: Any) -> bool:
    if is_supervisor_like(target_row):
        return True
    return _employee_type_canonical(target_row) == FULL_TIME


def is_valid_peer_relationship(
    target_user_row: Any,
    anchor_user_row: Any,
    candidate_peer_row: Any,
    *,
    title_keywords: tuple[str, ...] = DEFAULT_SUPERVISOR_TITLE_KEYWORDS,
) -> bool:
    anchor_netid = _normalize_text(_row_value(anchor_user_row, "SamAccountName", ""))
    candidate_netid = _normalize_text(_row_value(candidate_peer_row, "SamAccountName", ""))
    if anchor_netid and candidate_netid == anchor_netid:
        return True

    candidate_supervisor = is_supervisor_like(candidate_peer_row, title_keywords=title_keywords)
    if not candidate_supervisor:
        return True

    if _target_accepts_supervisor_peers(target_user_row):
        return True

    return False


@dataclass
class PeerPoolBuildResult:
    peer_pool: pd.DataFrame
    anchor_user_name: str = ""
    anchor_user_title: str = ""
    anchor_user_type: str = ""
    peer_pool_size: int = 0
    supervisor_users_excluded: list[str] = field(default_factory=list)
    outlier_users_excluded: list[str] = field(default_factory=list)
    peer_pool_composition: str = ""
    review_reason: str = ""
    peer_users: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict[str, object]:
        return {
            "AnchorUserName": self.anchor_user_name,
            "AnchorUserTitle": self.anchor_user_title,
            "AnchorUserType": self.anchor_user_type,
            "PeerPoolSize": self.peer_pool_size,
            "SupervisorUsersExcluded": ", ".join(self.supervisor_users_excluded),
            "OutlierUsersExcluded": ", ".join(self.outlier_users_excluded),
            "PeerPoolComposition": self.peer_pool_composition,
            "ReviewReason": self.review_reason,
            "PeerUsers": ", ".join(self.peer_users),
        }


def _normalize_role_text(value: object) -> str:
    text = _normalize_text(value)
    for old, new in [("&", " and "), (",", " "), ("/", " "), ("-", " ")]:
        text = text.replace(old, new)
    return " ".join(text.split())


def _manager_counts(users_df: pd.DataFrame) -> dict[str, int]:
    if "Manager" not in users_df.columns or "SamAccountName" not in users_df.columns:
        return {}
    managers = users_df["Manager"].astype(str).str.strip().str.lower()
    managers = managers.replace({"nan": "", "none": ""})
    counts: Counter[str] = Counter()
    for manager in managers:
        if manager:
            counts[manager] += 1
    return dict(counts)


def _median_group_count(users_df: pd.DataFrame) -> float:
    if users_df.empty:
        return 0.0
    counts = [_permission_count(row) for _, row in users_df.iterrows()]
    if not counts:
        return 0.0
    return float(pd.Series(counts).median())


def build_peer_pool_from_anchor(
    users_df: pd.DataFrame,
    anchor_user_row: Any,
    target_user_row: Any | None = None,
    *,
    title_keywords: tuple[str, ...] = DEFAULT_SUPERVISOR_TITLE_KEYWORDS,
) -> PeerPoolBuildResult:
    target_row = target_user_row if target_user_row is not None else anchor_user_row
    users = users_df.copy()
    manager_counts = _manager_counts(users)
    cohort_median = _median_group_count(users)

    anchor_title = str(_row_value(anchor_user_row, "Title", ""))
    anchor_department = str(_row_value(anchor_user_row, "Department", ""))
    anchor_type = _employee_type_canonical(anchor_user_row)
    anchor_name = str(_row_value(anchor_user_row, "DisplayName", "") or _row_value(anchor_user_row, "SamAccountName", ""))
    anchor_netid = str(_row_value(anchor_user_row, "SamAccountName", ""))
    anchor_group_count = _permission_count(anchor_user_row)
    anchor_supervisor = is_supervisor_like(
        anchor_user_row,
        title_keywords=title_keywords,
        manager_of_count=manager_counts.get(_normalize_text(anchor_name), 0),
        cohort_median_group_count=cohort_median,
    )

    users["TitleClean"] = users["Title"].apply(_normalize_role_text)
    users["DepartmentClean"] = users["Department"].apply(_normalize_role_text)
    anchor_title_clean = _normalize_role_text(anchor_title)
    anchor_department_clean = _normalize_role_text(anchor_department)

    candidates = users[users["DepartmentClean"] == anchor_department_clean].copy()
    if candidates.empty:
        candidates = users.copy()

    same_type = candidates
    if "EmployeeType" in candidates.columns:
        typed = candidates[candidates["EmployeeType"].apply(_employee_type_canonical) == anchor_type]
        if not typed.empty:
            same_type = typed

    title_scoped = same_type[same_type["TitleClean"] == anchor_title_clean]
    if not title_scoped.empty:
        candidates = title_scoped
    else:
        candidates = same_type

    selected_rows: list[pd.Series] = []
    excluded_supervisors: list[str] = []
    excluded_outliers: list[str] = []
    review_reasons: list[str] = []

    target_supervisor = is_supervisor_like(target_row, title_keywords=title_keywords)
    if _employee_type_canonical(target_row) == STUDENT and not target_supervisor and anchor_supervisor:
        review_reasons.append(
            "Copy-from user appears supervisory for a student target; supervisor permissions are review-only."
        )

    for _, candidate in candidates.iterrows():
        candidate_netid = str(candidate.get("SamAccountName", ""))
        if candidate_netid == anchor_netid:
            selected_rows.append(candidate)
            continue

        manager_count = manager_counts.get(_normalize_text(candidate.get("DisplayName", "")), 0)
        candidate_supervisor = is_supervisor_like(
            candidate,
            title_keywords=title_keywords,
            manager_of_count=manager_count,
            cohort_median_group_count=cohort_median,
        )

        if not is_valid_peer_relationship(
            target_row,
            anchor_user_row,
            candidate,
            title_keywords=title_keywords,
        ):
            if candidate_supervisor:
                excluded_supervisors.append(candidate_netid)
            else:
                excluded_outliers.append(candidate_netid)
            continue

        if not anchor_supervisor and candidate_supervisor:
            excluded_supervisors.append(candidate_netid)
            continue

        if not anchor_supervisor:
            candidate_group_count = _permission_count(candidate)
            if candidate_group_count > max(anchor_group_count * 1.5, anchor_group_count + 6):
                excluded_outliers.append(candidate_netid)
                continue

        selected_rows.append(candidate)

    if not any(str(row.get("SamAccountName", "")) == anchor_netid for row in selected_rows):
        anchor_match = users[users["SamAccountName"].astype(str) == anchor_netid]
        if not anchor_match.empty:
            selected_rows.insert(0, anchor_match.iloc[0])

    peer_pool = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame(columns=users.columns)
    peer_users = peer_pool["SamAccountName"].astype(str).tolist() if "SamAccountName" in peer_pool.columns else []
    selected_netids = set(peer_users)
    department_scope = users[users["DepartmentClean"] == anchor_department_clean]
    for _, candidate in department_scope.iterrows():
        candidate_netid = str(candidate.get("SamAccountName", ""))
        if not candidate_netid or candidate_netid in selected_netids:
            continue
        if candidate_netid in excluded_supervisors or candidate_netid in excluded_outliers:
            continue

        manager_count = manager_counts.get(_normalize_text(candidate.get("DisplayName", "")), 0)
        candidate_supervisor = is_supervisor_like(
            candidate,
            title_keywords=title_keywords,
            manager_of_count=manager_count,
            cohort_median_group_count=cohort_median,
        )
        if not candidate_supervisor:
            continue

        if not is_valid_peer_relationship(
            target_row,
            anchor_user_row,
            candidate,
            title_keywords=title_keywords,
        ) or (not anchor_supervisor):
            excluded_supervisors.append(candidate_netid)

    supervisor_count = sum(
        1
        for _, row in peer_pool.iterrows()
        if is_supervisor_like(row, title_keywords=title_keywords)
    )
    student_count = max(len(peer_pool) - supervisor_count, 0)
    composition = f"students={student_count}; supervisors={supervisor_count}"

    return PeerPoolBuildResult(
        peer_pool=peer_pool,
        anchor_user_name=anchor_name,
        anchor_user_title=anchor_title,
        anchor_user_type=anchor_type,
        peer_pool_size=len(peer_pool),
        supervisor_users_excluded=excluded_supervisors,
        outlier_users_excluded=excluded_outliers,
        peer_pool_composition=composition,
        review_reason=" ".join(review_reasons).strip(),
        peer_users=peer_users,
    )


@dataclass
class ContaminationStats:
    peer_student_support_count: int
    supervisor_support_count: int
    supervisor_contamination_flag: bool
    recommendation_source: str
    review_reason: str = ""

    def as_row_metadata(self) -> dict[str, object]:
        return {
            "PeerStudentSupportCount": self.peer_student_support_count,
            "SupervisorSupportCount": self.supervisor_support_count,
            "SupervisorContaminationFlag": self.supervisor_contamination_flag,
            "RecommendationSource": self.recommendation_source,
            "ReviewReason": self.review_reason,
        }


def _user_has_group(row: Any, group_name: str, normalizer) -> bool:
    normalized_target = normalizer(group_name)
    for group in filter_group_list(_row_value(row, "GroupsList", [])):
        if normalizer(group) == normalized_target:
            return True
    return False


def contamination_stats_for_group(
    peer_pool: pd.DataFrame,
    group_name: str,
    *,
    normalizer,
    title_keywords: tuple[str, ...] = DEFAULT_SUPERVISOR_TITLE_KEYWORDS,
) -> ContaminationStats:
    if peer_pool.empty:
        return ContaminationStats(0, 0, False, "insufficient_peer_evidence")

    manager_counts = _manager_counts(peer_pool)
    student_support = 0
    supervisor_support = 0

    for _, row in peer_pool.iterrows():
        if not _user_has_group(row, group_name, normalizer):
            continue
        manager_count = manager_counts.get(_normalize_text(row.get("DisplayName", "")), 0)
        if is_supervisor_like(
            row,
            title_keywords=title_keywords,
            manager_of_count=manager_count,
        ):
            supervisor_support += 1
        else:
            student_support += 1

    contamination = (
        supervisor_support >= 2
        and student_support <= 1
        and supervisor_support > student_support
    )
    source = "student_peer_baseline" if student_support >= 2 else "mixed_or_supervisor_evidence"
    review_reason = ""
    if contamination:
        review_reason = (
            "Supervisor-heavy support in a student peer cohort; treat as review evidence."
        )
        source = "supervisor_contamination_risk"

    return ContaminationStats(
        peer_student_support_count=student_support,
        supervisor_support_count=supervisor_support,
        supervisor_contamination_flag=contamination,
        recommendation_source=source,
        review_reason=review_reason,
    )


def build_target_user_row(
    *,
    title: str,
    department: str,
    employee_type: str,
    sam_account_name: str = "",
    display_name: str = "",
) -> dict[str, object]:
    return {
        "SamAccountName": sam_account_name,
        "DisplayName": display_name,
        "Title": title,
        "Department": department,
        "EmployeeType": employee_type,
        "GroupsList": [],
    }
