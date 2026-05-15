from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from DataLayer.access_exclusions import filter_group_list
from DataLayer.permission_normalization import normalize_groups_input
from DataLayer.canonical_role import (
    MATCH_PATH_EXACT_FALLBACK,
    MATCH_PATH_REGISTRY,
    RoleCanonicalResult,
    canonical_role_id,
    normalize_role_text,
)
from DataLayer.workforce_type import FULL_TIME, STUDENT, UNKNOWN, canonical_from_ui_label

logger = logging.getLogger(__name__)

DEFAULT_SUPERVISOR_TITLE_KEYWORDS: tuple[str, ...] = (
    "manager",
    "supervisor",
    "director",
    "administrator",
    "assistant director",
    "dean",
    "principal",
    "owner",
    "chair",
    "faculty",
)

SENSITIVE_GROUP_KEYWORDS: tuple[str, ...] = (
    "admin",
    "owner",
    "privileged",
    "superuser",
    "domain",
)

FULL_TIME_STAFF_GROUP = "a.FULL TIME STAFF"
STUDENT_GROUP_MARKERS: tuple[str, ...] = (
    "a.FULL TIME STUDENT SENIOR",
    "a.FULL TIME STUDENT JUNIOR",
    "a.FULL TIME STUDENT",
)
INVALID_GROUP_FRAGMENTS: tuple[str, ...] = ("cannot find an object",)

WORKFORCE_FULL_TIME = "full_time"
WORKFORCE_STUDENT = "student"
WORKFORCE_UNKNOWN = "unknown"


def _normalize_text(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return ""
    return text


def parse_manager_netid(manager_dn: object) -> str | None:
    text = "" if manager_dn is None else str(manager_dn).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    match = re.search(r"(?:^|,)\s*CN=([^,]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    netid = match.group(1).strip().lower()
    return netid or None


def normalize_groups(groups: object) -> list[str]:
    """Expand GroupsList cells to permission strings (delegates to normalize_groups_input).

    Handles numpy.ndarray / pandas Series values without stringifying the whole array,
    which previously produced a single pseudo-token and broke supervisor heuristics.
    """
    cleaned: list[str] = []
    for value in normalize_groups_input(groups):
        lowered = value.lower()
        if any(fragment in lowered for fragment in INVALID_GROUP_FRAGMENTS):
            continue
        cleaned.append(value)
    return filter_group_list(cleaned)


def _groups_from_row(row: Any) -> list[str]:
    groups = _row_value(row, "GroupsList", None)
    if groups is None or (isinstance(groups, float) and pd.isna(groups)):
        groups = _row_value(row, "Groups", "")
    return normalize_groups(groups)


def infer_workforce_type_from_groups(
    groups: object,
    employee_type: object | None = None,
) -> str:
    group_list = normalize_groups(groups)
    lowered = {group.lower() for group in group_list}

    if FULL_TIME_STAFF_GROUP.lower() in lowered:
        return WORKFORCE_FULL_TIME
    if any(marker.lower() in lowered for marker in STUDENT_GROUP_MARKERS):
        return WORKFORCE_STUDENT

    if employee_type is not None and str(employee_type).strip():
        canonical = canonical_from_ui_label(employee_type)
        if canonical == FULL_TIME:
            return WORKFORCE_FULL_TIME
        if canonical == STUDENT:
            return WORKFORCE_STUDENT
    return WORKFORCE_UNKNOWN


def infer_workforce_type(row: Any) -> str:
    return infer_workforce_type_from_groups(
        _groups_from_row(row),
        employee_type=_row_value(row, "EmployeeType", None),
    )


def _workforce_to_canonical(workforce_type: str) -> str:
    """Convert peer-cohort workforce constants to canonical workforce_type values.

    Returns UNKNOWN rather than silently defaulting to STUDENT when the type
    cannot be resolved — callers that need a display value should handle this.
    """
    if workforce_type == WORKFORCE_FULL_TIME:
        return FULL_TIME
    if workforce_type == WORKFORCE_STUDENT:
        return STUDENT
    return UNKNOWN


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
    return len(_groups_from_row(row))


def _title_matches_keywords(title: str, keywords: tuple[str, ...]) -> bool:
    if not title:
        return False
    t = title.lower().strip()
    for keyword in keywords:
        kw = keyword.strip().lower()
        if not kw:
            continue
        if " " in kw:
            parts = [re.escape(p) for p in kw.split()]
            pattern = r"\b" + r"\s+".join(parts) + r"\b"
        else:
            pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, t):
            return True
    return False


def _sensitive_keyword_match(lowered: str, keyword: str) -> bool:
    """True when ``keyword`` appears as its own token, not as a substring (e.g. *admin* in *DomainAdmins*)."""
    if not keyword:
        return False
    for m in re.finditer(re.escape(keyword), lowered):
        i, j = m.span()
        before = lowered[i - 1] if i > 0 else " "
        after = lowered[j] if j < len(lowered) else " "
        if not before.isalnum() and not after.isalnum():
            return True
    return False


def _owns_sensitive_groups(row: Any) -> bool:
    for group in _groups_from_row(row):
        lowered = str(group).lower()
        if any(_sensitive_keyword_match(lowered, keyword) for keyword in SENSITIVE_GROUP_KEYWORDS):
            return True
    return False


def _has_staff_group(row: Any) -> bool:
    lowered = {group.lower() for group in _groups_from_row(row)}
    return FULL_TIME_STAFF_GROUP.lower() in lowered


def _has_student_group(row: Any) -> bool:
    lowered = {group.lower() for group in _groups_from_row(row)}
    return any(marker.lower() in lowered for marker in STUDENT_GROUP_MARKERS)


def manager_netids_set(users_df: pd.DataFrame) -> set[str]:
    if "Manager" not in users_df.columns:
        return set()
    netids: set[str] = set()
    for manager_dn in users_df["Manager"]:
        netid = parse_manager_netid(manager_dn)
        if netid:
            netids.add(netid)
    return netids


def _candidate_netid(row: Any) -> str:
    return _normalize_text(_row_value(row, "SamAccountName", ""))


def is_manager_of_others(users_df: pd.DataFrame, candidate_netid: object) -> bool:
    netid = _normalize_text(candidate_netid)
    if not netid:
        return False
    return netid in manager_netids_set(users_df)


def is_supervisor_like(
    row: Any,
    *,
    users_df: pd.DataFrame | None = None,
    title_keywords: tuple[str, ...] = DEFAULT_SUPERVISOR_TITLE_KEYWORDS,
    cohort_median_group_count: float | None = None,
    target_workforce_type: str | None = None,
    decision_notes: list[str] | None = None,
) -> bool:
    def _note(msg: str) -> None:
        if decision_notes is not None:
            decision_notes.append(msg)

    if _truthy_flag(_row_value(row, "IsSupervisor", False)):
        _note("matched:IsSupervisor_column_truthy")
        return True

    title = _normalize_text(_row_value(row, "Title", ""))
    if _title_matches_keywords(title, title_keywords):
        _note("matched:title_supervisor_keywords")
        return True

    candidate_netid = _candidate_netid(row)
    if users_df is not None and candidate_netid and is_manager_of_others(users_df, candidate_netid):
        _note("matched:manager_of_others_graph")
        return True

    if target_workforce_type == WORKFORCE_STUDENT and _has_staff_group(row):
        _note("matched:student_target_with_full_time_staff_group")
        return True

    group_count = _permission_count(row)
    if cohort_median_group_count is not None and cohort_median_group_count > 0:
        if group_count >= max(cohort_median_group_count * 1.75, cohort_median_group_count + 8):
            _note(
                "matched:permission_count_outlier_vs_cohort_median "
                f"(count={group_count}, median={cohort_median_group_count})"
            )
            return True

    if _owns_sensitive_groups(row):
        _note("matched:sensitive_group_keyword_on_individual_groups")
        return True

    _note("no_match:not_supervisor_like")
    return False


def _target_accepts_supervisor_peers(
    target_row: Any,
    *,
    users_df: pd.DataFrame | None = None,
) -> bool:
    target_workforce = infer_workforce_type(target_row)
    return is_supervisor_like(
        target_row,
        users_df=users_df,
        target_workforce_type=target_workforce,
    )


def is_valid_peer_relationship(
    target_user_row: Any,
    anchor_user_row: Any,
    candidate_peer_row: Any,
    *,
    users_df: pd.DataFrame | None = None,
    title_keywords: tuple[str, ...] = DEFAULT_SUPERVISOR_TITLE_KEYWORDS,
) -> bool:
    anchor_netid = _normalize_text(_row_value(anchor_user_row, "SamAccountName", ""))
    candidate_netid = _normalize_text(_row_value(candidate_peer_row, "SamAccountName", ""))
    if anchor_netid and candidate_netid == anchor_netid:
        return True

    target_workforce = infer_workforce_type(target_user_row)
    candidate_workforce = infer_workforce_type(candidate_peer_row)

    if target_workforce != WORKFORCE_UNKNOWN and candidate_workforce != WORKFORCE_UNKNOWN:
        if target_workforce != candidate_workforce:
            if not (
                target_workforce == WORKFORCE_FULL_TIME
                and _target_accepts_supervisor_peers(target_user_row, users_df=users_df)
                and is_supervisor_like(
                    candidate_peer_row,
                    users_df=users_df,
                    target_workforce_type=target_workforce,
                    title_keywords=title_keywords,
                )
            ):
                return False

    if target_workforce == WORKFORCE_STUDENT:
        if users_df is not None and is_manager_of_others(users_df, candidate_netid):
            return False
        if _has_staff_group(candidate_peer_row):
            return False
        if not _has_student_group(candidate_peer_row) and candidate_workforce != WORKFORCE_STUDENT:
            return False

    candidate_supervisor = is_supervisor_like(
        candidate_peer_row,
        users_df=users_df,
        target_workforce_type=target_workforce,
        title_keywords=title_keywords,
    )
    if candidate_supervisor and not _target_accepts_supervisor_peers(target_user_row, users_df=users_df):
        return False

    return True


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
    target_workforce_type: str = WORKFORCE_UNKNOWN
    anchor_workforce_type: str = WORKFORCE_UNKNOWN
    anchor_mismatch_flag: bool = False
    manager_netid: str = ""
    full_time_excluded_for_student_target: list[str] = field(default_factory=list)
    students_excluded_for_full_time_target: list[str] = field(default_factory=list)
    manager_of_others_excluded: list[str] = field(default_factory=list)
    fallback_reason: str = ""
    cohort_filter_diagnostics: dict[str, Any] | None = None

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
            "TargetWorkforceType": self.target_workforce_type,
            "AnchorWorkforceType": self.anchor_workforce_type,
            "AnchorMismatchFlag": self.anchor_mismatch_flag,
            "ManagerNetId": self.manager_netid,
            "FullTimeExcludedForStudentTarget": ", ".join(self.full_time_excluded_for_student_target),
            "StudentsExcludedForFullTimeTarget": ", ".join(self.students_excluded_for_full_time_target),
            "ManagerOfOthersExcluded": ", ".join(self.manager_of_others_excluded),
            "FallbackReason": self.fallback_reason,
        }


def _normalize_role_text(value: object) -> str:
    return normalize_role_text(value)


def _role_for_row(
    row: Any,
    *,
    workforce_canonical: str,
) -> RoleCanonicalResult:
    return canonical_role_id(
        title=_row_value(row, "Title", ""),
        department=_row_value(row, "Department", ""),
        employee_type=_row_value(row, "EmployeeType", None),
        workforce_canonical=workforce_canonical,
    )


def _median_group_count(users_df: pd.DataFrame) -> float:
    if users_df.empty:
        return 0.0
    counts = [_permission_count(row) for _, row in users_df.iterrows()]
    if not counts:
        return 0.0
    return float(pd.Series(counts).median())


def median_permission_count(users_df: pd.DataFrame) -> float:
    """Median permission count per user row (for diagnostics; matches peer-pool cohort median)."""
    return _median_group_count(users_df)


def _manager_netid_for_row(row: Any) -> str:
    return parse_manager_netid(_row_value(row, "Manager", "")) or ""


def peer_cohort_user_snapshot(row: Any) -> dict[str, Any]:
    """Stable diagnostic fields for a user row (Series or dict-like)."""
    return {
        "SamAccountName": str(_row_value(row, "SamAccountName", "")),
        "Title": str(_row_value(row, "Title", "")),
        "Department": str(_row_value(row, "Department", "")),
        "EmployeeType": str(_row_value(row, "EmployeeType", "")),
        "Manager": str(_row_value(row, "Manager", "")),
        "IsSupervisor": bool(_truthy_flag(_row_value(row, "IsSupervisor", False))),
    }


def explain_peer_cohort_build(
    users_df: pd.DataFrame,
    anchor_user_row: Any,
    target_user_row: Any | None = None,
    *,
    title_keywords: tuple[str, ...] = DEFAULT_SUPERVISOR_TITLE_KEYWORDS,
) -> dict[str, Any] | None:
    """Return cohort filter diagnostics (for notebooks / optional engine flag)."""
    return build_peer_pool_from_anchor(
        users_df,
        anchor_user_row,
        target_user_row,
        title_keywords=title_keywords,
        cohort_diagnostics=True,
    ).cohort_filter_diagnostics


def _is_cohort_wide_role_id(role_id: str) -> bool:
    return str(role_id).startswith("cohort:")


def build_peer_pool_from_anchor(
    users_df: pd.DataFrame,
    anchor_user_row: Any,
    target_user_row: Any | None = None,
    *,
    title_keywords: tuple[str, ...] = DEFAULT_SUPERVISOR_TITLE_KEYWORDS,
    cohort_diagnostics: bool = False,
    target_role: RoleCanonicalResult | None = None,
) -> PeerPoolBuildResult:
    target_row = target_user_row if target_user_row is not None else anchor_user_row
    users = users_df.copy()

    anchor_title = str(_row_value(anchor_user_row, "Title", ""))
    anchor_department = str(_row_value(anchor_user_row, "Department", ""))
    anchor_name = str(
        _row_value(anchor_user_row, "DisplayName", "")
        or _row_value(anchor_user_row, "SamAccountName", "")
    )
    anchor_netid = str(_row_value(anchor_user_row, "SamAccountName", ""))
    anchor_group_count = _permission_count(anchor_user_row)
    target_workforce = infer_workforce_type(target_row)
    anchor_workforce = infer_workforce_type(anchor_user_row)
    anchor_manager_netid = _manager_netid_for_row(anchor_user_row)

    users["TitleClean"] = users["Title"].apply(_normalize_role_text)
    users["DepartmentClean"] = users["Department"].apply(_normalize_role_text)
    anchor_title_clean = _normalize_role_text(anchor_title)
    anchor_department_clean = _normalize_role_text(anchor_department)
    target_workforce_canonical = _workforce_to_canonical(target_workforce)
    if target_role is not None:
        anchor_role = target_role
    else:
        anchor_role = _role_for_row(
            anchor_user_row,
            workforce_canonical=target_workforce_canonical,
        )

    filter_stages: list[dict[str, Any]] = []

    def _filter_stage(name: str, frame: pd.DataFrame) -> None:
        if not cohort_diagnostics:
            return
        netids: list[str] = []
        if not frame.empty and "SamAccountName" in frame.columns:
            netids = frame["SamAccountName"].astype(str).head(48).tolist()
        filter_stages.append({"stage": name, "count": int(len(frame)), "sample_netids": netids})

    _filter_stage("00_full_users_with_clean_columns", users)

    department_candidates = users[users["DepartmentClean"] == anchor_department_clean].copy()
    if department_candidates.empty:
        department_candidates = users.copy()
    _filter_stage("01_after_department_scope", department_candidates)

    cohort_median = median_permission_count(department_candidates)

    anchor_supervisor = is_supervisor_like(
        anchor_user_row,
        users_df=users,
        target_workforce_type=target_workforce,
        title_keywords=title_keywords,
        cohort_median_group_count=cohort_median,
    )
    anchor_mismatch = (
        target_workforce == WORKFORCE_STUDENT
        and (
            anchor_workforce == WORKFORCE_FULL_TIME
            or anchor_supervisor
        )
    )

    same_type = department_candidates[
        department_candidates.apply(lambda row: infer_workforce_type(row) == target_workforce, axis=1)
    ].copy()
    if same_type.empty:
        same_type = department_candidates
    _filter_stage("02_after_workforce_alignment", same_type)

    same_type = same_type.copy()

    def _assign_role_columns(frame: pd.DataFrame) -> pd.DataFrame:
        role_ids: list[str] = []
        match_paths: list[str] = []
        for _, row in frame.iterrows():
            resolved = _role_for_row(
                row,
                workforce_canonical=target_workforce_canonical,
            )
            role_ids.append(resolved.canonical_role_id)
            match_paths.append(resolved.match_path)
        frame = frame.copy()
        frame["CanonicalRoleId"] = role_ids
        frame["RoleMatchPath"] = match_paths
        return frame

    same_type = _assign_role_columns(same_type)

    if _is_cohort_wide_role_id(anchor_role.canonical_role_id):
        candidates = same_type
    else:
        role_scoped = same_type[
            same_type["CanonicalRoleId"] == anchor_role.canonical_role_id
        ]
        if not role_scoped.empty:
            candidates = role_scoped
        elif anchor_role.match_path == MATCH_PATH_EXACT_FALLBACK:
            title_scoped = same_type[same_type["TitleClean"] == anchor_title_clean]
            candidates = title_scoped if not title_scoped.empty else same_type
        else:
            candidates = same_type
    _filter_stage("03_pre_pairwise_selection", candidates)

    removals: list[dict[str, Any]] = []
    scoped_snapshots: list[dict[str, Any]] = []
    if cohort_diagnostics:
        for _, r in candidates.iterrows():
            scoped_snapshots.append(peer_cohort_user_snapshot(r))

    def _record_removal(row: pd.Series, rule: str) -> None:
        if cohort_diagnostics:
            entry = peer_cohort_user_snapshot(row)
            entry["exclusion_rule"] = rule
            removals.append(entry)
        logger.debug(
            "peer_cohort removal: sam=%s rule=%s title=%r dept=%r",
            str(row.get("SamAccountName", "")),
            rule,
            str(row.get("Title", "")),
            str(row.get("Department", "")),
        )

    selected_rows: list[pd.Series] = []
    excluded_supervisors: list[str] = []
    excluded_outliers: list[str] = []
    full_time_excluded: list[str] = []
    students_excluded: list[str] = []
    manager_of_others_excluded: list[str] = []
    review_reasons: list[str] = []
    fallback_reason = ""

    if anchor_mismatch:
        review_reasons.append(
            "Copy-from user workforce does not match student target; anchor evidence is review-only."
        )

    logger.debug(
        "peer_cohort filter_start: scoped_candidates=%s anchor=%s",
        len(candidates),
        anchor_netid,
    )

    for _, candidate in candidates.iterrows():
        candidate_netid = str(candidate.get("SamAccountName", ""))
        if candidate_netid == anchor_netid:
            if anchor_mismatch:
                _record_removal(candidate, "anchor_mismatch_skip_self_copy")
                continue
            selected_rows.append(candidate)
            continue

        candidate_workforce = infer_workforce_type(candidate)
        candidate_supervisor = is_supervisor_like(
            candidate,
            users_df=users,
            target_workforce_type=target_workforce,
            title_keywords=title_keywords,
            cohort_median_group_count=cohort_median,
        )

        if target_workforce == WORKFORCE_STUDENT and candidate_workforce == WORKFORCE_FULL_TIME:
            full_time_excluded.append(candidate_netid)
            _record_removal(candidate, "student_target_excludes_full_time_peer")
            continue
        if target_workforce == WORKFORCE_FULL_TIME and candidate_workforce == WORKFORCE_STUDENT:
            students_excluded.append(candidate_netid)
            _record_removal(candidate, "full_time_target_excludes_student_peer")
            continue
        if target_workforce == WORKFORCE_STUDENT and is_manager_of_others(users, candidate_netid):
            manager_of_others_excluded.append(candidate_netid)
            _record_removal(candidate, "student_target_excludes_manager_of_others")
            continue

        if not is_valid_peer_relationship(
            target_row,
            anchor_user_row,
            candidate,
            users_df=users,
            title_keywords=title_keywords,
        ):
            if candidate_supervisor:
                excluded_supervisors.append(candidate_netid)
                _record_removal(
                    candidate,
                    "invalid_peer_relationship_supervisor_like_candidate",
                )
            else:
                excluded_outliers.append(candidate_netid)
                _record_removal(
                    candidate,
                    "invalid_peer_relationship_workforce_or_peer_policy",
                )
            continue

        if target_workforce == WORKFORCE_STUDENT and not anchor_supervisor and candidate_supervisor:
            excluded_supervisors.append(candidate_netid)
            _record_removal(candidate, "student_target_excludes_supervisor_like_peer")
            continue

        if not anchor_supervisor:
            candidate_group_count = _permission_count(candidate)
            if candidate_group_count > max(anchor_group_count * 1.5, anchor_group_count + 6):
                excluded_outliers.append(candidate_netid)
                _record_removal(candidate, "outlier_permission_count_vs_anchor")
                continue

        selected_rows.append(candidate)

    if (
        not anchor_mismatch
        and not any(str(row.get("SamAccountName", "")) == anchor_netid for row in selected_rows)
    ):
        anchor_match = users[users["SamAccountName"].astype(str) == anchor_netid]
        if not anchor_match.empty:
            selected_rows.insert(0, anchor_match.iloc[0])

    peer_pool = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame(columns=users.columns)
    if target_workforce == WORKFORCE_STUDENT and anchor_manager_netid and not peer_pool.empty:
        peer_pool = peer_pool.assign(
            _ManagerNetId=peer_pool["Manager"].apply(_manager_netid_for_row),
            _SameManager=peer_pool["Manager"].apply(
                lambda value: (_manager_netid_for_row({"Manager": value}) == anchor_manager_netid)
            ),
        )
        same_manager = peer_pool[peer_pool["_SameManager"]].drop(columns=["_ManagerNetId", "_SameManager"])
        other_peers = peer_pool[~peer_pool["_SameManager"]].drop(columns=["_ManagerNetId", "_SameManager"])
        if not same_manager.empty:
            peer_pool = pd.concat([same_manager, other_peers], ignore_index=True)
        else:
            peer_pool = peer_pool.drop(columns=["_ManagerNetId", "_SameManager"])

    logger.debug(
        "peer_cohort filter_end: final_peer_pool=%s anchor=%s",
        len(peer_pool),
        anchor_netid,
    )
    _filter_stage("04_after_pairwise_and_outlier_filters", peer_pool)

    peer_users = peer_pool["SamAccountName"].astype(str).tolist() if "SamAccountName" in peer_pool.columns else []
    selected_netids = set(peer_users)

    # Single pass over the full department scope to:
    # (a) record workforce-type exclusions for department members not in the candidate pool
    # (b) record supervisor exclusions for those same department members
    # Previously two separate loops; merged here to avoid the duplicate iteration.
    department_scope = users[users["DepartmentClean"] == anchor_department_clean]
    for _, candidate in department_scope.iterrows():
        candidate_netid = str(candidate.get("SamAccountName", ""))
        if not candidate_netid or candidate_netid in selected_netids:
            continue

        candidate_workforce = infer_workforce_type(candidate)

        # Track workforce-type exclusions (department members outside the selected pool)
        if target_workforce == WORKFORCE_STUDENT and candidate_workforce == WORKFORCE_FULL_TIME:
            if candidate_netid not in full_time_excluded:
                full_time_excluded.append(candidate_netid)
        elif target_workforce == WORKFORCE_FULL_TIME and candidate_workforce == WORKFORCE_STUDENT:
            if candidate_netid not in students_excluded:
                students_excluded.append(candidate_netid)

        # Track supervisor exclusions (skip those already classified in the selection pass)
        if candidate_netid in excluded_supervisors or candidate_netid in excluded_outliers:
            continue

        candidate_supervisor = is_supervisor_like(
            candidate,
            users_df=users,
            target_workforce_type=target_workforce,
            title_keywords=title_keywords,
            cohort_median_group_count=cohort_median,
        )
        if not candidate_supervisor:
            continue

        if not is_valid_peer_relationship(
            target_row,
            anchor_user_row,
            candidate,
            users_df=users,
            title_keywords=title_keywords,
        ) or (target_workforce == WORKFORCE_STUDENT and not anchor_supervisor):
            excluded_supervisors.append(candidate_netid)

    if target_workforce == WORKFORCE_STUDENT and len(peer_pool) < 2:
        fallback_reason = "Insufficient student peers after workforce and manager filtering."

    supervisor_count = sum(
        1
        for _, row in peer_pool.iterrows()
        if is_supervisor_like(
            row,
            users_df=users,
            target_workforce_type=target_workforce,
            title_keywords=title_keywords,
        )
    )
    student_count = max(len(peer_pool) - supervisor_count, 0)
    composition = (
        f"students={student_count}; supervisors={supervisor_count}; "
        f"target={target_workforce}; anchor={anchor_workforce}"
    )

    cohort_filter_diagnostics: dict[str, Any] | None = None
    if cohort_diagnostics:
        final_peers = (
            [peer_cohort_user_snapshot(r) for _, r in peer_pool.iterrows()]
            if not peer_pool.empty
            else []
        )
        cohort_filter_diagnostics = {
            "anchor_samaccountname": anchor_netid,
            "anchor_department_clean": anchor_department_clean,
            "anchor_title_clean": anchor_title_clean,
            "anchor_canonical_role_id": anchor_role.canonical_role_id,
            "anchor_role_match_path": anchor_role.match_path,
            "anchor_raw_title": anchor_role.raw_title,
            "target_canonical_role_id": anchor_role.canonical_role_id,
            "target_workforce_type": target_workforce,
            "anchor_mismatch": anchor_mismatch,
            "scoped_candidate_count": int(len(candidates)),
            "scoped_candidates": scoped_snapshots,
            "removals": removals,
            "final_peer_count": int(len(peer_pool)),
            "final_peers": final_peers,
            "filter_stages": filter_stages,
        }

    return PeerPoolBuildResult(
        peer_pool=peer_pool,
        anchor_user_name=anchor_name,
        anchor_user_title=anchor_title,
        anchor_user_type=_workforce_to_canonical(anchor_workforce),
        peer_pool_size=len(peer_pool),
        supervisor_users_excluded=excluded_supervisors,
        outlier_users_excluded=excluded_outliers,
        peer_pool_composition=composition,
        review_reason=" ".join(review_reasons).strip(),
        peer_users=peer_users,
        target_workforce_type=target_workforce,
        anchor_workforce_type=anchor_workforce,
        anchor_mismatch_flag=anchor_mismatch,
        manager_netid=anchor_manager_netid,
        full_time_excluded_for_student_target=full_time_excluded,
        students_excluded_for_full_time_target=students_excluded,
        manager_of_others_excluded=manager_of_others_excluded,
        fallback_reason=fallback_reason,
        cohort_filter_diagnostics=cohort_filter_diagnostics,
    )


@dataclass
class ContaminationStats:
    peer_student_support_count: int
    supervisor_support_count: int
    supervisor_contamination_flag: bool
    recommendation_source: str
    review_reason: str = ""
    full_time_support_count: int = 0
    same_manager_peer_support_count: int = 0

    def as_row_metadata(self) -> dict[str, object]:
        return {
            "PeerStudentSupportCount": self.peer_student_support_count,
            "SupervisorSupportCount": self.supervisor_support_count,
            "FullTimeSupportCount": self.full_time_support_count,
            "SameManagerPeerSupportCount": self.same_manager_peer_support_count,
            "SupervisorContaminationFlag": self.supervisor_contamination_flag,
            "RecommendationSource": self.recommendation_source,
            "ReviewReason": self.review_reason,
        }


def _user_has_group(row: Any, group_name: str, normalizer) -> bool:
    normalized_target = normalizer(group_name)
    for group in _groups_from_row(row):
        if normalizer(group) == normalized_target:
            return True
    return False


def contamination_stats_for_group(
    peer_pool: pd.DataFrame,
    group_name: str,
    *,
    normalizer,
    target_row: Any | None = None,
    users_df: pd.DataFrame | None = None,
    title_keywords: tuple[str, ...] = DEFAULT_SUPERVISOR_TITLE_KEYWORDS,
) -> ContaminationStats:
    if peer_pool.empty:
        return ContaminationStats(0, 0, False, "insufficient_peer_evidence")

    target_workforce = infer_workforce_type(target_row) if target_row is not None else WORKFORCE_UNKNOWN
    target_manager = _manager_netid_for_row(target_row) if target_row is not None else ""
    student_support = 0
    supervisor_support = 0
    full_time_support = 0
    same_manager_support = 0

    for _, row in peer_pool.iterrows():
        if not _user_has_group(row, group_name, normalizer):
            continue

        row_workforce = infer_workforce_type(row)
        row_supervisor = is_supervisor_like(
            row,
            users_df=users_df,
            target_workforce_type=target_workforce,
            title_keywords=title_keywords,
        )
        if row_workforce == WORKFORCE_FULL_TIME or _has_staff_group(row):
            full_time_support += 1
        if row_supervisor:
            supervisor_support += 1
        elif row_workforce != WORKFORCE_FULL_TIME and not _has_staff_group(row):
            student_support += 1
        if target_manager and _manager_netid_for_row(row) == target_manager:
            same_manager_support += 1

    contamination = (
        supervisor_support >= 2
        and student_support <= 1
        and supervisor_support > student_support
    )
    if target_workforce == WORKFORCE_STUDENT and full_time_support > student_support:
        contamination = True

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
        full_time_support_count=full_time_support,
        same_manager_peer_support_count=same_manager_support,
    )


def build_target_user_row(
    *,
    title: str,
    department: str,
    employee_type: str,
    sam_account_name: str = "",
    display_name: str = "",
    manager: str = "",
    groups: object | None = None,
) -> dict[str, object]:
    groups_list = normalize_groups(groups) if groups is not None else []
    return {
        "SamAccountName": sam_account_name,
        "DisplayName": display_name,
        "Title": title,
        "Department": department,
        "EmployeeType": employee_type,
        "Manager": manager,
        "GroupsList": groups_list,
    }
