"""
Partial-input onboarding: infer canonical_role_id when job title is missing or unreliable.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from DataLayer.canonical_role import (
    MATCH_PATH_EXACT_FALLBACK,
    MATCH_PATH_REGISTRY,
    RoleCanonicalResult,
    canonical_role_id,
    normalize_role_text,
)
from DataLayer.peer_cohort import infer_workforce_type, parse_manager_netid
from DataLayer.workforce_type import (
    FULL_TIME,
    STUDENT,
    UNKNOWN,
    canonical_from_ui_label,
)

MATCH_PATH_INFERRED_COPY_FROM = "inferred_from_copy_from"
MATCH_PATH_INFERRED_SUPERVISOR = "inferred_from_supervisor"
MATCH_PATH_INFERRED_DEPT_WORKFORCE = "inferred_from_department_workforce"
MATCH_PATH_TITLE_CONFLICT_PREFERS_COPY_FROM = "title_conflict_prefers_copy_from"

INFERRED_MATCH_PATHS = frozenset(
    {
        MATCH_PATH_INFERRED_COPY_FROM,
        MATCH_PATH_INFERRED_SUPERVISOR,
        MATCH_PATH_INFERRED_DEPT_WORKFORCE,
        MATCH_PATH_TITLE_CONFLICT_PREFERS_COPY_FROM,
    }
)

CONFIDENCE_CEILING_EXACT = 1.0
CONFIDENCE_CEILING_COPY_FROM = 0.85
CONFIDENCE_CEILING_SUPERVISOR = 0.80
CONFIDENCE_CEILING_DEPT_WORKFORCE = 0.70
CONFIDENCE_CEILING_AMBIGUOUS = 0.60

_SUPERVISOR_SPLIT_RE = re.compile(r"[\n,;]+")
_SUPERVISOR_INFERENCE_MIN_REPORTS = 2
_SUPERVISOR_INFERENCE_MIN_SHARE = 0.60


@dataclass(frozen=True)
class OnboardingRoleResolution:
    role: RoleCanonicalResult
    confidence_ceiling: float
    warning: str = ""
    ambiguous: bool = False
    candidate_roles: tuple[tuple[str, int], ...] = ()
    inference_debug: dict[str, object] = field(default_factory=dict)


def is_blank_title(title: object) -> bool:
    return not normalize_role_text(title)


def split_supervisor_tokens(supervisor: object) -> list[str]:
    """Split ticket/reference supervisor cells on newline, comma, or semicolon."""
    text = "" if supervisor is None else str(supervisor).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    tokens: list[str] = []
    for part in _SUPERVISOR_SPLIT_RE.split(text):
        tok = normalize_role_text(part)
        if tok:
            tokens.append(tok)
    return tokens


def supervisor_cell_matches_target(cell_value: object, target_supervisor: object) -> bool:
    """True if any token in a reference supervisor cell matches the ticket supervisor."""
    target_tokens = split_supervisor_tokens(target_supervisor)
    if not target_tokens:
        return False
    cell_tokens = split_supervisor_tokens(cell_value)
    if not cell_tokens:
        cell_tokens = [normalize_role_text(cell_value)]
    target_set = set(target_tokens)
    return bool(target_set.intersection(cell_tokens))


def _workforce_canonical(employee_type: object) -> str:
    return canonical_from_ui_label(employee_type)


def _cohort_fallback_role_id(department_clean: str, workforce_canonical: str) -> str:
    wf = workforce_canonical.lower() if workforce_canonical != UNKNOWN else "unknown"
    return f"cohort:{wf}:{department_clean}"


def _role_from_user_row(
    row: pd.Series | dict,
    *,
    workforce_canonical: str,
) -> RoleCanonicalResult:
    if isinstance(row, dict):
        title = row.get("Title", "")
        department = row.get("Department", "")
        employee_type = row.get("EmployeeType")
    else:
        title = row.get("Title", "")
        department = row.get("Department", "")
        employee_type = row.get("EmployeeType")
    return canonical_role_id(
        title=title,
        department=department,
        employee_type=employee_type,
        workforce_canonical=workforce_canonical,
    )


def _role_from_netid(
    users_df: pd.DataFrame,
    netid: str,
    workforce_canonical: str,
) -> RoleCanonicalResult | None:
    key = str(netid).strip().lower()
    if not key or "SamAccountName" not in users_df.columns:
        return None
    rows = users_df[users_df["SamAccountName"].astype(str).str.lower() == key]
    if rows.empty:
        return None
    return _role_from_user_row(rows.iloc[0], workforce_canonical=workforce_canonical)


def _supervisor_lookup_tokens(supervisor: object, users_df: pd.DataFrame) -> set[str]:
    tokens = set(split_supervisor_tokens(supervisor))
    if not tokens or users_df.empty or "SamAccountName" not in users_df.columns:
        return tokens
    for tok in list(tokens):
        hit = users_df[users_df["SamAccountName"].astype(str).str.lower() == tok]
        if not hit.empty:
            tokens.add(tok)
    return tokens


def _reports_for_supervisor(
    users_df: pd.DataFrame,
    *,
    supervisor: object,
    department_clean: str,
    workforce_canonical: str,
) -> pd.DataFrame:
    if users_df.empty or not department_clean:
        return users_df.iloc[0:0]

    scope = users_df.copy()
    if "DepartmentClean" not in scope.columns and "Department" in scope.columns:
        scope["DepartmentClean"] = scope["Department"].apply(normalize_role_text)
    scope = scope[scope["DepartmentClean"] == department_clean]
    if scope.empty:
        return scope

    tokens = _supervisor_lookup_tokens(supervisor, users_df)
    if not tokens:
        return users_df.iloc[0:0]

    matched_idx: list[int] = []
    for idx, row in scope.iterrows():
        row_wf = infer_workforce_type(row)
        if row_wf == "full_time" and workforce_canonical == STUDENT:
            continue
        if row_wf == "student" and workforce_canonical == FULL_TIME:
            continue
        mgr_netid = parse_manager_netid(row.get("Manager", ""))
        if mgr_netid and mgr_netid in tokens:
            matched_idx.append(idx)
            continue
        sam = str(row.get("SamAccountName", "")).strip().lower()
        if sam and sam in tokens:
            matched_idx.append(idx)

    if not matched_idx:
        return users_df.iloc[0:0]
    return scope.loc[matched_idx]


def infer_role_from_supervisor(
    users_df: pd.DataFrame,
    *,
    supervisor: object,
    department: object,
    employee_type: object,
    workforce_canonical: str | None = None,
) -> OnboardingRoleResolution:
    wf = workforce_canonical or _workforce_canonical(employee_type)
    dept_clean = normalize_role_text(department)
    reports = _reports_for_supervisor(
        users_df,
        supervisor=supervisor,
        department_clean=dept_clean,
        workforce_canonical=wf,
    )
    debug: dict[str, object] = {
        "supervisor_input": str(supervisor or ""),
        "supervisor_tokens": sorted(_supervisor_lookup_tokens(supervisor, users_df)),
        "report_count": int(len(reports)),
        "department_clean": dept_clean,
        "workforce_canonical": wf,
    }

    if reports.empty:
        role = RoleCanonicalResult(
            canonical_role_id=_cohort_fallback_role_id(dept_clean, wf),
            match_path=MATCH_PATH_INFERRED_DEPT_WORKFORCE,
            raw_title="",
            title_clean="",
            department_clean=dept_clean,
            workforce_canonical=wf,
        )
        return OnboardingRoleResolution(
            role=role,
            confidence_ceiling=CONFIDENCE_CEILING_DEPT_WORKFORCE,
            warning="No direct reports found for supervisor; using department and workforce cohort.",
            inference_debug=debug,
        )

    counts: Counter[str] = Counter()
    for _, row in reports.iterrows():
        resolved = _role_from_user_row(row, workforce_canonical=wf)
        counts[resolved.canonical_role_id] += 1

    ranked = counts.most_common()
    debug["role_counts"] = dict(ranked)
    top_role_id, top_count = ranked[0]
    share = top_count / max(len(reports), 1)

    if len(ranked) > 1 and share < _SUPERVISOR_INFERENCE_MIN_SHARE:
        candidates = tuple(ranked)
        role = RoleCanonicalResult(
            canonical_role_id=_cohort_fallback_role_id(dept_clean, wf),
            match_path=MATCH_PATH_INFERRED_DEPT_WORKFORCE,
            raw_title="",
            title_clean="",
            department_clean=dept_clean,
            workforce_canonical=wf,
        )
        names = ", ".join(f"{rid} ({cnt})" for rid, cnt in candidates[:5])
        return OnboardingRoleResolution(
            role=role,
            confidence_ceiling=CONFIDENCE_CEILING_AMBIGUOUS,
            warning=f"Supervisor's team has mixed roles ({names}); not inferring a single canonical role.",
            ambiguous=True,
            candidate_roles=candidates,
            inference_debug=debug,
        )

    if top_count < _SUPERVISOR_INFERENCE_MIN_REPORTS:
        role = RoleCanonicalResult(
            canonical_role_id=_cohort_fallback_role_id(dept_clean, wf),
            match_path=MATCH_PATH_INFERRED_DEPT_WORKFORCE,
            raw_title="",
            title_clean="",
            department_clean=dept_clean,
            workforce_canonical=wf,
        )
        return OnboardingRoleResolution(
            role=role,
            confidence_ceiling=CONFIDENCE_CEILING_DEPT_WORKFORCE,
            warning="Too few users under supervisor to infer a role; using department and workforce cohort.",
            inference_debug=debug,
        )

    sample = reports.iloc[0]
    inferred = _role_from_user_row(sample, workforce_canonical=wf)
    role = RoleCanonicalResult(
        canonical_role_id=top_role_id,
        match_path=MATCH_PATH_INFERRED_SUPERVISOR,
        raw_title=str(sample.get("Title", "")),
        title_clean=inferred.title_clean,
        department_clean=dept_clean,
        workforce_canonical=wf,
    )
    return OnboardingRoleResolution(
        role=role,
        confidence_ceiling=CONFIDENCE_CEILING_SUPERVISOR,
        inference_debug=debug,
    )


def infer_role_from_copy_from(
    users_df: pd.DataFrame,
    *,
    copy_from_netid: str,
    department: object,
    employee_type: object,
    workforce_canonical: str | None = None,
) -> OnboardingRoleResolution | None:
    wf = workforce_canonical or _workforce_canonical(employee_type)
    copy_role = _role_from_netid(users_df, copy_from_netid, wf)
    if copy_role is None:
        return None
    dept_clean = normalize_role_text(department)
    role = RoleCanonicalResult(
        canonical_role_id=copy_role.canonical_role_id,
        match_path=MATCH_PATH_INFERRED_COPY_FROM,
        raw_title=copy_role.raw_title,
        title_clean=copy_role.title_clean,
        department_clean=dept_clean or copy_role.department_clean,
        workforce_canonical=wf,
    )
    return OnboardingRoleResolution(
        role=role,
        confidence_ceiling=CONFIDENCE_CEILING_COPY_FROM,
        inference_debug={
            "copy_from_netid": copy_from_netid,
            "copy_from_title": copy_role.raw_title,
            "copy_from_canonical_role_id": copy_role.canonical_role_id,
        },
    )


def infer_role_department_workforce(
    *,
    department: object,
    employee_type: object,
    workforce_canonical: str | None = None,
) -> OnboardingRoleResolution:
    wf = workforce_canonical or _workforce_canonical(employee_type)
    dept_clean = normalize_role_text(department)
    role = RoleCanonicalResult(
        canonical_role_id=_cohort_fallback_role_id(dept_clean, wf),
        match_path=MATCH_PATH_INFERRED_DEPT_WORKFORCE,
        raw_title="",
        title_clean="",
        department_clean=dept_clean,
        workforce_canonical=wf,
    )
    return OnboardingRoleResolution(
        role=role,
        confidence_ceiling=CONFIDENCE_CEILING_DEPT_WORKFORCE,
        warning="Job title missing; using department and employee type cohort only.",
        inference_debug={
            "department_clean": dept_clean,
            "workforce_canonical": wf,
        },
    )


def resolve_onboarding_role(
    *,
    title: object,
    department: object,
    employee_type: object,
    supervisor: object | None = None,
    copy_from_netid: str | None = None,
    users_df: pd.DataFrame | None = None,
) -> OnboardingRoleResolution:
    """
    Resolve the canonical role for a new hire / ticket.

    When title is present and agrees with copy-from (if any), use normal canonical_role_id.
    When title is blank or conflicts with copy-from, infer from copy-from, supervisor, or dept+workforce.
    """
    wf = _workforce_canonical(employee_type)
    dept_clean = normalize_role_text(department)
    users = users_df if users_df is not None else pd.DataFrame()
    debug: dict[str, object] = {
        "raw_title": "" if title is None else str(title).strip(),
        "department": "" if department is None else str(department).strip(),
        "employee_type": "" if employee_type is None else str(employee_type).strip(),
        "supervisor": "" if supervisor is None else str(supervisor).strip(),
        "copy_from_netid": copy_from_netid or "",
        "title_blank": is_blank_title(title),
    }

    title_role: RoleCanonicalResult | None = None
    if not is_blank_title(title):
        title_role = canonical_role_id(
            title=title,
            department=department,
            employee_type=employee_type,
            workforce_canonical=wf,
        )

    copy_resolution: OnboardingRoleResolution | None = None
    if copy_from_netid and not users.empty:
        copy_resolution = infer_role_from_copy_from(
            users,
            copy_from_netid=str(copy_from_netid),
            department=department,
            employee_type=employee_type,
            workforce_canonical=wf,
        )

    if copy_resolution is not None:
        if title_role is not None and title_role.canonical_role_id != copy_resolution.role.canonical_role_id:
            conflict = RoleCanonicalResult(
                canonical_role_id=copy_resolution.role.canonical_role_id,
                match_path=MATCH_PATH_TITLE_CONFLICT_PREFERS_COPY_FROM,
                raw_title=title_role.raw_title,
                title_clean=title_role.title_clean,
                department_clean=dept_clean,
                workforce_canonical=wf,
            )
            return OnboardingRoleResolution(
                role=conflict,
                confidence_ceiling=CONFIDENCE_CEILING_COPY_FROM,
                warning=(
                    "Provided job title does not match the copy-from user's role; "
                    "using copy-from canonical role."
                ),
                inference_debug={
                    **debug,
                    **copy_resolution.inference_debug,
                    "provided_canonical_role_id": title_role.canonical_role_id,
                    "copy_from_canonical_role_id": copy_resolution.role.canonical_role_id,
                },
            )
        if is_blank_title(title):
            return OnboardingRoleResolution(
                role=copy_resolution.role,
                confidence_ceiling=copy_resolution.confidence_ceiling,
                inference_debug={**debug, **copy_resolution.inference_debug},
            )

    if title_role is not None:
        return OnboardingRoleResolution(
            role=title_role,
            confidence_ceiling=CONFIDENCE_CEILING_EXACT
            if title_role.match_path in {MATCH_PATH_REGISTRY, MATCH_PATH_EXACT_FALLBACK}
            else CONFIDENCE_CEILING_EXACT,
            inference_debug=debug,
        )

    if supervisor and str(supervisor).strip() and not users.empty:
        sup = infer_role_from_supervisor(
            users,
            supervisor=supervisor,
            department=department,
            employee_type=employee_type,
            workforce_canonical=wf,
        )
        return OnboardingRoleResolution(
            role=sup.role,
            confidence_ceiling=sup.confidence_ceiling,
            warning=sup.warning,
            ambiguous=sup.ambiguous,
            candidate_roles=sup.candidate_roles,
            inference_debug={**debug, **sup.inference_debug},
        )

    dept_res = infer_role_department_workforce(
        department=department,
        employee_type=employee_type,
        workforce_canonical=wf,
    )
    return OnboardingRoleResolution(
        role=dept_res.role,
        confidence_ceiling=dept_res.confidence_ceiling,
        warning=dept_res.warning,
        inference_debug={**debug, **dept_res.inference_debug},
    )
