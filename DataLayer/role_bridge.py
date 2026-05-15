"""
Deterministic role bridge resolution: link ticket/copy-from AD titles to access-sheet templates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from DataLayer.access_exclusions import filter_group_list
from DataLayer.canonical_role import (
    MATCH_PATH_EXACT_FALLBACK,
    MATCH_PATH_REGISTRY,
    REFERENCE_ROLE_DEPARTMENT_ALIASES,
    RoleCanonicalResult,
    canonical_role_id,
    normalize_role_text,
)
from DataLayer.permission_normalization import canonical_permission_id
from DataLayer.role_inference import (
    INFERRED_MATCH_PATHS,
    supervisor_cell_matches_target,
)
from DataLayer.workforce_type import (
    FULL_TIME,
    STUDENT,
    UNKNOWN,
    canonical_from_reference_employee_type,
    canonical_from_ui_label,
    reference_match_value,
)

MATCH_PATH_CONFIRMED_ROLE_BRIDGE = "confirmed_role_bridge"

BRIDGE_CONFIDENCE_HIGH = 0.72
BRIDGE_CONFIDENCE_CERTAIN = 0.88
BRIDGE_CONFIDENCE_LOW = 0.55
BRIDGE_AMBIGUITY_DELTA = 0.05
BRIDGE_PERMISSION_OVERLAP_MIN = 0.35
BRIDGE_COHORT_PERMISSION_OVERLAP_MIN = 0.50
BRIDGE_TOP_N = 3

@dataclass(frozen=True)
class RoleBridgeCandidate:
    access_template_title: str
    access_template_department: str
    employee_type: str
    matched_reference_permissions_count: int
    copy_from_permission_overlap_count: int
    copy_from_permission_overlap_ratio: float
    title_similarity_score: float
    ticket_title_similarity: float
    copy_from_title_similarity: float
    supervisor_match: bool
    cohort_overlap_count: int
    bridge_confidence: float
    explanation: str
    template_permission_ids: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["template_permission_ids"] = sorted(self.template_permission_ids)
        return data


@dataclass(frozen=True)
class RoleBridgeConfirmation:
    access_template_title: str
    access_template_department: str
    employee_type: str

    @classmethod
    def from_mapping(cls, value: dict[str, object] | None) -> RoleBridgeConfirmation | None:
        if not value:
            return None
        title = str(value.get("access_template_title", "")).strip()
        dept = str(value.get("access_template_department", "")).strip()
        et = str(value.get("employee_type", "")).strip()
        if not title or not dept:
            return None
        return cls(
            access_template_title=title,
            access_template_department=dept,
            employee_type=et,
        )


@dataclass
class RoleBridgeResolution:
    candidates: list[RoleBridgeCandidate] = field(default_factory=list)
    needs_confirmation: bool = False
    ambiguous: bool = False
    prompt_message: str = ""
    best_candidate: RoleBridgeCandidate | None = None
    debug: dict[str, object] = field(default_factory=dict)


def title_token_similarity(left: str, right: str) -> float:
    """Deterministic Jaccard similarity on normalized title tokens."""
    a = normalize_role_text(left)
    b = normalize_role_text(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def allowed_reference_departments(
    ticket_title_clean: str,
    ticket_department_clean: str,
) -> frozenset[str]:
    allowed = {ticket_department_clean} if ticket_department_clean else set()
    for (title_key, dept_key), alts in REFERENCE_ROLE_DEPARTMENT_ALIASES.items():
        if title_key == ticket_title_clean and dept_key == ticket_department_clean:
            allowed.update(alt[1] for alt in alts)
        for alt_title, alt_dept in alts:
            if alt_title == ticket_title_clean and alt_dept == ticket_department_clean:
                allowed.add(dept_key)
    return frozenset(allowed)


def _copy_from_permission_set(
    users_df: pd.DataFrame,
    copy_from_netid: str | None,
) -> frozenset[str]:
    if not copy_from_netid or users_df.empty or "SamAccountName" not in users_df.columns:
        return frozenset()
    key = str(copy_from_netid).strip().lower()
    rows = users_df[users_df["SamAccountName"].astype(str).str.lower() == key]
    if rows.empty:
        return frozenset()
    row = rows.iloc[0]
    groups = filter_group_list(row.get("GroupsList"))
    return frozenset(
        pid
        for g in groups
        if (pid := canonical_permission_id(g))
    )


def _copy_from_title(users_df: pd.DataFrame, copy_from_netid: str | None) -> str:
    if not copy_from_netid or users_df.empty:
        return ""
    key = str(copy_from_netid).strip().lower()
    rows = users_df[users_df["SamAccountName"].astype(str).str.lower() == key]
    if rows.empty:
        return ""
    return str(rows.iloc[0].get("Title", "")).strip()


def _user_permission_set(row: Any) -> set[str]:
    groups = filter_group_list(row.get("GroupsList") if hasattr(row, "get") else None)
    return {pid for g in groups if (pid := canonical_permission_id(g))}


def _permission_overlap_ratio(left: frozenset[str], right: frozenset[str]) -> float:
    if not right:
        return 0.0
    if not left:
        return 0.0
    return len(left & right) / len(right)


def _cohort_overlap_count(
    users_df: pd.DataFrame,
    *,
    template_title_clean: str,
    template_department_clean: str,
    workforce_canonical: str,
    template_permissions: frozenset[str],
) -> int:
    if users_df.empty:
        return 0
    users = users_df.copy()
    if "TitleClean" not in users.columns:
        users["TitleClean"] = users["Title"].apply(normalize_role_text)
    if "DepartmentClean" not in users.columns:
        users["DepartmentClean"] = users["Department"].apply(normalize_role_text)

    count = 0
    for _, row in users.iterrows():
        if row.get("DepartmentClean") != template_department_clean:
            continue
        wf = canonical_from_ui_label(row.get("EmployeeType"))
        if wf == UNKNOWN:
            wf = canonical_from_reference_employee_type(row.get("EmployeeType"))
        if workforce_canonical == STUDENT and wf != STUDENT:
            continue
        if workforce_canonical == FULL_TIME and wf != FULL_TIME:
            continue
        title_match = str(row.get("TitleClean", "")) == template_title_clean
        perm_overlap = _permission_overlap_ratio(
            frozenset(_user_permission_set(row)),
            template_permissions,
        )
        if title_match or perm_overlap >= BRIDGE_COHORT_PERMISSION_OVERLAP_MIN:
            count += 1
    return count


def is_weak_role_or_reference_match(
    *,
    role: RoleCanonicalResult,
    reference_match_path: str,
    reference_row_count: int,
) -> bool:
    if reference_match_path in {"exact_title_dept", "copy_from_reference_name"} and reference_row_count >= 2:
        return False
    if role.match_path == MATCH_PATH_REGISTRY and reference_row_count >= 3:
        return False
    if role.match_path in INFERRED_MATCH_PATHS:
        return True
    if role.match_path == MATCH_PATH_EXACT_FALLBACK:
        return True
    if reference_row_count <= 0:
        return True
    if reference_match_path in {"", "no_reference_match"}:
        return True
    return True


def _reference_templates(
    reference_df: pd.DataFrame,
    *,
    employee_type_clean: str,
    allowed_departments: frozenset[str],
) -> list[dict[str, object]]:
    if reference_df.empty or not allowed_departments:
        return []
    ref = reference_df.copy()
    if "JobTitleClean" not in ref.columns:
        ref["JobTitleClean"] = ref["JobTitle"].apply(normalize_role_text)
    if "DepartmentClean" not in ref.columns:
        ref["DepartmentClean"] = ref["Department"].apply(normalize_role_text)
    if "EmployeeTypeClean" not in ref.columns:
        ref["EmployeeTypeClean"] = ref["EmployeeType"].astype(str).str.lower().str.strip()
    if "AccessNameClean" not in ref.columns:
        ref["AccessNameClean"] = ref["AccessName"].apply(canonical_permission_id)

    scoped = ref[
        (ref["EmployeeTypeClean"] == employee_type_clean)
        & (ref["DepartmentClean"].isin(sorted(allowed_departments)))
    ]
    if scoped.empty:
        return []

    templates: list[dict[str, object]] = []
    group_cols = ["JobTitle", "Department", "EmployeeTypeClean"]
    for keys, group in scoped.groupby(group_cols, dropna=False):
        job_title = str(keys[0]).strip()
        department = str(keys[1]).strip()
        perms = frozenset(
            str(p)
            for p in group["AccessNameClean"].dropna().astype(str)
            if str(p).strip()
        )
        supervisor_vals = group["Supervisor"].dropna().astype(str).tolist() if "Supervisor" in group.columns else []
        templates.append(
            {
                "access_template_title": job_title,
                "access_template_department": department,
                "employee_type_clean": str(keys[2]),
                "template_permissions": perms,
                "supervisor_values": supervisor_vals,
                "title_clean": normalize_role_text(job_title),
                "department_clean": normalize_role_text(department),
            }
        )
    return templates


def _score_template(
    template: dict[str, object],
    *,
    ticket_title: str,
    copy_from_title: str,
    copy_from_permissions: frozenset[str],
    supervisor: object | None,
    users_df: pd.DataFrame,
    workforce_canonical: str,
) -> RoleBridgeCandidate:
    perms: frozenset[str] = template["template_permissions"]  # type: ignore[assignment]
    overlap_count = len(copy_from_permissions & perms) if copy_from_permissions else 0
    overlap_ratio = overlap_count / max(len(perms), 1) if perms else 0.0

    ticket_sim = title_token_similarity(ticket_title, str(template["access_template_title"]))
    copy_sim = title_token_similarity(copy_from_title, str(template["access_template_title"]))
    title_sim = max(ticket_sim, copy_sim)

    supervisor_match = False
    if supervisor and template.get("supervisor_values"):
        for cell in template["supervisor_values"]:
            if supervisor_cell_matches_target(cell, supervisor):
                supervisor_match = True
                break

    cohort_n = _cohort_overlap_count(
        users_df,
        template_title_clean=str(template["title_clean"]),
        template_department_clean=str(template["department_clean"]),
        workforce_canonical=workforce_canonical,
        template_permissions=perms,
    )

    bridge_confidence = (
        0.22 * title_sim
        + 0.42 * min(1.0, overlap_ratio)
        + 0.18 * min(1.0, cohort_n / 5.0)
        + 0.10 * (1.0 if supervisor_match else 0.0)
        + 0.08 * min(1.0, len(perms) / 12.0)
    )
    if overlap_ratio >= 0.5 and title_sim < 0.2:
        bridge_confidence = min(1.0, bridge_confidence + 0.12)

    explanation_parts = [
        f"title_sim={title_sim:.2f}",
        f"copy_from_overlap={overlap_count}/{max(len(perms), 1)}",
        f"cohort_users={cohort_n}",
    ]
    if supervisor_match:
        explanation_parts.append("supervisor_match=yes")

    return RoleBridgeCandidate(
        access_template_title=str(template["access_template_title"]),
        access_template_department=str(template["access_template_department"]),
        employee_type=str(template["employee_type_clean"]),
        matched_reference_permissions_count=len(perms),
        copy_from_permission_overlap_count=overlap_count,
        copy_from_permission_overlap_ratio=round(overlap_ratio, 4),
        title_similarity_score=round(title_sim, 4),
        ticket_title_similarity=round(ticket_sim, 4),
        copy_from_title_similarity=round(copy_sim, 4),
        supervisor_match=supervisor_match,
        cohort_overlap_count=cohort_n,
        bridge_confidence=round(min(1.0, bridge_confidence), 4),
        explanation="; ".join(explanation_parts),
        template_permission_ids=perms,
    )


def generate_role_bridge_candidates(
    *,
    ticket_title: str,
    department: str,
    employee_type: str,
    supervisor: object | None,
    copy_from_netid: str | None,
    current_role: RoleCanonicalResult,
    reference_df: pd.DataFrame,
    users_df: pd.DataFrame,
    reference_match_path: str = "",
    reference_permission_count: int = 0,
) -> RoleBridgeResolution:
    ticket_clean = normalize_role_text(ticket_title)
    dept_clean = normalize_role_text(department)
    workforce = current_role.workforce_canonical
    if workforce == UNKNOWN:
        workforce = canonical_from_ui_label(employee_type)

    employee_type_clean = reference_match_value(canonical_from_ui_label(employee_type))
    if not employee_type_clean:
        employee_type_clean = reference_match_value(workforce)

    debug: dict[str, object] = {
        "ticket_title_clean": ticket_clean,
        "department_clean": dept_clean,
        "workforce_canonical": workforce,
        "employee_type_clean": employee_type_clean,
        "current_canonical_role_id": current_role.canonical_role_id,
    }

    if workforce == UNKNOWN or not employee_type_clean:
        return RoleBridgeResolution(
            debug={**debug, "reason": "unsafe_missing_workforce"},
        )

    if not is_weak_role_or_reference_match(
        role=current_role,
        reference_match_path=reference_match_path,
        reference_row_count=reference_permission_count,
    ):
        return RoleBridgeResolution(debug={**debug, "reason": "strong_exact_match"})

    allowed_depts = allowed_reference_departments(ticket_clean, dept_clean)
    debug["allowed_reference_departments"] = sorted(allowed_depts)

    templates = _reference_templates(
        reference_df,
        employee_type_clean=employee_type_clean,
        allowed_departments=allowed_depts,
    )
    if not templates:
        return RoleBridgeResolution(debug={**debug, "reason": "no_templates_in_scope"})

    copy_title = _copy_from_title(users_df, copy_from_netid)
    copy_perms = _copy_from_permission_set(users_df, copy_from_netid)

    scored: list[RoleBridgeCandidate] = []
    for template in templates:
        candidate = _score_template(
            template,
            ticket_title=ticket_title,
            copy_from_title=copy_title,
            copy_from_permissions=copy_perms,
            supervisor=supervisor,
            users_df=users_df,
            workforce_canonical=workforce,
        )
        if candidate.copy_from_permission_overlap_ratio < BRIDGE_PERMISSION_OVERLAP_MIN:
            if candidate.title_similarity_score < 0.25 and not candidate.supervisor_match:
                continue
        if candidate.bridge_confidence < BRIDGE_CONFIDENCE_LOW:
            continue
        scored.append(candidate)

    scored.sort(
        key=lambda c: (
            -c.bridge_confidence,
            -c.copy_from_permission_overlap_count,
            -c.cohort_overlap_count,
            c.access_template_title.lower(),
        )
    )
    top = scored[:BRIDGE_TOP_N]
    debug["template_count_scored"] = len(scored)

    if not top:
        return RoleBridgeResolution(candidates=[], debug={**debug, "reason": "no_candidates_above_threshold"})

    ambiguous = False
    if len(top) >= 2 and (top[0].bridge_confidence - top[1].bridge_confidence) < BRIDGE_AMBIGUITY_DELTA:
        ambiguous = True

    best = top[0]
    needs_confirmation = (
        not ambiguous
        and BRIDGE_CONFIDENCE_HIGH <= best.bridge_confidence < BRIDGE_CONFIDENCE_CERTAIN
    ) or ambiguous

    prompt = ""
    if ambiguous:
        prompt = (
            "Multiple access-sheet templates are similarly close. "
            "Confirm which template to use before bridging."
        )
    elif needs_confirmation:
        prompt = (
            "The ticket title does not exactly match the access sheet, but this access "
            f"template appears closest: **{best.access_template_title}** "
            f"({best.access_template_department})."
        )

    return RoleBridgeResolution(
        candidates=top,
        needs_confirmation=needs_confirmation or ambiguous,
        ambiguous=ambiguous,
        prompt_message=prompt,
        best_candidate=best,
        debug=debug,
    )


def confirmed_bridge_role_result(
    confirmation: RoleBridgeConfirmation,
    *,
    workforce_canonical: str,
) -> RoleCanonicalResult:
    resolved = canonical_role_id(
        title=confirmation.access_template_title,
        department=confirmation.access_template_department,
        employee_type=confirmation.employee_type,
        workforce_canonical=workforce_canonical,
    )
    return RoleCanonicalResult(
        canonical_role_id=resolved.canonical_role_id,
        match_path=MATCH_PATH_CONFIRMED_ROLE_BRIDGE,
        raw_title=confirmation.access_template_title,
        title_clean=resolved.title_clean,
        department_clean=resolved.department_clean,
        workforce_canonical=workforce_canonical,
    )
