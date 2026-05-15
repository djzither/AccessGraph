"""
Deterministic canonical role IDs for cohort matching.

Maps (department, workforce, title) to a stable role id. Known operational clusters
live in ROLE_CLUSTERS; everything else falls back to conservative per-title exact ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from DataLayer.workforce_type import (
    FULL_TIME,
    STUDENT,
    UNKNOWN,
    canonical_from_ui_label,
    canonical_from_reference_employee_type,
)

MATCH_PATH_REGISTRY = "registry"
MATCH_PATH_EXACT_FALLBACK = "exact_fallback"


@dataclass(frozen=True)
class RoleCanonicalResult:
    canonical_role_id: str
    match_path: str
    raw_title: str
    title_clean: str
    department_clean: str
    workforce_canonical: str


def normalize_role_text(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return ""
    for old, new in [("&", " and "), (",", " "), ("/", " "), ("-", " ")]:
        text = text.replace(old, new)
    return " ".join(text.split())


def title_stem(title_clean: str) -> str:
    """Strip trailing numeric suffixes (e.g. 'student worker 5' -> 'student worker')."""
    stem = re.sub(r"\s+\d+$", "", title_clean.strip())
    return stem or title_clean


def _workforce_from_inputs(
    employee_type: object | None,
    workforce_canonical: str | None,
) -> str:
    if workforce_canonical in {FULL_TIME, STUDENT, UNKNOWN}:
        return workforce_canonical
    if employee_type is not None and str(employee_type).strip():
        from_ui = canonical_from_ui_label(employee_type)
        if from_ui != UNKNOWN:
            return from_ui
        from_ref = canonical_from_reference_employee_type(employee_type)
        if from_ref != UNKNOWN:
            return from_ref
    return UNKNOWN


@dataclass(frozen=True)
class _RoleClusterSpec:
    department_clean: str
    workforce_canonical: str
    titles: frozenset[str] = frozenset()
    title_stems: frozenset[str] = frozenset()


# Reference-sheet (title, department) alternates for cluster department expansion.
REFERENCE_ROLE_DEPARTMENT_ALIASES: dict[tuple[str, str], frozenset[tuple[str, str]]] = {
    (
        "academic outreach and sales rep",
        "ce academic outreach and sales",
    ): frozenset(
        {
            ("academic outreach sales rep", "marketing and customer support"),
        }
    ),
    (
        "computing specialist",
        "ce it help desk",
    ): frozenset(
        {
            ("computing specialist", "information technology"),
        }
    ),
}

_REFERENCE_STEM_NUMBERED_SUFFIXES = tuple(str(n) for n in range(1, 10))

# Operational clusters: extend here instead of ad-hoc title aliases.
ROLE_CLUSTERS: dict[str, _RoleClusterSpec] = {
    "role:ce_it_helpdesk_student_support": _RoleClusterSpec(
        department_clean="ce it help desk",
        workforce_canonical=STUDENT,
        titles=frozenset(
            {
                "computing specialist",
                "computer specialist",
            }
        ),
        title_stems=frozenset({"student worker"}),
    ),
    "role:ce_it_helpdesk_fulltime_specialist": _RoleClusterSpec(
        department_clean="ce it help desk",
        workforce_canonical=FULL_TIME,
        titles=frozenset(
            {
                "computing specialist",
                "computer specialist",
            }
        ),
    ),
}

_REGISTRY_LOOKUP: dict[tuple[str, str, str], str] = {}
_STEM_LOOKUP: dict[tuple[str, str, str], str] = {}


def _register_clusters() -> None:
    for role_id, spec in ROLE_CLUSTERS.items():
        for title in spec.titles:
            _REGISTRY_LOOKUP[(spec.department_clean, spec.workforce_canonical, title)] = role_id
        for stem in spec.title_stems:
            _STEM_LOOKUP[(spec.department_clean, spec.workforce_canonical, stem)] = role_id


_register_clusters()


def _cluster_reference_departments(spec: _RoleClusterSpec) -> set[str]:
    departments = {spec.department_clean}
    for (_title_key, dept_key), alts in REFERENCE_ROLE_DEPARTMENT_ALIASES.items():
        if dept_key == spec.department_clean:
            departments.update(alt[1] for alt in alts)
        for _alt_title, alt_dept in alts:
            if alt_dept == spec.department_clean:
                departments.add(dept_key)
    return departments


def _cluster_reference_titles(spec: _RoleClusterSpec) -> set[str]:
    titles = set(spec.titles)
    for stem in spec.title_stems:
        titles.add(stem)
        for suffix in _REFERENCE_STEM_NUMBERED_SUFFIXES:
            titles.add(f"{stem} {suffix}")
    return titles


def cluster_reference_candidates(role_id: str) -> set[tuple[str, str]]:
    """
    Reference lookup (title_clean, department_clean) pairs for a canonical role cluster.
    """
    spec = ROLE_CLUSTERS.get(role_id)
    if spec is None:
        return set()

    departments = _cluster_reference_departments(spec)
    titles = _cluster_reference_titles(spec)
    candidates: set[tuple[str, str]] = set()
    for title in titles:
        for dept in departments:
            candidates.add((title, dept))

    for title in titles:
        alias_key = (title, spec.department_clean)
        for alt_title, alt_dept in REFERENCE_ROLE_DEPARTMENT_ALIASES.get(alias_key, ()):
            candidates.add((alt_title, alt_dept))

    return candidates


def canonical_role_id(
    *,
    title: object,
    department: object,
    employee_type: object | None = None,
    workforce_canonical: str | None = None,
) -> RoleCanonicalResult:
    raw_title = "" if title is None else str(title).strip()
    title_clean = normalize_role_text(title)
    department_clean = normalize_role_text(department)
    workforce = _workforce_from_inputs(employee_type, workforce_canonical)

    if department_clean and workforce != UNKNOWN:
        hit = _REGISTRY_LOOKUP.get((department_clean, workforce, title_clean))
        if hit:
            return RoleCanonicalResult(
                canonical_role_id=hit,
                match_path=MATCH_PATH_REGISTRY,
                raw_title=raw_title,
                title_clean=title_clean,
                department_clean=department_clean,
                workforce_canonical=workforce,
            )
        stem = title_stem(title_clean)
        stem_hit = _STEM_LOOKUP.get((department_clean, workforce, stem))
        if stem_hit:
            return RoleCanonicalResult(
                canonical_role_id=stem_hit,
                match_path=MATCH_PATH_REGISTRY,
                raw_title=raw_title,
                title_clean=title_clean,
                department_clean=department_clean,
                workforce_canonical=workforce,
            )

    wf_token = workforce.lower() if workforce != UNKNOWN else "unknown"
    fallback_id = f"exact:{wf_token}:{department_clean}:{title_clean}"
    return RoleCanonicalResult(
        canonical_role_id=fallback_id,
        match_path=MATCH_PATH_EXACT_FALLBACK,
        raw_title=raw_title,
        title_clean=title_clean,
        department_clean=department_clean,
        workforce_canonical=workforce,
    )


