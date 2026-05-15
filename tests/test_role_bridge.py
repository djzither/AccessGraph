import pandas as pd
import pytest

from DataLayer.canonical_role import canonical_role_id
from DataLayer.peer_cohort import (
    PROVENANCE_BRIDGE_PERMISSION,
    PROVENANCE_BRIDGE_TITLE,
    build_bridge_expanded_cohort,
)
from DataLayer.role_bridge import (
    MATCH_PATH_CONFIRMED_ROLE_BRIDGE,
    RoleBridgeConfirmation,
    generate_role_bridge_candidates,
    is_weak_role_or_reference_match,
    supervisor_cell_matches_target,
    title_token_similarity,
)
from DataLayer.workforce_type import STUDENT
from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


def _ta_reference_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "EmployeeType": "Student",
                "JobTitle": "Teaching Assistant",
                "Department": "Psychology",
                "Supervisor": "Dr Smith",
                "AccessCategory": "AD Rights",
                "AccessName": "Email",
                "SourceFile": "student_employee_access.xlsx",
            },
            {
                "EmployeeType": "Student",
                "JobTitle": "Teaching Assistant",
                "Department": "Psychology",
                "Supervisor": "Dr Smith",
                "AccessCategory": "AD Rights",
                "AccessName": "VPN",
                "SourceFile": "student_employee_access.xlsx",
            },
            {
                "EmployeeType": "Student",
                "JobTitle": "Learning Assistant",
                "Department": "Psychology",
                "Supervisor": "Dr Smith",
                "AccessCategory": "AD Rights",
                "AccessName": "Email",
                "SourceFile": "student_employee_access.xlsx",
            },
        ]
    )


def _ta_users_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SamAccountName": "copyuser",
                "Title": "BYUO TA - PSYCH 307 + 320",
                "Department": "Psychology",
                "EmployeeType": "Student",
                "GroupsList": ["Email", "VPN"],
                "Manager": "",
            },
            {
                "SamAccountName": "peer1",
                "Title": "Teaching Assistant",
                "Department": "Psychology",
                "EmployeeType": "Student",
                "GroupsList": ["Email", "VPN"],
                "Manager": "",
            },
            {
                "SamAccountName": "peer2",
                "Title": "Student Worker 5",
                "Department": "Psychology",
                "EmployeeType": "Student",
                "GroupsList": ["Email"],
                "Manager": "",
            },
        ]
    )


def test_title_differs_but_permission_overlap_generates_bridge_candidate():
    users = _ta_users_df()
    ref = _ta_reference_df()
    role = canonical_role_id(
        title="Teaching Assistant",
        department="Psychology",
        employee_type="Student",
        workforce_canonical=STUDENT,
    )
    weak = canonical_role_id(
        title="BYUO TA - PSYCH 307 + 320",
        department="Psychology",
        employee_type="Student",
        workforce_canonical=STUDENT,
    )
    assert is_weak_role_or_reference_match(
        role=weak,
        reference_match_path="no_reference_match",
        reference_row_count=0,
    )
    resolution = generate_role_bridge_candidates(
        ticket_title="Teaching Assistant",
        department="Psychology",
        employee_type="Student",
        supervisor=None,
        copy_from_netid="copyuser",
        current_role=role,
        reference_df=ref,
        users_df=users,
        reference_match_path="no_reference_match",
        reference_permission_count=0,
    )
    assert resolution.candidates
    top = resolution.candidates[0]
    assert top.access_template_title == "Teaching Assistant"
    assert top.copy_from_permission_overlap_count >= 2
    assert top.bridge_confidence >= 0.55


def test_ambiguous_bridge_when_two_templates_score_close():
    users = _ta_users_df()
    ref = _ta_reference_df()
    role = canonical_role_id(
        title="Teaching Assistant",
        department="Psychology",
        employee_type="Student",
        workforce_canonical=STUDENT,
    )
    resolution = generate_role_bridge_candidates(
        ticket_title="Teaching Assistant",
        department="Psychology",
        employee_type="Student",
        supervisor="Dr Smith",
        copy_from_netid="copyuser",
        current_role=role,
        reference_df=ref,
        users_df=users,
        reference_match_path="no_reference_match",
        reference_permission_count=0,
    )
    assert len(resolution.candidates) >= 2
    if (
        resolution.candidates[0].bridge_confidence - resolution.candidates[1].bridge_confidence
        < 0.05
    ):
        assert resolution.ambiguous is True
        assert resolution.needs_confirmation is True


def test_no_bridge_when_workforce_department_mismatch_unsafe():
    users = _ta_users_df()
    ref = _ta_reference_df()
    role = canonical_role_id(
        title="Teaching Assistant",
        department="Psychology",
        employee_type="Full Time",
        workforce_canonical="FULL_TIME",
    )
    resolution = generate_role_bridge_candidates(
        ticket_title="Teaching Assistant",
        department="Psychology",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid="copyuser",
        current_role=role,
        reference_df=ref,
        users_df=users,
        reference_match_path="no_reference_match",
        reference_permission_count=0,
    )
    assert not resolution.candidates
    assert resolution.debug.get("reason") in {"no_templates_in_scope", "no_candidates_above_threshold"}


def test_confirmed_bridge_adds_reference_permissions():
    users = _ta_users_df()
    ref = _ta_reference_df()
    engine = AccessRecommendationEngine(min_confidence=0.1)
    bridge = {
        "access_template_title": "Teaching Assistant",
        "access_template_department": "Psychology",
        "employee_type": "student",
        "template_permission_ids": ["email", "vpn"],
    }
    recs = engine.recommend_for_hire(
        users_df=users,
        reference_df=ref,
        title="BYUO TA - PSYCH 307 + 320",
        department="Psychology",
        employee_type="Student",
        copy_from_netid="copyuser",
        confirmed_role_bridge=bridge,
    )
    assert recs.attrs.get("role_bridge", {}).get("confirmed") is True
    if not recs.empty:
        assert bool(recs["InReferenceSheet"].any())


def test_confirmed_bridge_expands_cohort_with_provenance():
    users = _ta_users_df()
    ref = _ta_reference_df()
    bridge_role = canonical_role_id(
        title="Teaching Assistant",
        department="Psychology",
        employee_type="Student",
        workforce_canonical=STUDENT,
    )
    expanded = build_bridge_expanded_cohort(
        users,
        department_clean="psychology",
        workforce_canonical=STUDENT,
        canonical_role_id_target=bridge_role.canonical_role_id,
        bridge_title_clean="teaching assistant",
        template_permission_ids=frozenset({"email", "vpn"}),
    )
    assert len(expanded.cohort) >= 2
    assert (
        expanded.provenance_summary.get(PROVENANCE_BRIDGE_TITLE, 0)
        + expanded.provenance_summary.get(PROVENANCE_BRIDGE_PERMISSION, 0)
    ) >= 1
    assert expanded.cohort.attrs.get("bridge_expanded_cohort") is True


def test_engine_confirmed_bridge_sets_role_match_path():
    users = _ta_users_df()
    ref = _ta_reference_df()
    engine = AccessRecommendationEngine(min_confidence=0.1)
    recs = engine.recommend_for_hire(
        users_df=users,
        reference_df=ref,
        title="",
        department="Psychology",
        employee_type="Student",
        copy_from_netid="copyuser",
        confirmed_role_bridge={
            "access_template_title": "Teaching Assistant",
            "access_template_department": "Psychology",
            "employee_type": "student",
        },
    )
    assert recs.attrs["role_inference"]["role_match_path"] == MATCH_PATH_CONFIRMED_ROLE_BRIDGE


def test_supervisor_multiname_reference_cell_matching():
    assert supervisor_cell_matches_target("Dr Smith\nJane Doe", "jane doe") is True
    assert title_token_similarity("Teaching Assistant", "Learning Assistant") > 0.3


def test_no_bridge_candidates_when_strong_exact_registry_match():
    users = _ta_users_df()
    ref = _ta_reference_df()
    role = canonical_role_id(
        title="Teaching Assistant",
        department="Psychology",
        employee_type="Student",
        workforce_canonical=STUDENT,
    )
    resolution = generate_role_bridge_candidates(
        ticket_title="Teaching Assistant",
        department="Psychology",
        employee_type="Student",
        supervisor=None,
        copy_from_netid=None,
        current_role=role,
        reference_df=ref,
        users_df=users,
        reference_match_path="exact_title_dept",
        reference_permission_count=10,
    )
    assert resolution.debug.get("reason") == "strong_exact_match"
    assert not resolution.candidates
