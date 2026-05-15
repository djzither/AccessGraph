import pandas as pd
import pytest

from DataLayer.canonical_role import MATCH_PATH_REGISTRY
from DataLayer.role_inference import (
    MATCH_PATH_INFERRED_COPY_FROM,
    MATCH_PATH_INFERRED_DEPT_WORKFORCE,
    MATCH_PATH_INFERRED_SUPERVISOR,
    MATCH_PATH_TITLE_CONFLICT_PREFERS_COPY_FROM,
    infer_role_from_supervisor,
    resolve_onboarding_role,
    split_supervisor_tokens,
    supervisor_cell_matches_target,
)
from DataLayer.workforce_type import STUDENT
from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


def _ce_it_student_users() -> pd.DataFrame:
    groups = [f"PERM{i:03d}" for i in range(8)]
    return pd.DataFrame(
        [
            {
                "SamAccountName": "ag877",
                "Title": "Computing Specialist",
                "Department": "CE IT Help Desk",
                "EmployeeType": "Student",
                "GroupsList": groups,
                "Manager": "",
                "IsSupervisor": False,
            },
            {
                "SamAccountName": "peer1",
                "Title": "Computer Specialist",
                "Department": "CE IT Help Desk",
                "EmployeeType": "Student",
                "GroupsList": groups,
                "Manager": "CN=ag877,OU=People,DC=byu,DC=local",
                "IsSupervisor": False,
            },
            {
                "SamAccountName": "peer2",
                "Title": "Student Worker 5",
                "Department": "CE IT Help Desk",
                "EmployeeType": "Student",
                "GroupsList": groups,
                "Manager": "CN=ag877,OU=People,DC=byu,DC=local",
                "IsSupervisor": False,
            },
        ]
    )


def test_blank_title_copy_from_ag877_resolves_ce_it_student_support():
    users = _ce_it_student_users()
    res = resolve_onboarding_role(
        title="",
        department="CE IT Help Desk",
        employee_type="Student",
        copy_from_netid="ag877",
        users_df=users,
    )
    assert res.role.canonical_role_id == "role:ce_it_helpdesk_student_support"
    assert res.role.match_path == MATCH_PATH_INFERRED_COPY_FROM
    assert res.inference_debug.get("copy_from_netid") == "ag877"


def test_blank_title_without_copy_from_uses_department_workforce_cohort():
    users = _ce_it_student_users()
    res = resolve_onboarding_role(
        title="",
        department="CE IT Help Desk",
        employee_type="Student",
        users_df=users,
    )
    assert res.role.match_path == MATCH_PATH_INFERRED_DEPT_WORKFORCE
    assert res.role.canonical_role_id.startswith("cohort:")


def test_wrong_title_with_copy_from_warns_and_prefers_copy_from():
    users = _ce_it_student_users()
    res = resolve_onboarding_role(
        title="Receptionist",
        department="CE IT Help Desk",
        employee_type="Student",
        copy_from_netid="ag877",
        users_df=users,
    )
    assert res.role.canonical_role_id == "role:ce_it_helpdesk_student_support"
    assert res.role.match_path == MATCH_PATH_TITLE_CONFLICT_PREFERS_COPY_FROM
    assert "copy-from" in res.warning.lower()


def test_supervisor_inference_resolves_ce_it_student_support():
    users = _ce_it_student_users()
    res = resolve_onboarding_role(
        title="",
        department="CE IT Help Desk",
        employee_type="Student",
        supervisor="ag877",
        users_df=users,
    )
    assert res.role.canonical_role_id == "role:ce_it_helpdesk_student_support"
    assert res.role.match_path == MATCH_PATH_INFERRED_SUPERVISOR
    assert not res.ambiguous


def test_mixed_supervisor_cohort_returns_ambiguity():
    users = pd.DataFrame(
        [
            {
                "SamAccountName": "mgr",
                "Title": "Manager",
                "Department": "CE IT Help Desk",
                "EmployeeType": "Student",
                "GroupsList": ["Email"],
                "Manager": "",
            },
            {
                "SamAccountName": "r1",
                "Title": "Computing Specialist",
                "Department": "CE IT Help Desk",
                "EmployeeType": "Student",
                "GroupsList": ["Email"],
                "Manager": "CN=mgr,OU=People,DC=x",
            },
            {
                "SamAccountName": "r2",
                "Title": "Receptionist",
                "Department": "CE IT Help Desk",
                "EmployeeType": "Student",
                "GroupsList": ["Email"],
                "Manager": "CN=mgr,OU=People,DC=x",
            },
        ]
    )
    res = infer_role_from_supervisor(
        users,
        supervisor="mgr",
        department="CE IT Help Desk",
        employee_type="Student",
        workforce_canonical=STUDENT,
    )
    assert res.ambiguous is True
    assert len(res.candidate_roles) >= 2


def test_supervisor_cell_splits_multiname_reference():
    assert split_supervisor_tokens("alice\nbob") == ["alice", "bob"]
    assert supervisor_cell_matches_target("alice\nbob", "bob") is True
    assert supervisor_cell_matches_target("alice; bob", "alice") is True
    assert supervisor_cell_matches_target("alice, bob", "carol") is False


def test_provided_title_still_uses_registry_path():
    users = _ce_it_student_users()
    res = resolve_onboarding_role(
        title="Computing Specialist",
        department="CE IT Help Desk",
        employee_type="Student",
        users_df=users,
    )
    assert res.role.match_path == MATCH_PATH_REGISTRY
    assert res.confidence_ceiling == 1.0


def test_engine_applies_confidence_ceiling_for_inferred_role():
    users = _ce_it_student_users()
    reference_df = pd.DataFrame(
        columns=[
            "EmployeeType",
            "JobTitle",
            "Department",
            "Supervisor",
            "AccessCategory",
            "AccessName",
            "AccessNameClean",
            "SourceFile",
        ]
    )
    engine = AccessRecommendationEngine(min_confidence=0.1)
    recs = engine.recommend_for_hire(
        users_df=users,
        reference_df=reference_df,
        title="",
        department="CE IT Help Desk",
        employee_type="Student",
        copy_from_netid="ag877",
    )
    assert recs.attrs["role_inference"]["role_match_path"] == MATCH_PATH_INFERRED_COPY_FROM
    if not recs.empty and "FinalScore" in recs.columns:
        assert recs["FinalScore"].max() <= 0.85 + 1e-6
