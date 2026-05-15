import pandas as pd

from DataLayer.canonical_role import (
    MATCH_PATH_EXACT_FALLBACK,
    MATCH_PATH_REGISTRY,
    canonical_role_id,
    cluster_reference_candidates,
)
from DataLayer.peer_cohort import build_peer_pool_from_anchor, build_target_user_row
from DataLayer.workforce_type import FULL_TIME, STUDENT


def test_cluster_reference_candidates_ce_it_student_support():
    role_id = "role:ce_it_helpdesk_student_support"
    pairs = cluster_reference_candidates(role_id)
    expected = {
        ("computing specialist", "information technology"),
        ("computer specialist", "information technology"),
        ("student worker 5", "information technology"),
        ("computing specialist", "ce it help desk"),
        ("computer specialist", "ce it help desk"),
        ("student worker 5", "ce it help desk"),
    }
    assert expected <= pairs


def test_cluster_reference_candidates_unknown_role_is_empty():
    assert cluster_reference_candidates("role:does_not_exist") == set()


def test_ce_it_student_title_variants_share_canonical_role():
    dept = "CE IT Help Desk"
    titles = ("Computing Specialist", "Computer Specialist", "Student Worker 5")
    role_ids = {
        canonical_role_id(
            title=title,
            department=dept,
            employee_type="Student",
            workforce_canonical=STUDENT,
        ).canonical_role_id
        for title in titles
    }
    assert len(role_ids) == 1
    assert role_ids.pop() == "role:ce_it_helpdesk_student_support"
    assert (
        canonical_role_id(
            title="Computing Specialist",
            department=dept,
            employee_type="Student",
            workforce_canonical=STUDENT,
        ).match_path
        == MATCH_PATH_REGISTRY
    )


def test_same_title_different_department_uses_different_role():
    title = "Computing Specialist"
    student = STUDENT
    a = canonical_role_id(
        title=title,
        department="CE IT Help Desk",
        workforce_canonical=student,
    )
    b = canonical_role_id(
        title=title,
        department="Financial Services",
        workforce_canonical=student,
    )
    assert a.canonical_role_id != b.canonical_role_id
    assert a.match_path == MATCH_PATH_REGISTRY
    assert b.match_path == MATCH_PATH_EXACT_FALLBACK


def test_same_title_different_workforce_uses_different_role():
    dept = "CE IT Help Desk"
    title = "Computing Specialist"
    student = canonical_role_id(
        title=title,
        department=dept,
        workforce_canonical=STUDENT,
    )
    full_time = canonical_role_id(
        title=title,
        department=dept,
        workforce_canonical=FULL_TIME,
    )
    assert student.canonical_role_id != full_time.canonical_role_id
    assert student.canonical_role_id == "role:ce_it_helpdesk_student_support"
    assert full_time.canonical_role_id == "role:ce_it_helpdesk_fulltime_specialist"


def test_student_worker_in_finance_does_not_join_ce_it_cluster():
    ce = canonical_role_id(
        title="Student Worker 5",
        department="CE IT Help Desk",
        workforce_canonical=STUDENT,
    )
    finance = canonical_role_id(
        title="Student Worker 5",
        department="Finance",
        workforce_canonical=STUDENT,
    )
    assert ce.match_path == MATCH_PATH_REGISTRY
    assert finance.match_path == MATCH_PATH_EXACT_FALLBACK
    assert ce.canonical_role_id != finance.canonical_role_id


def test_peer_pool_groups_computing_and_computer_specialist_by_canonical_role():
    dept = "CE IT Help Desk"
    groups = [f"PERM{i:03d}" for i in range(20)]
    rows = [
        {
            "SamAccountName": "anchor",
            "Title": "Computing Specialist",
            "Department": dept,
            "EmployeeType": "Full Time",
            "GroupsList": groups,
            "Manager": "",
            "IsSupervisor": False,
        },
        {
            "SamAccountName": "peer.comp",
            "Title": "Computer Specialist",
            "Department": dept,
            "EmployeeType": "Full Time",
            "GroupsList": groups,
            "Manager": "",
            "IsSupervisor": False,
        },
    ]
    users_df = pd.DataFrame(rows)
    anchor = users_df.iloc[0]
    target = build_target_user_row(
        title="Computing Specialist",
        department=dept,
        employee_type="Full Time",
    )
    result = build_peer_pool_from_anchor(users_df, anchor, target)
    peer_ids = set(result.peer_pool["SamAccountName"].astype(str))
    assert "peer.comp" in peer_ids
