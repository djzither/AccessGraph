import pandas as pd
import pytest

from DataLayer.peer_cohort import (
    _owns_sensitive_groups,
    build_peer_pool_from_anchor,
    build_target_user_row,
    contamination_stats_for_group,
    explain_peer_cohort_build,
    infer_workforce_type_from_groups,
    is_manager_of_others,
    is_supervisor_like,
    is_valid_peer_relationship,
    median_permission_count,
    normalize_groups,
    parse_manager_netid,
    peer_cohort_user_snapshot,
)
from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


def _normalize_group_name(value: object) -> str:
    return AccessRecommendationEngine._normalize_group_name(value)


def test_parse_manager_netid_extracts_cn_value():
    assert parse_manager_netid("CN=davedad7,OU=People,DC=byu,DC=local") == "davedad7"
    assert parse_manager_netid("") is None


def test_infer_workforce_type_from_groups_prefers_ad_markers():
    assert infer_workforce_type_from_groups("a.FULL TIME STAFF;Email") == "full_time"
    assert infer_workforce_type_from_groups("a.FULL TIME STUDENT;Email") == "student"
    assert infer_workforce_type_from_groups("Email", employee_type="Full Time") == "full_time"


def test_normalize_groups_splits_and_filters_invalid_entries():
    groups = normalize_groups("Email; Cannot find an object with name 'x'; VPN")
    assert groups == ["Email", "VPN"]


def test_normalize_groups_expands_numpy_array_to_distinct_strings():
    import numpy as np

    arr = np.array(
        ["DCE.CMP.DomainAdmins", "DCE-LocalAdmin", "DCE-DomainUsers"],
        dtype=object,
    )
    out = normalize_groups(arr)
    assert out == [
        "DCE.CMP.DomainAdmins",
        "DCE-LocalAdmin",
        "DCE-DomainUsers",
    ]


def test_normalize_groups_numpy_array_not_single_repr_blob():
    import numpy as np

    arr = np.array(["DCE.CMP.DomainAdmins", "DCE-DomainUsers"], dtype=object)
    out = normalize_groups(arr)
    assert len(out) == 2
    assert not any("dtype=" in g for g in out)


def test_owns_sensitive_groups_false_for_baseline_dce_groups():
    row = {
        "GroupsList": [
            "DCE.CMP.DomainAdmins",
            "DCE-LocalAdmin",
            "DCE-DomainUsers",
        ],
    }
    assert _owns_sensitive_groups(row) is False


def test_owns_sensitive_groups_detects_standalone_sensitive_token():
    row = {"GroupsList": ["Contoso-Privileged-Access"]}
    assert _owns_sensitive_groups(row) is True


def test_is_supervisor_like_still_true_for_manager_with_numpy_baseline_groups():
    import numpy as np

    users = pd.DataFrame(
        [
            {
                "SamAccountName": "mgr1",
                "Title": "IT Manager",
                "EmployeeType": "Full Time",
                "GroupsList": np.array(
                    ["DCE.CMP.DomainAdmins", "DCE-DomainUsers"],
                    dtype=object,
                ),
                "IsSupervisor": False,
            }
        ]
    )
    row = users.iloc[0]
    assert is_supervisor_like(row, users_df=users, cohort_median_group_count=5.0) is True


def test_is_supervisor_like_decision_notes_when_is_supervisor_truthy():
    users = pd.DataFrame(
        [
            {
                "SamAccountName": "u1",
                "Title": "Analyst",
                "EmployeeType": "Full Time",
                "GroupsList": ["DCE-DomainUsers"],
                "IsSupervisor": True,
            }
        ]
    )
    row = users.iloc[0]
    notes: list[str] = []
    assert is_supervisor_like(row, users_df=users, decision_notes=notes) is True
    assert "matched:IsSupervisor_column_truthy" in notes


def _synthetic_permission_groups(count: int) -> list[str]:
    return [f"PERM{i:03d}" for i in range(count)]


def test_computing_and_computer_specialist_share_peer_cohort():
    dept = "CE IT Help Desk"
    group_count = 20
    groups = _synthetic_permission_groups(group_count)
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
        {
            "SamAccountName": "peer.cs",
            "Title": "Computing Specialist",
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
    assert "peer.cs" in peer_ids


def test_ce_it_help_desk_high_group_count_not_supervisor_outlier_with_dept_median():
    dept = "CE IT Help Desk"
    groups = _synthetic_permission_groups(22)
    dept_users = pd.DataFrame(
        [
            {
                "SamAccountName": f"it{i}",
                "Title": "Computing Specialist",
                "Department": dept,
                "EmployeeType": "Full Time",
                "GroupsList": groups,
                "IsSupervisor": False,
            }
            for i in range(5)
        ]
    )
    org_padding = pd.DataFrame(
        [
            {
                "SamAccountName": f"pad{i}",
                "Title": "Clerk",
                "Department": f"Dept{i % 10}",
                "EmployeeType": "Full Time",
                "GroupsList": ["Email"],
                "IsSupervisor": False,
            }
            for i in range(40)
        ]
    )
    users_df = pd.concat([dept_users, org_padding], ignore_index=True)
    row = dept_users.iloc[0]
    dept_median = median_permission_count(dept_users)
    org_median = median_permission_count(users_df)
    assert org_median == 1.0
    assert dept_median >= 20.0
    assert (
        is_supervisor_like(
            row,
            users_df=users_df,
            cohort_median_group_count=dept_median,
        )
        is False
    )
    assert (
        is_supervisor_like(
            row,
            users_df=users_df,
            cohort_median_group_count=org_median,
        )
        is True
    )


def test_ce_it_help_desk_computing_specialist_cohort_not_collapsed_by_global_median():
    """Regression: org-wide median=1 must not shrink IT specialist peer pool to ~2."""
    dept = "CE IT Help Desk"
    group_count = 20
    groups = _synthetic_permission_groups(group_count)

    specialist_rows = []
    specialist_rows.append(
        {
            "SamAccountName": "anchor",
            "Title": "Computing Specialist",
            "Department": dept,
            "EmployeeType": "Full Time",
            "GroupsList": groups,
            "Manager": "",
            "IsSupervisor": False,
        }
    )
    specialist_rows.append(
        {
            "SamAccountName": "peer.comp",
            "Title": "Computer Specialist",
            "Department": dept,
            "EmployeeType": "Full Time",
            "GroupsList": groups,
            "Manager": "",
            "IsSupervisor": False,
        }
    )
    for i in range(2, 8):
        specialist_rows.append(
            {
                "SamAccountName": f"peer{i}",
                "Title": "Computing Specialist" if i % 2 == 0 else "Computer Specialist",
                "Department": dept,
                "EmployeeType": "Full Time",
                "GroupsList": groups,
                "Manager": "",
                "IsSupervisor": False,
            }
        )
    org_padding = [
        {
            "SamAccountName": f"pad{i}",
            "Title": "Clerk",
            "Department": f"Other {i}",
            "EmployeeType": "Full Time",
            "GroupsList": ["Email"],
            "IsSupervisor": False,
        }
        for i in range(50)
    ]
    users_df = pd.DataFrame(specialist_rows + org_padding)
    anchor = users_df.iloc[0]
    target = build_target_user_row(
        title="Computing Specialist",
        department=dept,
        employee_type="Full Time",
    )
    result = build_peer_pool_from_anchor(users_df, anchor, target)
    assert result.peer_pool_size >= 6
    peer_ids = set(result.peer_pool["SamAccountName"].astype(str))
    assert len(peer_ids) >= 6
    assert "peer.comp" in peer_ids
    assert all(f"peer{i}" in peer_ids for i in range(2, 8))


def test_computing_specialist_peer_pool_not_collapsed_by_numpy_groupslist():
    """Regression: ndarray GroupsList must not stringify to one blob (supervisor false positives)."""
    import numpy as np

    baseline = np.array(
        ["DCE.CMP.DomainAdmins", "DCE-LocalAdmin", "DCE-DomainUsers"],
        dtype=object,
    )
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "anchor1",
                "DisplayName": "Anchor",
                "Title": "Computing Specialist",
                "Department": "Information Technology",
                "EmployeeType": "Full Time",
                "GroupsList": baseline,
                "Manager": "",
            },
            {
                "SamAccountName": "peer1",
                "DisplayName": "Peer One",
                "Title": "Computing Specialist",
                "Department": "Information Technology",
                "EmployeeType": "Full Time",
                "GroupsList": baseline,
                "Manager": "",
            },
            {
                "SamAccountName": "peer2",
                "DisplayName": "Peer Two",
                "Title": "Computing Specialist",
                "Department": "Information Technology",
                "EmployeeType": "Full Time",
                "GroupsList": baseline,
                "Manager": "",
            },
        ]
    )
    anchor = users_df.iloc[0]
    target = build_target_user_row(
        title="Computing Specialist",
        department="Information Technology",
        employee_type="Full Time",
    )
    result = build_peer_pool_from_anchor(users_df, anchor, target)
    assert result.peer_pool_size >= 2
    peer_ids = set(result.peer_pool["SamAccountName"].astype(str))
    assert "peer1" in peer_ids
    assert "peer2" in peer_ids


def test_is_supervisor_like_detects_title_and_manager_signals():
    row = {
        "Title": "IT Help Desk Manager",
        "EmployeeType": "Student",
        "GroupsList": ["Email"],
        "IsSupervisor": False,
    }
    assert is_supervisor_like(row) is True

    student_row = {
        "Title": "Student Worker",
        "EmployeeType": "Student",
        "GroupsList": ["Email"],
        "IsSupervisor": False,
    }
    assert is_supervisor_like(student_row) is False

    full_time_ic = {
        "SamAccountName": "ft.ic",
        "Title": "Technician",
        "EmployeeType": "Full Time",
        "GroupsList": ["a.FULL TIME STAFF", "VPN"],
        "IsSupervisor": False,
    }
    assert is_supervisor_like(full_time_ic) is False


def test_is_manager_of_others_matches_manager_dn_to_samaccountname():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "boss1",
                "Manager": "",
                "GroupsList": ["a.FULL TIME STAFF"],
            },
            {
                "SamAccountName": "student.peer",
                "Manager": "CN=boss1,OU=People,DC=byu,DC=local",
                "GroupsList": ["a.FULL TIME STUDENT"],
            },
        ]
    )
    assert is_manager_of_others(users_df, "boss1") is True
    assert is_manager_of_others(users_df, "BOSS1") is True
    assert is_manager_of_others(users_df, "student.peer") is False


def test_is_valid_peer_relationship_excludes_supervisors_for_student_targets():
    target = build_target_user_row(
        title="Student Worker",
        department="IT",
        employee_type="Student",
    )
    anchor = {
        "SamAccountName": "student.anchor",
        "Title": "Student Worker",
        "Department": "IT",
        "EmployeeType": "Student",
        "GroupsList": ["Email"],
    }
    supervisor_peer = {
        "SamAccountName": "mgr1",
        "Title": "IT Manager",
        "Department": "IT",
        "EmployeeType": "Full Time",
        "GroupsList": ["AdminGroup"],
        "IsSupervisor": True,
    }
    student_peer = {
        "SamAccountName": "stu2",
        "Title": "Student Worker",
        "Department": "IT",
        "EmployeeType": "Student",
        "GroupsList": ["Email"],
    }

    assert is_valid_peer_relationship(target, anchor, student_peer) is True
    assert is_valid_peer_relationship(target, anchor, supervisor_peer) is False


def test_build_peer_pool_from_anchor_excludes_supervisors_for_student_anchor():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "DisplayName": "Student Anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["Email"],
            },
            {
                "SamAccountName": "student.peer",
                "DisplayName": "Student Peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["Email"],
            },
            {
                "SamAccountName": "mgr.peer",
                "DisplayName": "Manager Peer",
                "Title": "IT Manager",
                "Department": "IT",
                "EmployeeType": "Full Time",
                "GroupsList": ["AdminGroup"],
                "IsSupervisor": True,
            },
        ]
    )
    target = build_target_user_row(
        title="Student Worker",
        department="IT",
        employee_type="Student",
    )
    result = build_peer_pool_from_anchor(
        users_df=users_df,
        anchor_user_row=users_df.iloc[0],
        target_user_row=target,
    )

    peer_ids = set(result.peer_pool["SamAccountName"].astype(str))
    assert peer_ids == {"student.anchor", "student.peer"}
    assert "mgr.peer" in result.supervisor_users_excluded or "mgr.peer" in result.full_time_excluded_for_student_target


def test_student_target_excludes_manager_of_others():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
                "Manager": "CN=boss1,OU=People,DC=byu,DC=local",
            },
            {
                "SamAccountName": "student.peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
                "Manager": "CN=boss1,OU=People,DC=byu,DC=local",
            },
            {
                "SamAccountName": "boss1",
                "Title": "IT Manager",
                "Department": "IT",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "AdminGroup"],
                "Manager": "",
            },
            {
                "SamAccountName": "student.other",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
                "Manager": "CN=boss1,OU=People,DC=byu,DC=local",
            },
        ]
    )
    users_df.loc[users_df["SamAccountName"] == "student.other", "Manager"] = (
        "CN=boss1,OU=People,DC=byu,DC=local"
    )
    assert is_manager_of_others(users_df, "boss1") is True

    target = build_target_user_row(
        title="Student Worker",
        department="IT",
        employee_type="Student",
        groups="a.FULL TIME STUDENT;Email",
    )
    result = build_peer_pool_from_anchor(
        users_df=users_df,
        anchor_user_row=users_df.iloc[0],
        target_user_row=target,
    )
    peer_ids = set(result.peer_pool["SamAccountName"].astype(str))
    assert "boss1" not in peer_ids
    assert "boss1" in result.manager_of_others_excluded or "boss1" in result.supervisor_users_excluded


def test_full_time_target_excludes_student_peers():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "ft.anchor",
                "Title": "Technician",
                "Department": "IT",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "VPN"],
            },
            {
                "SamAccountName": "ft.peer",
                "Title": "Technician",
                "Department": "IT",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "VPN"],
            },
            {
                "SamAccountName": "student.peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
        ]
    )
    target = build_target_user_row(
        title="Technician",
        department="IT",
        employee_type="Full Time",
        groups="a.FULL TIME STAFF;VPN",
    )
    result = build_peer_pool_from_anchor(
        users_df=users_df,
        anchor_user_row=users_df.iloc[0],
        target_user_row=target,
    )
    peer_ids = set(result.peer_pool["SamAccountName"].astype(str))
    assert "student.peer" not in peer_ids
    assert "student.peer" in result.students_excluded_for_full_time_target


def test_copy_from_supervisor_for_student_target_sets_anchor_mismatch_flag():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "student.peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "mgr.peer",
                "Title": "IT Manager",
                "Department": "IT",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "AdminGroup", "Email"],
                "IsSupervisor": True,
            },
        ]
    )
    target = build_target_user_row(
        title="Student Worker",
        department="IT",
        employee_type="Student",
        groups="a.FULL TIME STUDENT;Email",
    )
    result = build_peer_pool_from_anchor(
        users_df=users_df,
        anchor_user_row=users_df.iloc[2],
        target_user_row=target,
    )
    assert result.anchor_mismatch_flag is True
    assert "mgr.peer" not in set(result.peer_pool["SamAccountName"].astype(str))


def test_same_manager_student_peers_are_preferred_when_available():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
                "Manager": "CN=boss1,OU=People,DC=byu,DC=local",
            },
            {
                "SamAccountName": "student.sameboss",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
                "Manager": "CN=boss1,OU=People,DC=byu,DC=local",
            },
            {
                "SamAccountName": "student.otherboss",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
                "Manager": "CN=boss2,OU=People,DC=byu,DC=local",
            },
        ]
    )
    target = build_target_user_row(
        title="Student Worker",
        department="IT",
        employee_type="Student",
        groups="a.FULL TIME STUDENT;Email",
        manager="CN=boss1,OU=People,DC=byu,DC=local",
    )
    result = build_peer_pool_from_anchor(
        users_df=users_df,
        anchor_user_row=users_df.iloc[0],
        target_user_row=target,
    )
    peer_ids = result.peer_pool["SamAccountName"].astype(str).tolist()
    assert peer_ids.index("student.sameboss") < peer_ids.index("student.otherboss")


def test_contamination_stats_flag_supervisor_heavy_permissions():
    peer_pool = pd.DataFrame(
        [
            {
                "SamAccountName": "stu1",
                "DisplayName": "Student One",
                "Title": "Student Worker",
                "EmployeeType": "Student",
                "GroupsList": ["Email"],
            },
            {
                "SamAccountName": "mgr1",
                "DisplayName": "Manager One",
                "Title": "IT Manager",
                "EmployeeType": "Full Time",
                "GroupsList": ["AdminGroup", "Email"],
                "IsSupervisor": True,
            },
            {
                "SamAccountName": "mgr2",
                "DisplayName": "Manager Two",
                "Title": "IT Manager",
                "EmployeeType": "Full Time",
                "GroupsList": ["AdminGroup"],
                "IsSupervisor": True,
            },
        ]
    )

    email_stats = contamination_stats_for_group(
        peer_pool,
        "Email",
        normalizer=_normalize_group_name,
    )
    admin_stats = contamination_stats_for_group(
        peer_pool,
        "AdminGroup",
        normalizer=_normalize_group_name,
    )

    assert email_stats.peer_student_support_count == 1
    assert email_stats.supervisor_support_count == 1
    assert email_stats.supervisor_contamination_flag is False
    assert admin_stats.peer_student_support_count == 0
    assert admin_stats.supervisor_support_count == 2
    assert admin_stats.supervisor_contamination_flag is True


def test_student_recommendations_do_not_auto_approve_supervisor_only_groups():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "DisplayName": "Student Anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["Email"],
            },
            {
                "SamAccountName": "student.peer",
                "DisplayName": "Student Peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["Email"],
            },
            {
                "SamAccountName": "mgr.peer",
                "DisplayName": "Manager Peer",
                "Title": "IT Manager",
                "Department": "IT",
                "EmployeeType": "Full Time",
                "GroupsList": ["AdminGroup", "Email"],
                "IsSupervisor": True,
            },
        ]
    )
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

    engine = AccessRecommendationEngine(min_confidence=0.4)
    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Student Worker",
        department="IT",
        employee_type="Student",
        supervisor=None,
        copy_from_netid="student.anchor",
        new_hire_netid=None,
    )

    by_group = recommendations.set_index("GroupName")
    assert "Email" in by_group.index
    assert by_group.loc["Email", "FinalDecision"] in {"Strong Recommend", "Auto Assign", "Suggest"}
    if "AdminGroup" in by_group.index:
        assert by_group.loc["AdminGroup", "FinalDecision"] in {"Manual Review", "Ignore", "Suggest"}


def test_full_time_copy_from_supervisor_keeps_supervisor_comparison():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "deg11",
                "DisplayName": "Deg Eleven",
                "Title": "Manager, Tech Support",
                "Department": "CE IT Help Desk",
                "EmployeeType": "Full Time",
                "GroupsList": ["VPN", "Email"],
                "IsSupervisor": True,
            },
            {
                "SamAccountName": "peer1",
                "DisplayName": "Peer One",
                "Title": "Technician",
                "Department": "CE IT Help Desk",
                "EmployeeType": "Full Time",
                "GroupsList": ["VPN", "Email"],
                "IsSupervisor": False,
            },
            {
                "SamAccountName": "peer2",
                "DisplayName": "Peer Two",
                "Title": "Analyst",
                "Department": "CE IT Help Desk",
                "EmployeeType": "Full Time",
                "GroupsList": ["VPN"],
                "IsSupervisor": False,
            },
        ]
    )

    engine = AccessRecommendationEngine(min_confidence=0.4)
    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=pd.DataFrame(
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
        ),
        title="IT Help Desk Supervisor",
        department="IT",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid="deg11",
        new_hire_netid=None,
    )

    by_group = recommendations.set_index("GroupName")
    assert by_group.loc["VPN", "TotalUsersInRole"] >= 2
    assert by_group.loc["VPN", "ADConfidence"] == pytest.approx(1.0, rel=1e-6)


def test_anchor_mismatch_blocks_student_auto_approve_from_supervisor_copy_from():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "student.peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "mgr.peer",
                "Title": "IT Manager",
                "Department": "IT",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "AdminGroup"],
                "IsSupervisor": True,
            },
        ]
    )
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
    engine = AccessRecommendationEngine(min_confidence=0.4)
    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Student Worker",
        department="IT",
        employee_type="Student",
        supervisor=None,
        copy_from_netid="mgr.peer",
        new_hire_netid=None,
    )
    assert bool(recommendations["AnchorMismatchFlag"].iloc[0]) is True
    if "AdminGroup" in set(recommendations["GroupName"]):
        admin_row = recommendations[recommendations["GroupName"] == "AdminGroup"].iloc[0]
        assert admin_row["FinalDecision"] != "Auto Assign"


def test_ml_recommendations_use_locked_peer_pool_without_workforce_fallback():
    from MLLayer.recommender import MLRecommender

    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "student.peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "ft.other",
                "Title": "Technician",
                "Department": "IT",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "VPN"],
            },
        ]
    )
    cohort = users_df.iloc[:2].copy()
    cohort.attrs = {"peer_pool_locked": True, "workforce_fallback": False}
    recs = MLRecommender(users_df).recommend_for_peer_cohort(
        cohort_df=cohort,
        min_support=2,
        workforce_segment="STUDENT",
        peer_aggregate_fallback=False,
        respect_anchor_pool=True,
    )
    assert not recs.empty
    assert bool(recs.iloc[0]["MLWorkforcePoolFallback"]) is False
    assert recs.iloc[0]["MLComparedUsers"] == 2
    assert set(recs.iloc[0]["NearestUsers"].split(", ")) == {"student.anchor", "student.peer"}


def test_full_time_support_count_metadata_and_student_score_cap():
    peer_pool = pd.DataFrame(
        [
            {
                "SamAccountName": "stu1",
                "Title": "Student Worker",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "VPN"],
            },
            {
                "SamAccountName": "ft1",
                "Title": "Technician",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "VPN"],
            },
            {
                "SamAccountName": "ft2",
                "Title": "Technician",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "VPN"],
            },
            {
                "SamAccountName": "ft3",
                "Title": "Technician",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "VPN"],
            },
        ]
    )
    target = build_target_user_row(
        title="Student Worker",
        department="IT",
        employee_type="Student",
        groups="a.FULL TIME STUDENT;VPN",
    )
    stats = contamination_stats_for_group(
        peer_pool,
        "VPN",
        normalizer=_normalize_group_name,
        target_row=target,
        users_df=peer_pool,
    )
    assert stats.full_time_support_count == 3
    assert stats.peer_student_support_count == 1
    assert stats.as_row_metadata()["FullTimeSupportCount"] == 3

    engine = AccessRecommendationEngine(min_confidence=0.4)
    row = {
        "EmployeeTypeClean": "student",
        "InReferenceSheet": False,
        "AmbiguousReferenceTemplate": False,
        "CohortReliability": 0.8,
        "GlobalGroupRate": 0.1,
        "CohortWorkforceFallback": False,
        "MLWorkforcePoolFallback": False,
        "SupervisorContaminationFlag": False,
        "AnchorMismatchFlag": False,
        "CopyFromUserHasIt": False,
        "ADConfidence": 0.99,
        "MLConfidence": 0.99,
        "SupportRatio": 0.9,
        "RiskLevel": "Low",
        "FullTimeSupportCount": stats.full_time_support_count,
        "StudentPeerSupportCount": stats.peer_student_support_count,
        "PeerStudentSupportCount": stats.peer_student_support_count,
    }
    assert engine._score_row(row) <= 0.45


def test_anchor_mismatch_regression_blocks_auto_approve_and_copy_from_boost():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "student.peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "mgr.peer",
                "Title": "IT Manager",
                "Department": "IT",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "AdminGroup", "Email"],
                "IsSupervisor": True,
            },
        ]
    )
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
    engine = AccessRecommendationEngine(min_confidence=0.4)
    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Student Worker",
        department="IT",
        employee_type="Student",
        supervisor=None,
        copy_from_netid="mgr.peer",
        new_hire_netid=None,
    )
    assert bool(recommendations["AnchorMismatchFlag"].iloc[0]) is True
    for group_name in ("AdminGroup", "Email"):
        row = recommendations[recommendations["GroupName"] == group_name].iloc[0]
        assert row["FinalDecision"] not in {"Auto Assign", "Strong Recommend", "Suggest"}
        assert row["FinalScore"] < 0.70


def test_student_supervisor_only_ml_evidence_not_suggest_or_auto_approve():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": f"mgr{i}",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Full Time",
                "GroupsList": ["a.FULL TIME STAFF", "AdminGroup"],
                "IsSupervisor": True,
            }
            for i in range(1, 8)
        ]
    )
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
    engine = AccessRecommendationEngine(min_confidence=0.4)
    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Student Worker",
        department="IT",
        employee_type="Student",
        supervisor=None,
        copy_from_netid=None,
        new_hire_netid=None,
    )
    admin_row = recommendations[recommendations["GroupName"] == "AdminGroup"].iloc[0]
    assert admin_row["MLConfidence"] > 0
    assert admin_row["FinalDecision"] not in {"Auto Assign", "Strong Recommend", "Suggest"}
    assert admin_row["FinalScore"] < 0.50


def test_recommend_for_hire_exposes_cohort_diagnostics_metadata():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "student.peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
        ]
    )
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
    engine = AccessRecommendationEngine(min_confidence=0.4)
    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Student Worker",
        department="IT",
        employee_type="Student",
        supervisor=None,
        copy_from_netid="student.anchor",
        new_hire_netid=None,
    )
    row = recommendations.iloc[0]
    assert row["CohortUsedForScoring"] == "anchor_peer_pool"
    assert row["CohortFallbackLevel"] == 0
    assert row["CohortUsedMix"] == row["CohortEmployeeTypeMix"]
    assert row["CohortUsedMix"] == "Student=2"


@pytest.mark.parametrize(
    "title",
    [
        "Computing Specialist",
        "IT Specialist",
        "Data Analyst",
        "Event Coordinator",
        "Help Desk Admin",
        "Team Lead",
    ],
)
def test_is_supervisor_like_does_not_flag_common_technical_titles(title):
    row = {
        "SamAccountName": "tech.user",
        "Title": title,
        "EmployeeType": "Full Time",
        "GroupsList": ["a.FULL TIME STAFF", "VPN"],
        "IsSupervisor": False,
    }
    users = pd.DataFrame([row])
    assert (
        is_supervisor_like(
            row,
            users_df=users,
            cohort_median_group_count=10.0,
        )
        is False
    )


@pytest.mark.parametrize(
    "title",
    [
        "IT Help Desk Manager",
        "Shipping Supervisor",
        "Director of Operations",
        "Assistant Director Programs",
        "Systems Administrator",
    ],
)
def test_is_supervisor_like_detects_management_titles(title):
    row = {
        "SamAccountName": "mgr.user",
        "Title": title,
        "EmployeeType": "Full Time",
        "GroupsList": ["a.FULL TIME STAFF", "VPN"],
        "IsSupervisor": False,
    }
    users = pd.DataFrame([row])
    assert is_supervisor_like(row, users_df=users, cohort_median_group_count=10.0) is True


def test_peer_cohort_user_snapshot_includes_expected_fields():
    row = pd.Series(
        {
            "SamAccountName": "u1",
            "Title": "Analyst",
            "Department": "IT",
            "EmployeeType": "Student",
            "Manager": "CN=boss,OU=People,DC=x",
            "IsSupervisor": False,
        }
    )
    snap = peer_cohort_user_snapshot(row)
    assert snap["SamAccountName"] == "u1"
    assert snap["Title"] == "Analyst"
    assert snap["Department"] == "IT"
    assert snap["EmployeeType"] == "Student"
    assert "boss" in snap["Manager"].lower()
    assert snap["IsSupervisor"] is False


def test_build_peer_pool_cohort_diagnostics_records_removals():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "DisplayName": "Student Anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "super.peer",
                "DisplayName": "Super Peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
                "IsSupervisor": True,
            },
            {
                "SamAccountName": "student.peer",
                "DisplayName": "Student Peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
        ]
    )
    anchor = users_df.iloc[0]
    target = build_target_user_row(
        title="Student Worker",
        department="IT",
        employee_type="Student",
    )
    result = build_peer_pool_from_anchor(
        users_df,
        anchor,
        target,
        cohort_diagnostics=True,
    )
    diag = result.cohort_filter_diagnostics
    assert diag is not None
    assert diag["scoped_candidate_count"] >= 2
    assert len(diag["scoped_candidates"]) == diag["scoped_candidate_count"]
    by_netid = {r["SamAccountName"]: r for r in diag["removals"]}
    assert "super.peer" in by_netid
    assert (
        by_netid["super.peer"]["exclusion_rule"]
        == "invalid_peer_relationship_supervisor_like_candidate"
    )
    assert any(p["SamAccountName"] == "student.peer" for p in diag["final_peers"])
    explain_diag = explain_peer_cohort_build(users_df, anchor, target)
    assert explain_diag is not None
    assert explain_diag["removals"] == diag["removals"]


def test_recommend_for_hire_merged_attrs_include_cohort_filter_diagnostics_when_requested():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "student.anchor",
                "DisplayName": "Student Anchor",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
            {
                "SamAccountName": "student.peer",
                "DisplayName": "Student Peer",
                "Title": "Student Worker",
                "Department": "IT",
                "EmployeeType": "Student",
                "GroupsList": ["a.FULL TIME STUDENT", "Email"],
            },
        ]
    )
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
    engine = AccessRecommendationEngine(min_confidence=0.4)
    recs = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Student Worker",
        department="IT",
        employee_type="Student",
        supervisor=None,
        copy_from_netid="student.anchor",
        cohort_diagnostics=True,
    )
    diag = recs.attrs.get("cohort_filter_diagnostics")
    assert diag is not None
    assert diag["final_peer_count"] >= 1
    assert diag["scoped_candidate_count"] >= 1
