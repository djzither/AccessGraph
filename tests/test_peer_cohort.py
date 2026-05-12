import pandas as pd
import pytest

from DataLayer.peer_cohort import (
    build_peer_pool_from_anchor,
    build_target_user_row,
    contamination_stats_for_group,
    infer_workforce_type_from_groups,
    is_manager_of_others,
    is_supervisor_like,
    is_valid_peer_relationship,
    normalize_groups,
    parse_manager_netid,
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
