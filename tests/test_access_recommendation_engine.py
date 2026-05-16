import pandas as pd
import pytest

from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


class _StubTitleMatcher:
    def best_match(self, query_title, candidate_titles):
        for title in candidate_titles:
            title_clean = str(title).lower()
            if "assistant director" in title_clean and "fsy" in title_clean:
                return title, 0.91
        return None, 0.2


class _StubPrefersSupportTechnicianWhenPresent:
    """Picks Support Technician if the embed pool includes it; else first candidate."""

    def best_match(self, query_title, candidate_titles):
        titles = list(candidate_titles)
        for t in titles:
            if str(t).strip().lower() == "support technician":
                return t, 0.91
        return (titles[0], 0.5) if titles else (None, 0.0)


def test_computing_specialist_ce_it_help_desk_resolves_information_technology_reference():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "u1",
                "DisplayName": "User One",
                "Title": "Computing Specialist",
                "Department": "CE IT Help Desk",
                "GroupsList": ["IT.Template", "FS.Template"],
            },
        ]
    )
    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Computing Specialist",
                "Department": "Information Technology",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "IT.Template",
                "SourceFile": "full_time_employee_access.xlsx",
            },
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Accountant",
                "Department": "Financial Services",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "FS.Template",
                "SourceFile": "full_time_employee_access.xlsx",
            },
        ]
    )
    engine = AccessRecommendationEngine(min_confidence=0.4)
    recs = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Computing Specialist",
        department="CE IT Help Desk",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid=None,
        new_hire_netid=None,
    )
    by_group = recs.set_index("GroupName")
    assert bool(by_group.loc["IT.Template", "InReferenceSheet"]) is True
    assert bool(by_group.loc["FS.Template", "InReferenceSheet"]) is False


def test_reference_embed_fallback_does_not_cross_departments():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "u1",
                "DisplayName": "User One",
                "Title": "Contract Analyst",
                "Department": "Financial Services",
                "GroupsList": ["IT.Template", "FS.Template"],
            },
        ]
    )
    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Support Technician",
                "Department": "Information Technology",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "IT.Template",
                "SourceFile": "full_time_employee_access.xlsx",
            },
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Legacy Analyst",
                "Department": "Financial Services",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "FS.Template",
                "SourceFile": "full_time_employee_access.xlsx",
            },
        ]
    )
    engine = AccessRecommendationEngine(
        min_confidence=0.4,
        title_matcher=_StubPrefersSupportTechnicianWhenPresent(),
    )
    recs = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Contract Analyst",
        department="Financial Services",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid=None,
        new_hire_netid=None,
    )
    by_group = recs.set_index("GroupName")
    assert bool(by_group.loc["FS.Template", "InReferenceSheet"]) is True
    assert bool(by_group.loc["IT.Template", "InReferenceSheet"]) is False
    diag = recs.attrs.get("reference_diagnostics", {})
    assert diag.get("reference_match_path") == "fallback_title_same_department"


def test_reference_fallback_empty_when_no_department_overlap():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "u1",
                "DisplayName": "User One",
                "Title": "Contract Analyst",
                "Department": "Financial Services",
                "GroupsList": ["IT.Template"],
            },
        ]
    )
    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Support Technician",
                "Department": "Information Technology",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "IT.Template",
                "SourceFile": "full_time_employee_access.xlsx",
            },
        ]
    )
    engine = AccessRecommendationEngine(
        min_confidence=0.4,
        title_matcher=_StubPrefersSupportTechnicianWhenPresent(),
    )
    recs = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Contract Analyst",
        department="Financial Services",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid=None,
        new_hire_netid=None,
    )
    row = recs.set_index("GroupName").loc["IT.Template"]
    assert bool(row["InReferenceSheet"]) is False
    diag = recs.attrs.get("reference_diagnostics", {})
    assert diag.get("reference_match_path") == "no_reference_match"
    assert diag.get("fallback_empty_due_to_department_mismatch") is True


def test_recommend_for_hire_matches_reference_aliases_and_normalized_group_names():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "copyuser",
                "Title": "Academic Outreach & Sales Rep",
                "Department": "CE Academic Outreach & Sales",
                "GroupsList": ["m.Shared Drive"],
            },
            {
                "SamAccountName": "peer2",
                "Title": "Academic Outreach & Sales Rep",
                "Department": "CE Academic Outreach & Sales",
                "GroupsList": ["m.Shared Drive"],
            },
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Academic Outreach, Sales Rep",
                "Department": "Marketing & Customer Support",
                "Supervisor": None,
                "AccessCategory": "Storage",
                "AccessName": "Shared Drive",
                "AccessNameClean": "shared drive",
                "SourceFile": "full_time_employee_access.xlsx",
            }
        ]
    )

    engine = AccessRecommendationEngine(min_confidence=0.4)

    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Academic Outreach & Sales Rep",
        department="CE Academic Outreach & Sales",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid="copyuser",
        new_hire_netid=None,
    )

    row = recommendations.set_index("GroupName").loc["Shared Drive"]

    assert row.name == "Shared Drive"
    assert engine._normalize_group_name("m.Shared Drive") == engine._normalize_group_name("Shared Drive")
    assert bool(row["InReferenceSheet"]) is True
    assert row["ADConfidence"] == 1.0
    assert row["MLMode"] == "peer_aggregate"
    assert row["MLConfidence"] == 1.0
    assert bool(row["CopyFromUserHasIt"]) is True
    assert row["ReferenceCategories"] == "Storage"
    assert row["FinalScore"] >= 0.65
    assert row["FinalDecision"] in {"Strong Recommend", "Auto Assign"}


def test_recommend_for_hire_uses_peer_aggregate_ml_when_new_hire_missing():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "copyuser",
                "Title": "Academic Outreach & Sales Rep",
                "Department": "CE Academic Outreach & Sales",
                "GroupsList": ["VPN"],
                "IsSupervisor": False,
            },
            {
                "SamAccountName": "peer1",
                "Title": "Academic Outreach & Sales Rep",
                "Department": "CE Academic Outreach & Sales",
                "GroupsList": ["VPN", "Email", "SharedDrive"],
                "IsSupervisor": False,
            },
            {
                "SamAccountName": "peer2",
                "Title": "Academic Outreach & Sales Rep",
                "Department": "CE Academic Outreach & Sales",
                "GroupsList": ["VPN", "Email"],
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
        title="Academic Outreach & Sales Rep",
        department="CE Academic Outreach & Sales",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid="copyuser",
        new_hire_netid=None,
    )

    row = recommendations.set_index("GroupName").loc["Email"]

    assert row["MLMode"] == "peer_aggregate"
    assert row["MLAnchorNetID"] == ""
    assert row["MLSupportCount"] == 2
    assert row["MLComparedUsers"] == 3
    assert row["ADConfidence"] == 0.667
    assert row["FinalScore"] <= 0.2
    assert row["MLConfidence"] == pytest.approx(2 / 3, rel=1e-6)
    assert "peer-aggregate ML in 2/3 role peers" in row["Reason"]


def test_recommend_for_hire_uses_same_department_reference_overlap_for_ad_cohort():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "exact_title",
                "Title": "Computing Specialist",
                "Department": "CE IT Help Desk",
                "GroupsList": ["VPN", "Email"],
            },
            {
                "SamAccountName": "bad_title",
                "Title": "Comp Specialist",
                "Department": "CE IT Help Desk",
                "GroupsList": ["VPN", "Email"],
            },
            {
                "SamAccountName": "generic_title",
                "Title": "Student Worker 5",
                "Department": "CE IT Help Desk",
                "GroupsList": ["VPN", "Email"],
            },
            {
                "SamAccountName": "other_department",
                "Title": "Student Worker 5",
                "Department": "Finance",
                "GroupsList": ["VPN", "Email"],
            },
            {
                "SamAccountName": "same_department_other_job",
                "Title": "Receptionist",
                "Department": "CE IT Help Desk",
                "GroupsList": ["BadgeAccess"],
            },
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Student",
                "JobTitle": "Computing Specialist",
                "Department": "Information Technology",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "VPN",
                "AccessNameClean": "vpn",
                "SourceFile": "student_employee_access.xlsx",
            },
            {
                "EmployeeType": "Student",
                "JobTitle": "Computing Specialist",
                "Department": "Information Technology",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "Email",
                "AccessNameClean": "email",
                "SourceFile": "student_employee_access.xlsx",
            },
        ]
    )

    engine = AccessRecommendationEngine(min_confidence=0.4)

    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Computing Specialist",
        department="CE IT Help Desk",
        employee_type="Student",
        supervisor=None,
        copy_from_netid=None,
        new_hire_netid=None,
    )

    by_group = recommendations.set_index("GroupName")

    assert by_group.loc["VPN", "UserCountWithGroup"] == 3
    assert by_group.loc["VPN", "TotalUsersInRole"] == 3
    assert by_group.loc["VPN", "ADConfidence"] == 1.0
    assert by_group.loc["VPN", "MLComparedUsers"] == 3
    assert by_group.loc["VPN", "MLConfidence"] == 1.0
    assert "BadgeAccess" not in by_group.index


def test_recommend_for_hire_full_time_falls_back_to_copy_from_name_match():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "copyuser",
                "DisplayName": "Alex Doe",
                "Title": "Unrelated Title",
                "Department": "Unrelated Department",
                "GroupsList": ["VPN"],
            },
            {
                "SamAccountName": "peer2",
                "DisplayName": "Peer User",
                "Title": "Unrelated Title",
                "Department": "Unrelated Department",
                "GroupsList": ["VPN"],
            },
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Different Job",
                "Department": "Completely Different Department",
                "Supervisor": None,
                "ReferenceEmployeeName": "Alex Doe",
                "AccessCategory": "AD Rights",
                "AccessName": "VPN",
                "AccessNameClean": "vpn",
                "SourceFile": "full_time_employee_access.xlsx",
            }
        ]
    )

    engine = AccessRecommendationEngine(min_confidence=0.4)

    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Completely Different Title",
        department="Completely Different Department",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid="copyuser",
        new_hire_netid=None,
    )

    by_group = recommendations.set_index("GroupName")
    assert bool(by_group.loc["VPN", "InReferenceSheet"]) is True


def test_recommend_for_hire_full_time_uses_copy_from_ad_department_for_cohort():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "deg11",
                "DisplayName": "Deg Eleven",
                "Title": "Manager, Tech Support",
                "Department": "CE IT Help Desk",
                "GroupsList": ["VPN", "Email"],
                "IsSupervisor": True,
            },
            {
                "SamAccountName": "peer1",
                "DisplayName": "Peer One",
                "Title": "Technician",
                "Department": "CE IT Help Desk",
                "GroupsList": ["VPN", "Email"],
                "IsSupervisor": False,
            },
            {
                "SamAccountName": "peer2",
                "DisplayName": "Peer Two",
                "Title": "Analyst",
                "Department": "CE IT Help Desk",
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
                "ReferenceEmployeeName",
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
    assert by_group.loc["VPN", "TotalUsersInRole"] == 3
    assert by_group.loc["VPN", "ADConfidence"] == 1.0
    assert by_group.loc["Email", "TotalUsersInRole"] == 3


def test_scoring_prioritizes_reference_for_fsy_roles():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "copyuser",
                "DisplayName": "Copy User",
                "Title": "FSY Mentor",
                "Department": "FSY Programs",
                "GroupsList": ["VPN"],
            },
            {
                "SamAccountName": "peer1",
                "DisplayName": "Peer One",
                "Title": "FSY Mentor",
                "Department": "FSY Programs",
                "GroupsList": ["VPN"],
            },
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Full Time",
                "JobTitle": "FSY Mentor",
                "Department": "FSY Programs",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "VPN",
                "AccessNameClean": "vpn",
                "SourceFile": "full_time_employee_access.xlsx",
            }
        ]
    )

    engine = AccessRecommendationEngine(min_confidence=0.4)
    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="FSY Mentor",
        department="FSY Programs",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid="copyuser",
        new_hire_netid=None,
    )
    row = recommendations.set_index("GroupName").loc["VPN"]
    assert row["FinalScore"] >= 0.65


def test_scoring_prioritizes_reference_for_students():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "s1",
                "DisplayName": "Student One",
                "Title": "Student Worker",
                "Department": "IT",
                "GroupsList": ["Email"],
            },
            {
                "SamAccountName": "s2",
                "DisplayName": "Student Two",
                "Title": "Student Worker",
                "Department": "IT",
                "GroupsList": ["Email"],
            },
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Student",
                "JobTitle": "Student Worker",
                "Department": "IT",
                "Supervisor": None,
                "AccessCategory": "Messaging",
                "AccessName": "Email",
                "AccessNameClean": "email",
                "SourceFile": "student_employee_access.xlsx",
            }
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
    row = recommendations.set_index("GroupName").loc["Email"]
    assert row["FinalScore"] >= 0.60


def test_reference_matching_uses_embedding_fallback_for_title_variants():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "copyuser",
                "DisplayName": "Copy User",
                "Title": "Asst. Dir, FSY Programs",
                "Department": "CE FSY",
                "GroupsList": ["VPN"],
            }
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Full Time",
                "JobTitle": "FSY Assistant Director Programs",
                "Department": "CE FSY",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "VPN",
                "AccessNameClean": "vpn",
                "SourceFile": "full_time_employee_access.xlsx",
            }
        ]
    )

    engine = AccessRecommendationEngine(
        min_confidence=0.4,
        title_matcher=_StubTitleMatcher(),
    )
    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Asst. Dir, FSY Programs",
        department="CE FSY",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid="copyuser",
        new_hire_netid=None,
    )

    row = recommendations.set_index("GroupName").loc["VPN"]
    assert bool(row["InReferenceSheet"]) is True
    diag = recommendations.attrs.get("reference_diagnostics", {})
    assert diag.get("reference_match_path") == "fallback_title_same_department"


def test_reference_matching_is_separator_insensitive_for_group_names():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "copyuser",
                "DisplayName": "Copy User",
                "Title": "Student Worker",
                "Department": "IT",
                "GroupsList": ["IS.AllUsers", "DCE-DomainUsers", "EAG ce indstudy"],
            }
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Student",
                "JobTitle": "Student Worker",
                "Department": "IT",
                "Supervisor": None,
                "AccessCategory": "Core",
                "AccessName": "dce.is.allusers",
                "AccessNameClean": "dce.is.allusers",
                "SourceFile": "student_employee_access.xlsx",
            },
            {
                "EmployeeType": "Student",
                "JobTitle": "Student Worker",
                "Department": "IT",
                "Supervisor": None,
                "AccessCategory": "Core",
                "AccessName": "dce.domainusers",
                "AccessNameClean": "dce.domainusers",
                "SourceFile": "student_employee_access.xlsx",
            },
            {
                "EmployeeType": "Student",
                "JobTitle": "Student Worker",
                "Department": "IT",
                "Supervisor": None,
                "AccessCategory": "Core",
                "AccessName": "EAG-ce-indstudy",
                "AccessNameClean": "eag-ce-indstudy",
                "SourceFile": "student_employee_access.xlsx",
            },
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
        copy_from_netid="copyuser",
        new_hire_netid=None,
    )

    by_group = recommendations.set_index("GroupName")
    assert bool(by_group.loc["IS.AllUsers", "InReferenceSheet"]) is True
    assert bool(by_group.loc["DCE-DomainUsers", "InReferenceSheet"]) is True
    assert bool(by_group.loc["EAG ce indstudy", "InReferenceSheet"]) is True


def test_recommendations_align_with_reference_sheet_and_exact_same_rights_user():
    expected_groups = [
        "DCE-DomainUsers",
        "IS.AllUsers",
        "DCE.WriteTo.isdev.byu.edu_courses_is_site_t_tasks_web",
        "DCE.STUDENT",
        "DCE.ISRESPONSE",
        "DCE.IS.STUDENT",
        "EAG-ce-indstudy",
        "EAG-ce-byuo_highschool",
        "EAG-ce-cecsfax",
        "EAG-ce-dcesupport",
        "EAG-ce-efy",
        "EAG-ce-is_tech",
    ]

    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "aaron.moody",
                "DisplayName": "Aaron Moody",
                "Title": "Customer Service Rep I",
                "Department": "MCS",
                "GroupsList": expected_groups,
            },
            {
                "SamAccountName": "mikayla.penrod",
                "DisplayName": "Mikayla Penrod",
                "Title": "Customer Service Rep I",
                "Department": "MCS",
                "GroupsList": expected_groups,
            },
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Student",
                "JobTitle": "Customer Service Rep I",
                "Department": "MCS",
                "Supervisor": None,
                "AccessCategory": "Access Sheet",
                "AccessName": group_name,
                "AccessNameClean": group_name,
                "SourceFile": "student_employee_access.xlsx",
            }
            for group_name in expected_groups
        ]
    )

    engine = AccessRecommendationEngine(min_confidence=0.4)
    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Customer Service Rep I",
        department="MCS",
        employee_type="Student",
        supervisor=None,
        copy_from_netid="aaron.moody",
        new_hire_netid=None,
    )

    by_group = recommendations.set_index("GroupName")
    missing = [group_name for group_name in expected_groups if group_name not in by_group.index]
    assert missing == []

    for group_name in expected_groups:
        assert bool(by_group.loc[group_name, "InReferenceSheet"]) is True
        assert bool(by_group.loc[group_name, "CopyFromUserHasIt"]) is True
        assert by_group.loc[group_name, "FinalDecision"] in {
            "Auto Assign",
            "Strong Recommend",
            "Manual Review",
            "Suggest",
        }


def test_recommend_for_hire_reason_assignment_handles_non_scalar_apply_output(monkeypatch):
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "u1",
                "DisplayName": "User One",
                "Title": "Student Worker",
                "Department": "IT",
                "GroupsList": ["VPN"],
            },
            {
                "SamAccountName": "u2",
                "DisplayName": "User Two",
                "Title": "Student Worker",
                "Department": "IT",
                "GroupsList": ["VPN"],
            },
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Student",
                "JobTitle": "Student Worker",
                "Department": "IT",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "VPN",
                "AccessNameClean": "vpn",
                "SourceFile": "student_employee_access.xlsx",
            }
        ]
    )

    engine = AccessRecommendationEngine(min_confidence=0.4)

    def _series_reason(_row):
        return pd.Series({"primary": "primary reason", "secondary": "secondary reason"})

    monkeypatch.setattr(engine, "_reason", _series_reason)

    recs = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Student Worker",
        department="IT",
        employee_type="Student",
        supervisor=None,
    )

    assert recs.iloc[0]["Reason"] == "primary reason"


def test_reason_support_denominator_uses_actual_ml_compared_users():
    engine = AccessRecommendationEngine(min_confidence=0.4)
    row = {
        "InReferenceSheet": False,
        "AmbiguousReferenceTemplate": False,
        "ADConfidence": 0.0,
        "MLConfidence": 0.0,
        "MLMode": "peer_aggregate",
        "MLSupportCount": 0,
        "MLComparedUsers": 0,
        "CopyFromUserHasIt": True,
        "CopyFromNetID": "cng17",
        "FinalScore": 0.05,
        "CohortSize": 4,
        "GlobalGroupRate": 0.05,
    }
    reason = engine._reason(row)
    assert "support=0/0" in reason


def test_ad_support_counts_parse_semicolon_groupslist_for_copy_from_right():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "copyuser",
                "DisplayName": "Copy User",
                "Title": "Program Assistant",
                "Department": "IT",
                "GroupsList": ["DCE.IS.Speedback"],
            },
            {
                "SamAccountName": "peer1",
                "DisplayName": "Peer One",
                "Title": "Program Assistant",
                "Department": "IT",
                "GroupsList": "DCE.IS.Speedback;DCE.IS.Speedback-LimitedAccess",
            },
            {
                "SamAccountName": "peer2",
                "DisplayName": "Peer Two",
                "Title": "Program Assistant",
                "Department": "IT",
                "GroupsList": [],
            },
            {
                "SamAccountName": "peer3",
                "DisplayName": "Peer Three",
                "Title": "Program Assistant",
                "Department": "IT",
                "GroupsList": [],
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
        title="Program Assistant",
        department="IT",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid="copyuser",
        new_hire_netid=None,
    )
    row = recs[recs["GroupName"] == "DCE.IS.Speedback"].iloc[0]
    assert row["TotalUsersInRole"] == 4
    assert row["UserCountWithGroup"] == 2
    assert row["ADConfidence"] == pytest.approx(0.5, rel=1e-6)


def _ce_it_student_reference_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "u1",
                "Title": "Computing Specialist",
                "Department": "CE IT Help Desk",
                "GroupsList": ["VPN", "Email"],
            },
        ]
    )
    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Student",
                "JobTitle": "Computing Specialist",
                "Department": "Information Technology",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "VPN",
                "SourceFile": "student_employee_access.xlsx",
            },
            {
                "EmployeeType": "Student",
                "JobTitle": "Computing Specialist",
                "Department": "Information Technology",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "Email",
                "SourceFile": "student_employee_access.xlsx",
            },
        ]
    )
    return users_df, reference_df


@pytest.mark.parametrize(
    "title",
    [
        "Computing Specialist",
        "Computer Specialist",
        "Student Worker 5",
    ],
)
def test_ce_it_student_title_variants_share_reference_template(title):
    users_df, reference_df = _ce_it_student_reference_fixture()
    engine = AccessRecommendationEngine(min_confidence=0.4)
    recs = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title=title,
        department="CE IT Help Desk",
        employee_type="Student",
        supervisor=None,
        copy_from_netid=None,
        new_hire_netid=None,
        recommendation_debug=True,
    )
    role_inference = recs.attrs.get("role_inference", {})
    assert role_inference.get("canonical_role_id") == "role:ce_it_helpdesk_student_support"

    diag = recs.attrs.get("reference_diagnostics", {})
    assert diag.get("reference_match_path") != "no_reference_match"
    assert diag.get("reference_candidate_source") == "cluster_expanded"

    by_group = recs.set_index("GroupName")
    assert bool(by_group.loc["VPN", "InReferenceSheet"]) is True
    assert bool(by_group.loc["Email", "InReferenceSheet"]) is True
