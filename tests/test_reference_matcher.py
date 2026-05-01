import pandas as pd

from DeterministicLayer.reference_matcher import ReferenceMatcher


def make_reference_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Helpdesk",
                "Department": "CE",
                "Supervisor": "Alice",
                "AccessCategory": "Software",
                "AccessName": "VPN Access",
            },
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Helpdesk",
                "Department": "CE",
                "Supervisor": "Alice",
                "AccessCategory": "Storage",
                "AccessName": "Shared Drive",
            },
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Helpdesk",
                "Department": "CE",
                "Supervisor": "Bob",
                "AccessCategory": "Hardware",
                "AccessName": "Printer Access",
            },
            {
                "EmployeeType": "Student",
                "JobTitle": "Helpdesk",
                "Department": "CE",
                "Supervisor": "Bob",
                "AccessCategory": "Student Systems",
                "AccessName": "Student Portal",
            },
        ]
    )


def make_recommendations_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GroupName": "VPN Access",
                "Score": 0.9,
                "RiskLevel": "Low",
                "UserCountWithGroup": 9,
                "TotalUsersInRole": 10,
            },
            {
                "GroupName": "Printer Access",
                "Score": 0.5,
                "RiskLevel": "Low",
                "UserCountWithGroup": 5,
                "TotalUsersInRole": 10,
            },
            {
                "GroupName": "Shared Drive",
                "Score": 0.5,
                "RiskLevel": "Low",
                "UserCountWithGroup": 5,
                "TotalUsersInRole": 10,
            },
            {
                "GroupName": "Payroll Admin",
                "Score": 0.95,
                "RiskLevel": "High",
                "UserCountWithGroup": 19,
                "TotalUsersInRole": 20,
            },
            {
                "GroupName": "Student Portal",
                "Score": 0.85,
                "RiskLevel": "Low",
                "UserCountWithGroup": 17,
                "TotalUsersInRole": 20,
            },
        ]
    )


def test_match_recommendations_applies_employee_type_and_supervisor_filters():
    matcher = ReferenceMatcher(make_reference_df())

    matched = matcher.match_recommendations(
        recommendations=make_recommendations_df(),
        title="Helpdesk",
        department="CE",
        employee_type="Full Time",
        supervisor="Alice",
    )

    by_group = matched.set_index("GroupName")

    assert bool(by_group.loc["VPN Access", "ReferenceSheetMatch"]) is True
    assert by_group.loc["VPN Access", "ReferenceCategories"] == "Software"
    assert by_group.loc["VPN Access", "FinalDecision"] == "Strong Recommend"

    assert bool(by_group.loc["Printer Access", "ReferenceSheetMatch"]) is False
    assert by_group.loc["Printer Access", "FinalDecision"] == "Low Confidence"

    assert bool(by_group.loc["Shared Drive", "ReferenceSheetMatch"]) is True
    assert by_group.loc["Shared Drive", "ReferenceCategories"] == "Storage"
    assert by_group.loc["Shared Drive", "FinalDecision"] == "Manual Review"

    assert by_group.loc["Payroll Admin", "FinalDecision"] == "Manual Review"


def test_match_recommendations_uses_employee_type_match_for_students():
    matcher = ReferenceMatcher(make_reference_df())

    matched = matcher.match_recommendations(
        recommendations=make_recommendations_df(),
        title="Helpdesk",
        department="CE",
        employee_type="Student",
        supervisor="Someone Else",
    )

    by_group = matched.set_index("GroupName")

    assert bool(by_group.loc["Student Portal", "ReferenceSheetMatch"]) is True
    assert by_group.loc["Student Portal", "ReferenceCategories"] == "Student Systems"
    assert by_group.loc["Student Portal", "FinalDecision"] == "Strong Recommend"


def test_match_recommendations_generates_explanatory_reasons():
    matcher = ReferenceMatcher(make_reference_df())

    matched = matcher.match_recommendations(
        recommendations=make_recommendations_df(),
        title="Helpdesk",
        department="CE",
        employee_type="Full Time",
        supervisor="Alice",
    )

    by_group = matched.set_index("GroupName")

    assert (
        by_group.loc["VPN Access", "Reason"]
        == "Found in 9/10 similar users and listed in reference sheet."
    )
    assert (
        by_group.loc["Printer Access", "Reason"]
        == "Only found in 5/10 similar users and not listed in reference sheet."
    )
