import pandas as pd

from DeterministicLayer.rules_recommender import RulesRecommender


def test_recommend_for_new_user_combines_matrix_and_filter_logic():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "alice",
                "Title": "Helpdesk",
                "Department": "CE",
                "GroupsList": ["VPN", "Email", "Payroll Access"],
            },
            {
                "SamAccountName": "bob",
                "Title": "Helpdesk",
                "Department": "CE",
                "GroupsList": ["VPN", "Email", "Payroll Access"],
            },
            {
                "SamAccountName": "carol",
                "Title": "Helpdesk",
                "Department": "CE",
                "GroupsList": ["VPN"],
            },
            {
                "SamAccountName": "dave",
                "Title": "Analyst",
                "Department": "Finance",
                "GroupsList": ["ERP"],
            },
        ]
    )

    recommender = RulesRecommender(min_confidence=0.6)

    recommendations = recommender.recommend_for_new_user(
        users_df=users_df,
        title="Helpdesk",
        department="CE",
    )

    by_group = recommendations.set_index("GroupName")

    assert recommendations["GroupName"].tolist() == ["VPN", "Email", "Payroll Access"]
    assert (recommendations["Source"] == "Title + Department Match").all()

    assert by_group.loc["VPN", "Confidence"] == 1.0
    assert by_group.loc["VPN", "Decision"] == "Auto Assign"

    assert by_group.loc["Email", "Confidence"] == 0.667
    assert by_group.loc["Email", "Decision"] == "Suggest"

    assert by_group.loc["Payroll Access", "RiskLevel"] == "High"
    assert by_group.loc["Payroll Access", "Decision"] == "Manual Review"


def test_recommend_for_new_user_returns_empty_result_when_role_has_no_matches():
    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": "alice",
                "Title": "Helpdesk",
                "Department": "CE",
                "GroupsList": ["VPN"],
            }
        ]
    )

    recommender = RulesRecommender(min_confidence=0.6)

    recommendations = recommender.recommend_for_new_user(
        users_df=users_df,
        title="Analyst",
        department="Finance",
    )

    assert recommendations.empty
    assert recommendations.columns.tolist() == [
        "GroupName",
        "Source",
        "Confidence",
        "RiskLevel",
        "Decision",
        "UserCountWithGroup",
        "TotalUsersInRole",
    ]
