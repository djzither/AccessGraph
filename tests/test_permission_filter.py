import pandas as pd

from DeterministicLayer.permission_filter import PermissionFilter


def test_filter_recommendations_assigns_risk_scores_and_decisions():
    filterer = PermissionFilter()
    recommendations = pd.DataFrame(
        [
            {
                "GroupName": "VPN",
                "UserCountWithGroup": 5,
                "TotalUsersInRole": 5,
            },
            {
                "GroupName": "Email",
                "UserCountWithGroup": 4,
                "TotalUsersInRole": 5,
            },
            {
                "GroupName": "Shared Drive",
                "UserCountWithGroup": 2,
                "TotalUsersInRole": 5,
            },
            {
                "GroupName": "Domain Admins",
                "UserCountWithGroup": 4,
                "TotalUsersInRole": 5,
            },
            {
                "GroupName": "Legacy App",
                "UserCountWithGroup": 1,
                "TotalUsersInRole": 5,
            },
            {
                "GroupName": "disabled-old-group",
                "UserCountWithGroup": 5,
                "TotalUsersInRole": 5,
            },
            {
                "GroupName": "HCEB ALL Floors Basic Access",
                "UserCountWithGroup": 5,
                "TotalUsersInRole": 5,
                "ReferenceCategories": "HCEB Doors",
            },
        ]
    )

    filtered = filterer.filter_recommendations(recommendations)

    assert "disabled-old-group" not in filtered["GroupName"].tolist()
    assert "HCEB ALL Floors Basic Access" not in filtered["GroupName"].tolist()

    by_group = filtered.set_index("GroupName")

    assert by_group.loc["VPN", "RiskLevel"] == "Low"
    assert by_group.loc["VPN", "Decision"] == "Auto Assign"

    assert by_group.loc["Email", "Decision"] == "Strong Recommend"
    assert by_group.loc["Shared Drive", "Decision"] == "Low Confidence"

    assert by_group.loc["Domain Admins", "RiskLevel"] == "High"
    assert by_group.loc["Domain Admins", "Decision"] == "Manual Review"

    assert by_group.loc["Legacy App", "Decision"] == "This is probably an extra right"


def test_filter_recommendations_handles_empty_input():
    filterer = PermissionFilter()
    empty_df = pd.DataFrame(columns=["GroupName", "UserCountWithGroup", "TotalUsersInRole"])

    filtered = filterer.filter_recommendations(empty_df)

    assert filtered.empty
    assert "RiskLevel" in filtered.columns
    assert "Decision" in filtered.columns


def test_filter_recommendations_drops_hceb_hcen_physical_access_by_name_fallback():
    filterer = PermissionFilter()
    recommendations = pd.DataFrame(
        [
            {
                "GroupName": "HCEN Rm 348 Work Room",
                "UserCountWithGroup": 1,
                "TotalUsersInRole": 5,
            },
            {
                "GroupName": "DCE.BGSEV.RoomScheduler",
                "UserCountWithGroup": 5,
                "TotalUsersInRole": 5,
            },
        ]
    )

    filtered = filterer.filter_recommendations(recommendations)

    assert "HCEN Rm 348 Work Room" not in filtered["GroupName"].tolist()
    assert "DCE.BGSEV.RoomScheduler" in filtered["GroupName"].tolist()
