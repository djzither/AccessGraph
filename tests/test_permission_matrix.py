import pandas as pd

from DeterministicLayer.permission_matrix import PermissionMatrixBuilder


def make_users_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SamAccountName": "alice",
                "Title": "Helpdesk",
                "Department": "CE",
                "GroupsList": ["VPN", "Email"],
            },
            {
                "SamAccountName": "bob",
                "Title": "Helpdesk",
                "Department": "CE",
                "GroupsList": ["VPN", "Email"],
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


def test_build_by_title_department_calculates_counts_and_confidence():
    builder = PermissionMatrixBuilder()

    matrix = builder.build_by_title_department(make_users_df())

    helpdesk_rows = matrix[
        (matrix["Title"] == "Helpdesk") & (matrix["Department"] == "CE")
    ]

    vpn_row = helpdesk_rows[helpdesk_rows["GroupName"] == "VPN"].iloc[0]
    email_row = helpdesk_rows[helpdesk_rows["GroupName"] == "Email"].iloc[0]

    assert vpn_row["UserCountWithGroup"] == 3
    assert vpn_row["TotalUsersInRole"] == 3
    assert vpn_row["Confidence"] == 1.0

    assert email_row["UserCountWithGroup"] == 2
    assert email_row["TotalUsersInRole"] == 3
    assert email_row["Confidence"] == 0.667


def test_recommend_for_role_respects_min_confidence():
    builder = PermissionMatrixBuilder(min_confidence=0.7)
    matrix = builder.build_by_title_department(make_users_df())

    recommendations = builder.recommend_for_role(
        matrix=matrix,
        title="Helpdesk",
        department="CE",
    )

    assert recommendations["GroupName"].tolist() == ["VPN"]


def test_build_by_title_department_returns_empty_dataframe_for_empty_input():
    builder = PermissionMatrixBuilder()
    empty_df = pd.DataFrame(columns=["Title", "Department", "GroupsList"])

    matrix = builder.build_by_title_department(empty_df)

    assert matrix.empty
