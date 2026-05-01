import pandas as pd
import pytest

from MLLayer.recommender import MLRecommender


def make_recommender_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SamAccountName": "target",
                "Department": "CE IT",
                "GroupsList": ["VPN"],
                "IsSupervisor": False,
            },
            {
                "SamAccountName": "u1",
                "Department": "CE IT",
                "GroupsList": ["VPN", "Email", "SharedDrive"],
                "IsSupervisor": False,
            },
            {
                "SamAccountName": "u2",
                "Department": "CE IT",
                "GroupsList": ["VPN", "Email"],
                "IsSupervisor": False,
            },
            {
                "SamAccountName": "manager1",
                "Department": "CE IT",
                "GroupsList": ["VPN", "Payroll"],
                "IsSupervisor": True,
            },
            {
                "SamAccountName": "other_dept",
                "Department": "Finance",
                "GroupsList": ["ERP"],
                "IsSupervisor": False,
            },
        ]
    )


def test_recommend_for_user_returns_missing_rights_from_similar_users():
    recommender = MLRecommender(make_recommender_df())

    results = recommender.recommend_for_user(
        sam_account_name="target",
        department="CE IT",
        top_n_users=2,
        min_support=2,
        include_supervisors=False,
    )

    assert "Email" in results["GroupName"].tolist()
    assert "VPN" not in results["GroupName"].tolist()

    row = results.set_index("GroupName").loc["Email"]
    assert row["MLSupportCount"] == 2
    assert row["MLComparedUsers"] == 2
    assert row["MLConfidence"] == 1.0


def test_recommend_for_user_respects_min_support():
    recommender = MLRecommender(make_recommender_df())

    results = recommender.recommend_for_user(
        sam_account_name="target",
        department="CE IT",
        top_n_users=2,
        min_support=2,
        include_supervisors=False,
    )

    assert "SharedDrive" not in results["GroupName"].tolist()


def test_recommend_for_user_excludes_supervisors_by_default():
    recommender = MLRecommender(make_recommender_df())

    results = recommender.recommend_for_user(
        sam_account_name="target",
        department="CE IT",
        top_n_users=3,
        min_support=1,
        include_supervisors=False,
    )

    assert "Payroll" not in results["GroupName"].tolist()


def test_recommend_for_user_can_include_supervisors():
    recommender = MLRecommender(make_recommender_df())

    results = recommender.recommend_for_user(
        sam_account_name="target",
        department="CE IT",
        top_n_users=3,
        min_support=1,
        include_supervisors=True,
    )

    assert "Payroll" in results["GroupName"].tolist()


def test_recommend_for_user_returns_empty_when_pool_too_small():
    df = pd.DataFrame(
        [
            {
                "SamAccountName": "target",
                "Department": "CE IT",
                "GroupsList": ["VPN"],
                "IsSupervisor": False,
            }
        ]
    )
    recommender = MLRecommender(df)

    results = recommender.recommend_for_user(
        sam_account_name="target",
        department="CE IT",
    )

    assert results.empty


def test_recommend_for_user_raises_for_missing_target():
    recommender = MLRecommender(make_recommender_df())

    with pytest.raises(ValueError, match="not found"):
        recommender.recommend_for_user(
            sam_account_name="missing",
            department="CE IT",
        )


def test_recommend_for_peer_cohort_uses_all_users_in_selected_cohort():
    recommender = MLRecommender(make_recommender_df())
    cohort = pd.DataFrame(
        [
            {
                "SamAccountName": "u1",
                "GroupsList": ["VPN", "Email"],
            },
            {
                "SamAccountName": "u2",
                "GroupsList": ["VPN", "Email"],
            },
            {
                "SamAccountName": "u3",
                "GroupsList": ["VPN"],
            },
        ]
    )

    results = recommender.recommend_for_peer_cohort(
        cohort_df=cohort,
        min_support=2,
    )

    row = results.set_index("GroupName").loc["Email"]

    assert row["MLSupportCount"] == 2
    assert row["MLComparedUsers"] == 3
    assert row["MLConfidence"] == pytest.approx(2 / 3, rel=1e-6)
