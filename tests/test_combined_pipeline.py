import pandas as pd
import pytest

from scripts.run_combined_pipeline import (
    add_user_evidence_columns,
    assign_final_decision,
    classify_review_reason,
    combine_recommendations,
)


def test_combine_recommendations_merges_rules_and_ml_scores():
    deterministic = pd.DataFrame(
        [
            {
                "GroupName": "VPN",
                "Confidence": 1.0,
                "RiskLevel": "Low",
                "Decision": "Auto Assign",
                "UserCountWithGroup": 3,
                "TotalUsersInRole": 3,
            },
            {
                "GroupName": "Email",
                "Confidence": 0.667,
                "RiskLevel": "Low",
                "Decision": "Suggest",
                "UserCountWithGroup": 2,
                "TotalUsersInRole": 3,
            },
        ]
    )
    ml = pd.DataFrame(
        [
            {
                "GroupName": "Email",
                "MLSupportCount": 2,
                "MLComparedUsers": 3,
                "MLConfidence": 0.667,
                "NearestUsers": "alice, bob",
            },
            {
                "GroupName": "Printer",
                "MLSupportCount": 2,
                "MLComparedUsers": 3,
                "MLConfidence": 0.667,
                "NearestUsers": "alice, bob",
            },
        ]
    )

    combined = combine_recommendations(deterministic, ml).set_index("GroupName")

    assert combined.loc["VPN", "FinalSource"] == "Rules only"
    assert combined.loc["VPN", "FinalScore"] == 1.0

    assert combined.loc["Email", "FinalSource"] == "Rules + ML"
    assert combined.loc["Email", "FinalScore"] == pytest.approx(0.667, rel=1e-3)
    assert combined.loc["Email", "ScoreBreakdown"] == "Rules: 0.67 | ML: 0.67"

    assert combined.loc["Printer", "FinalSource"] == "ML only"
    assert combined.loc["Printer", "FinalScore"] == 0.667


def test_combine_recommendations_handles_single_source_inputs():
    ml_only = pd.DataFrame(
        [
            {
                "GroupName": "Printer",
                "MLSupportCount": 2,
                "MLComparedUsers": 3,
                "MLConfidence": 0.667,
                "NearestUsers": "alice, bob",
            }
        ]
    )
    rules_only = pd.DataFrame(
        [
            {
                "GroupName": "VPN",
                "Confidence": 1.0,
                "RiskLevel": "Low",
                "Decision": "Auto Assign",
                "UserCountWithGroup": 3,
                "TotalUsersInRole": 3,
            }
        ]
    )

    ml_result = combine_recommendations(pd.DataFrame(), ml_only)
    rules_result = combine_recommendations(rules_only, pd.DataFrame())

    assert ml_result.iloc[0]["FinalSource"] == "ML only"
    assert ml_result.iloc[0]["FinalScore"] == 0.667
    assert ml_result.iloc[0]["Confidence"] == 0

    assert rules_result.iloc[0]["FinalSource"] == "Rules only"
    assert rules_result.iloc[0]["FinalScore"] == 1.0
    assert rules_result.iloc[0]["MLConfidence"] == 0


def test_add_user_evidence_columns_marks_holders_and_existing_access():
    users = pd.DataFrame(
        [
            {"SamAccountName": "alice", "GroupsList": ["VPN", "Email"]},
            {"SamAccountName": "bob", "GroupsList": ["Email"]},
            {"SamAccountName": "carol", "GroupsList": ["VPN"]},
        ]
    )
    final_results = pd.DataFrame(
        [{"GroupName": "Email", "NearestUsers": "alice, bob"}]
    )

    enriched = add_user_evidence_columns(
        final_results=final_results,
        users=users,
        target_user="carol",
    )

    assert bool(enriched.loc[0, "Has_alice"]) is True
    assert bool(enriched.loc[0, "Has_bob"]) is True
    assert bool(enriched.loc[0, "TargetAlreadyHas"]) is False
    assert enriched.loc[0, "UsersWithThisRight"] == "alice, bob"


@pytest.mark.parametrize(
    ("group_name", "expected"),
    [
        ("FSY Region Access", "Review: region/location-specific"),
        ("Storage Room Access", "Review: physical/location access"),
        ("Software Suite", "Likely software access"),
        ("Domain Administrators", "Review: admin-level access"),
        ("Email", "Normal review"),
    ],
)
def test_classify_review_reason(group_name, expected):
    assert classify_review_reason(group_name) == expected


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"GroupName": "Domain Admins", "FinalScore": 0.95, "MLConfidence": 0.9}, "REVIEW (admin access)"),
        ({"GroupName": "VPN", "FinalScore": 0.85, "MLConfidence": 0.6}, "AUTO APPROVE"),
        ({"GroupName": "VPN", "FinalScore": 0.55, "MLConfidence": 0.1}, "REVIEW"),
        ({"GroupName": "VPN", "FinalScore": 0.2, "MLConfidence": 0.1}, "LOW CONFIDENCE"),
    ],
)
def test_assign_final_decision(row, expected):
    assert assign_final_decision(row) == expected
