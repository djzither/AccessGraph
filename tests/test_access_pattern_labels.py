import pandas as pd

from DataLayer.subgroup_detection import analyze_permission_subgroup, analyze_recommendation_subgroups
from DeterministicLayer.access_pattern_labels import (
    apply_access_pattern_columns,
    label_access_pattern,
    PATTERN_BASELINE,
    PATTERN_COMMON,
    PATTERN_HIGH_RISK,
    PATTERN_POSSIBLE_EXTRA,
    PATTERN_RARE,
    PATTERN_SUBROLE,
    PATTERN_UNIQUE,
    PATTERN_UNKNOWN,
)


def _row(**kwargs) -> pd.Series:
    base = {
        "GroupName": "App.VPN",
        "RiskLevel": "Low",
        "TotalUsersInRole": 10,
        "UserCountWithGroup": 5,
        "InReferenceSheet": False,
        "AmbiguousReferenceTemplate": False,
        "CopyFromUserHasIt": False,
        "MLConfidence": 0.0,
        "GlobalGroupRate": 0.1,
        "ADConfidence": 0.5,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_label_baseline_access():
    p, reason, q = label_access_pattern(_row(UserCountWithGroup=9, TotalUsersInRole=10, ADConfidence=0.9), has_subrole_evidence=False)
    assert p == PATTERN_BASELINE
    assert "90%" in reason or "9/10" in reason
    assert "standard" in q.lower() or "baseline" in q.lower() or "every hire" in q.lower()


def test_label_common_access():
    p, _, _ = label_access_pattern(_row(UserCountWithGroup=6, TotalUsersInRole=10, ADConfidence=0.6), has_subrole_evidence=False)
    assert p == PATTERN_COMMON


def test_label_subrole_access_with_subgroup_evidence():
    p, reason, q = label_access_pattern(
        _row(GroupName="Target.App", UserCountWithGroup=2, TotalUsersInRole=5, ADConfidence=0.4),
        has_subrole_evidence=True,
    )
    assert p == PATTERN_SUBROLE
    assert "subgroup" in reason.lower()
    assert "function" in q.lower() or "team" in q.lower()


def test_label_possible_extra_without_subgroup():
    p, _, _ = label_access_pattern(
        _row(UserCountWithGroup=2, TotalUsersInRole=5, ADConfidence=0.4),
        has_subrole_evidence=False,
    )
    assert p == PATTERN_POSSIBLE_EXTRA


def test_label_unique_single_holder():
    p, _, _ = label_access_pattern(
        _row(UserCountWithGroup=1, TotalUsersInRole=10, ADConfidence=0.1),
        has_subrole_evidence=False,
    )
    assert p == PATTERN_UNIQUE


def test_label_rare_very_low_support_multiple_holders():
    p, _, _ = label_access_pattern(
        _row(UserCountWithGroup=2, TotalUsersInRole=50, ADConfidence=0.04),
        has_subrole_evidence=False,
    )
    assert p == PATTERN_RARE


def test_label_high_risk_overrides_support():
    p, reason, q = label_access_pattern(
        _row(
            RiskLevel="High",
            UserCountWithGroup=10,
            TotalUsersInRole=10,
            ADConfidence=1.0,
        ),
        has_subrole_evidence=False,
    )
    assert p == PATTERN_HIGH_RISK
    assert "risk" in reason.lower() or "review" in reason.lower()
    assert "approv" in q.lower() or "governance" in q.lower()


def test_label_unknown_ambiguous_template_weak_support():
    p, _, _ = label_access_pattern(
        _row(
            UserCountWithGroup=2,
            TotalUsersInRole=10,
            InReferenceSheet=True,
            AmbiguousReferenceTemplate=True,
            ADConfidence=0.2,
        ),
        has_subrole_evidence=False,
    )
    assert p == PATTERN_UNKNOWN


def test_apply_access_pattern_columns_with_subgroup_dataframe():
    cohort = pd.DataFrame(
        [
            {"SamAccountName": "u1", "GroupsList": ["Baseline.Read", "Sub.Team", "Sub.Tool", "Target.App"]},
            {"SamAccountName": "u2", "GroupsList": ["Baseline.Read", "Sub.Team", "Sub.Tool", "Target.App"]},
            {"SamAccountName": "u3", "GroupsList": ["Baseline.Read"]},
            {"SamAccountName": "u4", "GroupsList": ["Baseline.Read"]},
            {"SamAccountName": "u5", "GroupsList": ["Baseline.Read"]},
        ]
    )
    merged = pd.DataFrame(
        [
            {
                "GroupName": "Target.App",
                "RiskLevel": "Low",
                "TotalUsersInRole": 5,
                "UserCountWithGroup": 2,
                "InReferenceSheet": False,
                "AmbiguousReferenceTemplate": False,
                "CopyFromUserHasIt": False,
                "MLConfidence": 0.0,
                "GlobalGroupRate": 0.02,
                "ADConfidence": 0.4,
            }
        ]
    )
    sub_single = analyze_permission_subgroup(cohort, "Target.App")
    assert sub_single["subgroup_assessment"] == "Subrole Access"

    sub_df = analyze_recommendation_subgroups(cohort, merged)
    out = apply_access_pattern_columns(merged, sub_df)
    assert out.iloc[0]["AccessPattern"] == PATTERN_SUBROLE
    assert "ReviewQuestion" in out.columns


def test_recommend_for_hire_includes_access_pattern_columns():
    from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine

    users_df = pd.DataFrame(
        [
            {
                "SamAccountName": f"u{i}",
                "Title": "Widget Analyst",
                "Department": "Widgets",
                "GroupsList": ["VPN"],
            }
            for i in range(10)
        ]
    )

    reference_df = pd.DataFrame(
        [
            {
                "EmployeeType": "Full Time",
                "JobTitle": "Widget Analyst",
                "Department": "Widgets",
                "Supervisor": None,
                "AccessCategory": "AD Rights",
                "AccessName": "VPN",
                "AccessNameClean": "vpn",
                "SourceFile": "full_time_employee_access.xlsx",
            }
        ]
    )
    engine = AccessRecommendationEngine(min_confidence=0.4)
    recs = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title="Widget Analyst",
        department="Widgets",
        employee_type="Full Time",
        supervisor=None,
        copy_from_netid=None,
        new_hire_netid=None,
    )
    assert "AccessPattern" in recs.columns
    assert "AmbiguityReason" in recs.columns
    assert "ReviewQuestion" in recs.columns
    vpn = recs.set_index("GroupName").loc["VPN"]
    assert vpn["AccessPattern"] == PATTERN_BASELINE
