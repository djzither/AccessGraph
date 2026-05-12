import pandas as pd

from DataLayer.subgroup_detection import analyze_permission_subgroup


def test_detects_hidden_subgroup_as_subrole_access():
    cohort = pd.DataFrame(
        [
            {
                "SamAccountName": "u1",
                "GroupsList": ["Baseline.Read", "App.Core", "Sub.Team", "Sub.Tool", "Target.Permission"],
            },
            {
                "SamAccountName": "u2",
                "GroupsList": "Baseline.Read;App.Core;Sub.Team;Sub.Tool;Target.Permission",
            },
            {
                "SamAccountName": "u3",
                "GroupsList": ["Baseline.Read", "App.Core"],
            },
            {
                "SamAccountName": "u4",
                "GroupsList": ["Baseline.Read", "App.Core"],
            },
            {
                "SamAccountName": "u5",
                "GroupsList": ["Baseline.Read", "App.Core"],
            },
        ]
    )

    result = analyze_permission_subgroup(
        comparison_cohort=cohort,
        permission="Target.Permission",
    )

    assert result["broad_cohort_size"] == 5
    assert result["users_with_permission"] == ["u1", "u2"]
    assert result["users_without_permission"] == ["u3", "u4", "u5"]
    assert "Sub.Team" in result["with_shared_permissions"]
    assert "Sub.Tool" in result["with_shared_permissions"]
    assert "Baseline.Read" in result["without_shared_permissions"]
    assert result["subgroup_assessment"] == "Subrole Access"
    indicators = [row["permission"] for row in result["strongest_subgroup_indicators"]]
    assert "Sub.Team" in indicators
    assert "Sub.Tool" in indicators


def test_handles_semicolon_and_list_groups_consistently():
    cohort = pd.DataFrame(
        [
            {
                "SamAccountName": "x1",
                "GroupsList": "Base.A;Cluster.K;Target.K",
            },
            {
                "SamAccountName": "x2",
                "GroupsList": ["Base.A", "Cluster.K", "Target.K"],
            },
            {
                "SamAccountName": "x3",
                "GroupsList": "Base.A",
            },
            {
                "SamAccountName": "x4",
                "GroupsList": ["Base.A"],
            },
            {
                "SamAccountName": "x5",
                "GroupsList": [],
            },
        ]
    )

    result = analyze_permission_subgroup(
        comparison_cohort=cohort,
        permission="Target.K",
    )

    assert result["broad_cohort_size"] == 5
    assert result["users_with_permission"] == ["x1", "x2"]
    assert result["subgroup_assessment"] == "Subrole Access"
    indicators = result["strongest_subgroup_indicators"]
    assert len(indicators) >= 1
    assert indicators[0]["permission"] == "Cluster.K"
