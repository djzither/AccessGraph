import pandas as pd
import pytest

from DataLayer.permission_cooccurrence import (
    build_cooccurrence_state,
    cooccurrence_from_state,
    cooccurrence_with_target,
    iter_user_group_sets,
)


def test_cooccurrence_metrics_known_pattern():
    """Four users: overlap A∩B = 2, |A|=3, |B|=3 → Jaccard 0.5, P(B|A)=2/3."""
    df = pd.DataFrame(
        [
            {"SamAccountName": "u1", "GroupsList": ["Perm.A", "Perm.B", "Perm.C"]},
            {"SamAccountName": "u2", "GroupsList": ["Perm.A", "Perm.B"]},
            {"SamAccountName": "u3", "GroupsList": ["Perm.A"]},
            {"SamAccountName": "u4", "GroupsList": "Perm.B;Perm.D"},
        ]
    )
    out = cooccurrence_with_target(df, "Perm.A", top_n=10)
    row_b = out[out["co_permission"] == "Perm.B"].iloc[0]
    assert int(row_b["users_with_target"]) == 3
    assert int(row_b["users_with_b"]) == 3
    assert int(row_b["users_with_both"]) == 2
    assert row_b["p_b_given_a"] == pytest.approx(2 / 3, rel=1e-3)
    assert row_b["p_a_given_b"] == pytest.approx(2 / 3, rel=1e-3)
    assert row_b["jaccard"] == pytest.approx(0.5, rel=1e-3)
    assert row_b["overlap_pct"] == pytest.approx(200 / 3, rel=1e-2)
    ex = row_b["example_users_overlap"]
    assert "u1" in ex and "u2" in ex


def test_groupslist_semicolon_and_list_equivalent():
    df = pd.DataFrame(
        [
            {"SamAccountName": "a", "GroupsList": "X;Y"},
            {"SamAccountName": "b", "GroupsList": ["X", "Y"]},
        ]
    )
    out = cooccurrence_with_target(df, "X", top_n=5)
    row_y = out[out["co_permission"] == "Y"].iloc[0]
    assert int(row_y["users_with_both"]) == 2


def test_excludes_crm_and_deprecated_style_groups():
    df = pd.DataFrame(
        [
            {
                "SamAccountName": "c1",
                "GroupsList": ["Keep.One", "salesforce EU", "Keep.Two"],
            },
            {
                "SamAccountName": "c2",
                "GroupsList": ["Keep.One", "cannot find an object xyz", "Keep.Two"],
            },
        ]
    )
    out = cooccurrence_with_target(df, "Keep.One", top_n=20)
    co = set(out["co_permission"].tolist())
    assert "Keep.Two" in co
    assert not any("salesforce" in x.lower() for x in co)
    assert not any("cannot find" in x.lower() for x in co)


def test_cooccurrence_from_state_matches_cooccurrence_with_target():
    df = pd.DataFrame(
        [
            {"SamAccountName": "u1", "GroupsList": ["Perm.A", "Perm.B"]},
            {"SamAccountName": "u2", "GroupsList": ["Perm.A"]},
        ]
    )
    state = build_cooccurrence_state(df)
    a = cooccurrence_from_state(state, "Perm.A", top_n=10)
    b = cooccurrence_with_target(df, "Perm.A", top_n=10)
    pd.testing.assert_frame_equal(a, b)


def test_iter_user_group_sets_skips_empty_id():
    df = pd.DataFrame(
        [
            {"SamAccountName": "", "GroupsList": ["A"]},
            {"SamAccountName": "ok", "GroupsList": ["A"]},
        ]
    )
    pairs = list(iter_user_group_sets(df))
    assert len(pairs) == 1
    assert pairs[0][0] == "ok"
