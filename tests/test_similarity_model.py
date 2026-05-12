import pandas as pd
import pytest

from MLLayer.similarity_model import SimilarityModel


def make_users_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SamAccountName": "alice", "GroupsList": ["VPN", "Email"]},
            {"SamAccountName": "bob", "GroupsList": ["VPN"]},
            {"SamAccountName": "carol", "GroupsList": ["Email", "Storage"]},
            {"SamAccountName": "dave", "GroupsList": ["", None, "VPN"]},
        ]
    )


def test_fit_builds_expected_matrix():
    model = SimilarityModel().fit(make_users_df())

    assert set(model.matrix.index) == {"alice", "bob", "carol", "dave"}
    assert set(model.matrix.columns) == {"VPN", "Email", "Storage"}

    assert model.matrix.loc["alice", "VPN"] == 1
    assert model.matrix.loc["alice", "Email"] == 1
    assert model.matrix.loc["alice", "Storage"] == 0


def test_fit_ignores_empty_group_values():
    model = SimilarityModel().fit(make_users_df())

    assert "" not in model.matrix.columns


def test_similar_users_excludes_target_and_orders_by_similarity():
    model = SimilarityModel().fit(make_users_df())

    results = model.similar_users("alice", top_n=2)

    assert "alice" not in results["SamAccountName"].tolist()
    assert results.iloc[0]["SamAccountName"] == "bob"


def test_similar_users_raises_for_missing_user():
    model = SimilarityModel().fit(make_users_df())

    with pytest.raises(ValueError, match="not found"):
        model.similar_users("missing_user")
