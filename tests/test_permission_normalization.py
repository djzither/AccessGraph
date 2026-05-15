import numpy as np
import pandas as pd

from DataLayer.permission_normalization import (
    canonical_permission_id,
    canonicalize_permission,
    normalize_groups_input,
    normalize_single_permission,
)


def test_normalize_single_permission_strips_and_drops_nan_like():
    assert normalize_single_permission("  foo  ") == "foo"
    assert normalize_single_permission("  ") is None
    assert normalize_single_permission(None) is None
    assert normalize_single_permission(float("nan")) is None
    assert normalize_single_permission("NaN") is None


def test_normalize_groups_input_semicolon_and_duplicate_delimiters():
    assert normalize_groups_input("a;;b; ;c") == ["a", "b", "c"]
    assert normalize_groups_input(["  x ", "", "y"]) == ["x", "y"]


def test_normalize_groups_input_numpy_array():
    arr = np.array(["a", " b "], dtype=object)
    assert normalize_groups_input(arr) == ["a", "b"]


def test_normalize_groups_input_pandas_na_scalar():
    assert normalize_groups_input(pd.NA) == []


def test_canonical_permission_id_collapses_cmp_allusers_variants():
    assert canonical_permission_id("CMP.AllUsers") == canonical_permission_id("DCE.CMP.Allusers")


def test_canonical_permission_id_collapses_domain_admins_variants():
    assert canonical_permission_id("CMP.Domain Admins") == canonical_permission_id(
        "DCE.CMP.DomainAdmins"
    )


def test_canonicalize_permission_preserves_raw_and_source():
    result = canonicalize_permission("DCE.CMP.Allusers", source="ad")
    assert result.raw_permission_name == "DCE.CMP.Allusers"
    assert result.canonical_permission_id == canonical_permission_id("CMP.AllUsers")
    assert result.source == "ad"
