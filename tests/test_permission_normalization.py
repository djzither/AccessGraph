import numpy as np
import pandas as pd

from DataLayer.permission_normalization import (
    PermissionNormalizationStats,
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


def test_normalize_groups_input_tracks_stats():
    st = PermissionNormalizationStats()
    normalize_groups_input("a;; ;b", stats=st)
    assert st.raw_segments == 3
    assert st.dropped_blank_or_invalid >= 1
    assert st.output_tokens == 2


def test_normalize_groups_input_pandas_na_scalar():
    assert normalize_groups_input(pd.NA) == []
