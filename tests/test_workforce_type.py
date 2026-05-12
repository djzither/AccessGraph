from DataLayer.workforce_type import (
    FULL_TIME,
    FULL_TIME_STAFF_AD_GROUP,
    STUDENT,
    canonical_from_reference_employee_type,
    canonical_from_ui_label,
    classify_from_normalized_groups,
    reference_match_value,
)


def test_classify_from_groups_uses_staff_marker():
    assert classify_from_normalized_groups([]) == STUDENT
    assert classify_from_normalized_groups(["VPN", FULL_TIME_STAFF_AD_GROUP]) == FULL_TIME


def test_ui_and_reference_canonical_mapping():
    assert canonical_from_ui_label("Full Time") == FULL_TIME
    assert canonical_from_ui_label("Student") == STUDENT
    assert reference_match_value(FULL_TIME) == "full time"
    assert reference_match_value(STUDENT) == "student"


def test_reference_sheet_labels():
    assert canonical_from_reference_employee_type("Full Time") == FULL_TIME
    assert canonical_from_reference_employee_type("Student") == STUDENT
