import pandas as pd
from DataLayer.cleaner import DataCleaner


def test_clean_groups():
    df = pd.DataFrame({
        "Groups": [
            "A; B; Cannot find an object with identity",
            None
        ]
    })

    cleaner = DataCleaner()
    cleaned = cleaner.clean_groups(df)

    assert cleaned["GroupsList"][0] == ["A", "B"]
    assert cleaned["GroupsList"][1] == []


def test_clean_groups_ignores_letter_dot_prefix_groups():
    df = pd.DataFrame({
        "Groups": [
            "Domain Users; a.afemp; m.Email Eligible; i.EMPLOYEE1; o.Some Org; CE.MarketingAndCustomerSupport.Admin"
        ]
    })

    cleaner = DataCleaner()
    cleaned = cleaner.clean_groups(df)

    assert cleaned["GroupsList"][0] == [
        "Domain Users",
        "CE.MarketingAndCustomerSupport.Admin",
    ]


def test_clean_groups_keeps_full_time_staff_marker_for_workforce_classification():
    df = pd.DataFrame({
        "Groups": ["a.FULL TIME STAFF; Domain Users"],
    })
    cleaned = DataCleaner().clean_groups(df)
    assert "a.FULL TIME STAFF" in cleaned["GroupsList"].iloc[0]
    assert cleaned["EmployeeType"].iloc[0] == "FULL_TIME"
