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