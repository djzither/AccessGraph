from pathlib import Path

from DataLayer.loader import DataLoader
from DataLayer.cleaner import DataCleaner


BASE_PATH = r"C:\Users\djzit\Downloads"
FILE_NAME = "ce_ad_user_rights_all.xlsx"


def main():
    print("=== AccessGraph Pipeline Start ===\n")

    # initialize
    loader = DataLoader(BASE_PATH)
    cleaner = DataCleaner("../data/processed/clean_users.parquet")

    # debug path
    full_path = Path(BASE_PATH) / FILE_NAME
    print(f"Looking for file:\n{full_path}\n")

    # load data
    df = loader.load_file(FILE_NAME, sheet_name=0)

    print("Loaded successfully.")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    # validate expected column
    if "Groups" not in df.columns:
        raise ValueError(
            "Missing 'Groups' column.\n"
            "Check your Excel sheet or correct sheet_name."
        )

    # clean data
    df = cleaner.clean_groups(df)

    print("\nCleaning complete.")
    print("Average clean group count:", round(df["CleanGroupCount"].mean(), 2))
    print("Max clean group count:", df["CleanGroupCount"].max())
    print("Min clean group count:", df["CleanGroupCount"].min())

    # preview
    print("\nSample cleaned data:\n")

    preview_cols = [
        "SamAccountName",
        "DisplayName",
        "Title",
        "Department",
        "GroupCount",
        "CleanGroupCount",
        "GroupsList",
    ]

    preview_cols = [c for c in preview_cols if c in df.columns]

    print(df[preview_cols].head())

    # save cleaned data
    # save cleaned data
    cleaner.save_cleaned(df)

    print("\nSaved cleaned dataset to:")
    print(cleaner.processed_path)

    # verify saved data
    print("\n=== Verify saved data ===")
    loaded_df = cleaner.load_cleaned()

    print("\nLoaded back from parquet:")
    print(loaded_df[preview_cols].head())

    print("\nColumns:")
    print(loaded_df.columns.tolist())

    print("\nExample GroupsList for first user:")
    print(loaded_df["GroupsList"].iloc[0])

    # optional exploded view
    exploded_df = loaded_df.explode("GroupsList")

    print("\nExploded view:")
    print(exploded_df[["SamAccountName", "DisplayName", "GroupsList"]].head(20))

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()