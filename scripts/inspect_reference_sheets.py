from pathlib import Path

from DataLayer.rights_sheets_loader import RightsSheetsLoader


def main():
    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / "data" / "raw"

    loader = RightsSheetsLoader(raw_path=raw_path)

    reference_df = loader.load_reference_sheets()

    print("Reference access rows loaded:")
    print(len(reference_df))

    print("\nColumns:")
    print(reference_df.columns.tolist())

    print("\nSample:")
    print(reference_df.head(30).to_string(index=False))

    print("\nAccess categories:")
    print(reference_df["AccessCategory"].value_counts())

    print("\nEmployee types:")
    print(reference_df["EmployeeType"].value_counts())


if __name__ == "__main__":
    main()