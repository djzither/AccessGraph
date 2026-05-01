from pathlib import Path
import pandas as pd

from DataLayer.cleaner import DataCleaner
from DataLayer.loader import DataLoader
from DataLayer.rights_sheets_loader import RightsSheetsLoader
from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


def main():
    project_root = Path(__file__).resolve().parents[1]

    raw_path = project_root / "data" / "raw"
    processed_path = project_root / "data" / "processed" / "clean_users.parquet"

    ad_file_name = "ce_ad_user_rights_all.xlsx"

    print("=== Access Recommendation Engine Test ===")

    cleaner = DataCleaner(processed_path=str(processed_path))

    if processed_path.exists():
        print(f"Loading cleaned users from: {processed_path}")
        users_df = cleaner.load_cleaned()
    else:
        print("Cleaned file not found. Loading raw AD export...")

        loader = DataLoader(base_path=str(raw_path))
        raw_users_df = loader.load_file(ad_file_name)

        users_df = cleaner.clean_and_save(raw_users_df)

        print(f"Saved cleaned users to: {processed_path}")

    print(f"Loaded users: {len(users_df)}")
    print("User columns:", list(users_df.columns))

    print("\nLoading reference access sheets...")
    rights_loader = RightsSheetsLoader(raw_path=str(raw_path))
    reference_df = rights_loader.load_reference_sheets()

    print(f"Loaded reference access rows: {len(reference_df)}")
    print("Reference columns:", list(reference_df.columns))

    engine = AccessRecommendationEngine(min_confidence=0.4)

    title = "Computing Specialist"
    department = "CE IT Help Desk"
    employee_type = "Student"
    supervisor = None
    copy_from_netid = "ag877"

    recommendations = engine.recommend_for_hire(
        users_df=users_df,
        reference_df=reference_df,
        title=title,
        department=department,
        employee_type=employee_type,
        supervisor=supervisor,
        copy_from_netid=copy_from_netid,
        new_hire_netid=None,
    )

    if recommendations.empty:
        print("\nNo recommendations found.")
        return

    columns_to_show = [
        "GroupName",
        "FinalDecision",
        "FinalScore",
        "RiskLevel",
        "InReferenceSheet",
        "ADConfidence",
        "MLConfidence",
        "CopyFromUserHasIt",
        "Reason",
    ]

    columns_to_show = [
        col for col in columns_to_show if col in recommendations.columns
    ]

    print("\nTop recommendations:")
    print(recommendations[columns_to_show].head(50).to_string(index=False))

    output_path = project_root / "data" / "processed" / "access_recommendations_test.csv"
    try:
        recommendations.to_csv(output_path, index=False)
    except PermissionError:
        output_path = (
            project_root / "data" / "processed" / "access_recommendations_test.latest.csv"
        )
        recommendations.to_csv(output_path, index=False)

    print(f"\nSaved full recommendations to: {output_path}")


if __name__ == "__main__":
    main()
