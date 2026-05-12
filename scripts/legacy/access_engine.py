from pathlib import Path
import pandas as pd

from DataLayer.cleaner import DataCleaner
from DataLayer.rights_sheets_loader import RightsSheetsLoader
from DeterministicLayer.title_embed_matcher import TitleEmbedMatcher
from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


def main():
    project_root = Path(__file__).resolve().parents[1]

    raw_path = project_root / "data" / "raw"
    processed_path = project_root / "data" / "processed" / "clean_users.parquet"

    print("=== Access Recommendation Engine Test ===")
    print("Canonical data path: data/processed/clean_users.parquet")

    cleaner = DataCleaner(processed_path=str(processed_path))

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Missing {processed_path}. Run `python -m DataLayer.build_clean_users` first."
        )
    print(f"Loading cleaned users from: {processed_path}")
    users_df = cleaner.load_cleaned()

    print(f"Loaded users: {len(users_df)}")
    print("User columns:", list(users_df.columns))

    print("\nLoading reference access sheets...")
    rights_loader = RightsSheetsLoader(raw_path=str(raw_path))
    reference_df = rights_loader.load_reference_sheets()

    print(f"Loaded reference access rows: {len(reference_df)}")
    print("Reference columns:", list(reference_df.columns))

    try:
        title_matcher = TitleEmbedMatcher(
            model_name="intfloat/e5-small-v2",
            threshold=0.78,
        )
        print("Loaded title embedding matcher: intfloat/e5-small-v2")
    except Exception as exc:
        title_matcher = None
        print(f"Could not load title embedding matcher. Continuing without it. ({exc})")

    engine = AccessRecommendationEngine(
        min_confidence=0.4,
        title_matcher=title_matcher,
    )

    title = "Customer Service Rep"
    department = "CE Customer Support"
    employee_type = "Student"
    supervisor = None
    copy_from_netid = "btang5"

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
