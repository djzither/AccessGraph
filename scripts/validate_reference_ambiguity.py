from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from DataLayer.rights_sheets_loader import RightsSheetsLoader
from DeterministicLayer.reference_matcher import ReferenceMatcher


DEFAULT_COLLISIONS_CSV = Path("reports/reference_collisions.csv")
DEFAULT_REFERENCE_PARQUET = Path("data/processed/access_reference.parquet")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_CSV = Path("reports/reference_ambiguity_validation.csv")


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).lower().strip()


def _load_reference_df(reference_parquet: Path, raw_dir: Path) -> tuple[pd.DataFrame, str]:
    if reference_parquet.exists():
        return pd.read_parquet(reference_parquet), f"parquet:{reference_parquet.as_posix()}"
    if raw_dir.exists():
        return RightsSheetsLoader(raw_path=raw_dir).load_reference_sheets(), f"rebuild:{raw_dir.as_posix()}"
    raise FileNotFoundError(
        "Could not load reference data. Expected either "
        f"{reference_parquet.as_posix()} or raw dir {raw_dir.as_posix()}."
    )


def _build_fake_recommendations(role_df: pd.DataFrame, max_groups: int = 10) -> pd.DataFrame:
    access_names = (
        role_df["AccessName"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    access_names = [name for name in access_names if name]
    access_names = list(dict.fromkeys(access_names))[:max_groups]
    if not access_names:
        access_names = ["<no-access-name-found>"]

    rows = []
    for i, group_name in enumerate(access_names):
        # Deterministic score ladder to expose decision bucket behavior.
        score = 0.9 if i % 3 == 0 else (0.7 if i % 3 == 1 else 0.5)
        rows.append(
            {
                "GroupName": group_name,
                "Score": score,
                "RiskLevel": "Low",
                "UserCountWithGroup": 7,
                "TotalUsersInRole": 10,
            }
        )
    return pd.DataFrame(rows)


def _decision_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    strong = int((df["FinalDecision"] == "Strong Recommend").sum())
    suggest = int((df["FinalDecision"] == "Suggest").sum())
    manual = int((df["FinalDecision"] == "Manual Review").sum())
    return strong, suggest, manual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ambiguity-aware behavior in ReferenceMatcher on top collision roles."
    )
    parser.add_argument("--top", type=int, default=10, help="How many high-collision roles to validate.")
    parser.add_argument("--collisions-csv", default=str(DEFAULT_COLLISIONS_CSV), help="Collision report CSV path.")
    parser.add_argument("--reference-parquet", default=str(DEFAULT_REFERENCE_PARQUET), help="Reference parquet path.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Raw dir fallback for rebuilding references.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_CSV), help="Validation output CSV path.")
    args = parser.parse_args()

    collisions_csv = Path(args.collisions_csv)
    reference_parquet = Path(args.reference_parquet)
    raw_dir = Path(args.raw_dir)
    out_csv = Path(args.out)

    if not collisions_csv.exists():
        raise FileNotFoundError(f"Collision report not found: {collisions_csv.as_posix()}")

    collisions = pd.read_csv(collisions_csv)
    top_collisions = (
        collisions[collisions["TemplateCount"] >= 2]
        .sort_values("Severity", ascending=False)
        .head(max(int(args.top), 1))
        .reset_index(drop=True)
    )
    if top_collisions.empty:
        print("No rows with TemplateCount >= 2 found in collision report.")
        return

    reference_df, source = _load_reference_df(reference_parquet, raw_dir)
    matcher = ReferenceMatcher(reference_df)
    ref = matcher.reference_df.copy()

    rows: list[dict] = []

    for _, role in top_collisions.iterrows():
        title_clean = _clean_text(role["JobTitleClean"])
        dept_clean = _clean_text(role["DepartmentClean"])

        role_df = ref[
            (ref["JobTitleClean"] == title_clean)
            & (ref["DepartmentClean"] == dept_clean)
        ].copy()
        if role_df.empty:
            continue

        fake_recs = _build_fake_recommendations(role_df)

        # Run 1: missing employee_type and supervisor (ambiguous context).
        ambiguous_run = matcher.match_recommendations(
            recommendations=fake_recs.copy(),
            title=title_clean,
            department=dept_clean,
            employee_type=None,
            supervisor=None,
        )

        # Run 2: narrowed context from one observed template when available.
        template_row = role_df.iloc[0]
        template_employee_type = template_row.get("EmployeeType")
        template_supervisor = template_row.get("Supervisor")
        narrowed_run = matcher.match_recommendations(
            recommendations=fake_recs.copy(),
            title=title_clean,
            department=dept_clean,
            employee_type=None if pd.isna(template_employee_type) else str(template_employee_type),
            supervisor=None if pd.isna(template_supervisor) else str(template_supervisor),
        )

        a_strong, a_suggest, a_manual = _decision_counts(ambiguous_run)
        n_strong, n_suggest, n_manual = _decision_counts(narrowed_run)

        rows.append(
            {
                "JobTitleClean": title_clean,
                "DepartmentClean": dept_clean,
                "CollisionTemplateCount": int(role["TemplateCount"]),
                "CollisionSeverity": float(role["Severity"]),
                "Ambiguous_Run_ReferenceTemplateCount": int(ambiguous_run["ReferenceTemplateCount"].iloc[0]),
                "Ambiguous_Run_AmbiguousReferenceTemplate": bool(ambiguous_run["AmbiguousReferenceTemplate"].iloc[0]),
                "Ambiguous_Run_StrongRecommendCount": a_strong,
                "Ambiguous_Run_SuggestCount": a_suggest,
                "Ambiguous_Run_ManualReviewCount": a_manual,
                "Ambiguous_Run_SampleReasons": " | ".join(
                    ambiguous_run["Reason"].astype(str).drop_duplicates().head(2).tolist()
                ),
                "Narrowed_Run_ReferenceTemplateCount": int(narrowed_run["ReferenceTemplateCount"].iloc[0]),
                "Narrowed_Run_AmbiguousReferenceTemplate": bool(narrowed_run["AmbiguousReferenceTemplate"].iloc[0]),
                "Narrowed_Run_StrongRecommendCount": n_strong,
                "Narrowed_Run_SuggestCount": n_suggest,
                "Narrowed_Run_ManualReviewCount": n_manual,
                "Narrowed_Run_SampleReasons": " | ".join(
                    narrowed_run["Reason"].astype(str).drop_duplicates().head(2).tolist()
                ),
                "ChosenTemplateEmployeeType": "" if pd.isna(template_employee_type) else str(template_employee_type),
                "ChosenTemplateSupervisor": "" if pd.isna(template_supervisor) else str(template_supervisor),
            }
        )

    report_df = pd.DataFrame(rows).sort_values(
        ["CollisionSeverity", "CollisionTemplateCount"],
        ascending=[False, False],
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_csv, index=False)

    print("Reference ambiguity validation complete")
    print(f"Reference source: {source}")
    print(f"Roles evaluated: {len(report_df)}")
    print(f"Wrote: {out_csv.as_posix()}")
    if not report_df.empty:
        print()
        print(
            report_df[
                [
                    "JobTitleClean",
                    "DepartmentClean",
                    "Ambiguous_Run_ReferenceTemplateCount",
                    "Ambiguous_Run_AmbiguousReferenceTemplate",
                    "Ambiguous_Run_StrongRecommendCount",
                    "Ambiguous_Run_SuggestCount",
                    "Ambiguous_Run_ManualReviewCount",
                    "Narrowed_Run_ReferenceTemplateCount",
                    "Narrowed_Run_AmbiguousReferenceTemplate",
                    "Narrowed_Run_StrongRecommendCount",
                    "Narrowed_Run_SuggestCount",
                    "Narrowed_Run_ManualReviewCount",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()

