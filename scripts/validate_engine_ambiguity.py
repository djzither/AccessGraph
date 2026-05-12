from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from DataLayer.access_exclusions import filter_reference_df, filter_user_groups_df
from DataLayer.rights_sheets_loader import RightsSheetsLoader
from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


DEFAULT_COLLISIONS_CSV = Path("reports/reference_collisions.csv")
DEFAULT_USERS_PARQUET = Path("data/processed/clean_users.parquet")
DEFAULT_REFERENCE_PARQUET = Path("data/processed/access_reference.parquet")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_CSV = Path("reports/engine_ambiguity_validation.csv")


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _norm_text(value: object) -> str:
    return _clean_text(value).lower()


def _load_reference_df(reference_parquet: Path, raw_dir: Path) -> tuple[pd.DataFrame, str]:
    if reference_parquet.exists():
        return pd.read_parquet(reference_parquet), f"parquet:{reference_parquet.as_posix()}"
    if raw_dir.exists():
        return RightsSheetsLoader(raw_path=raw_dir).load_reference_sheets(), f"rebuild:{raw_dir.as_posix()}"
    raise FileNotFoundError(
        "Could not load reference data. Expected either "
        f"{reference_parquet.as_posix()} or raw dir {raw_dir.as_posix()}."
    )


def _decision_count(df: pd.DataFrame, decision: str) -> int:
    if "FinalDecision" not in df.columns or df.empty:
        return 0
    return int((df["FinalDecision"] == decision).sum())


def _summary_row(role_title: str, role_dept: str, case_name: str, recs: pd.DataFrame) -> dict:
    avg_score = float(recs["FinalScore"].mean()) if "FinalScore" in recs.columns and not recs.empty else 0.0
    ambiguous_rows = (
        int(recs["AmbiguousReferenceTemplate"].fillna(False).sum())
        if "AmbiguousReferenceTemplate" in recs.columns and not recs.empty
        else 0
    )
    return {
        "RowType": "summary",
        "Case": case_name,
        "JobTitleClean": role_title,
        "DepartmentClean": role_dept,
        "GroupName": "",
        "FinalScore": round(avg_score, 4),
        "FinalDecision": "",
        "InReferenceSheet": "",
        "ReferenceTemplateCount": "",
        "AmbiguousReferenceTemplate": "",
        "Reason": "",
        "StrongRecommendCount": _decision_count(recs, "Strong Recommend"),
        "SuggestCount": _decision_count(recs, "Suggest"),
        "ManualReviewCount": _decision_count(recs, "Manual Review"),
        "LowConfidenceCount": _decision_count(recs, "Low Confidence"),
        "AverageFinalScore": round(avg_score, 4),
        "AmbiguousRowsCount": ambiguous_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate production AccessRecommendationEngine behavior for ambiguous reference templates."
    )
    parser.add_argument("--top", type=int, default=10, help="Number of high-collision roles to evaluate.")
    parser.add_argument("--collisions-csv", default=str(DEFAULT_COLLISIONS_CSV), help="Collision report CSV path.")
    parser.add_argument("--users-parquet", default=str(DEFAULT_USERS_PARQUET), help="Users parquet path.")
    parser.add_argument("--reference-parquet", default=str(DEFAULT_REFERENCE_PARQUET), help="Reference parquet path.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Raw dir fallback for references.")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="Engine min_confidence for AD signal.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_CSV), help="Output CSV path.")
    parser.add_argument(
        "--fail-on-zero-ambiguity",
        action="store_true",
        help="Fail if all ambiguous_context summary rows have AmbiguousRowsCount == 0.",
    )
    args = parser.parse_args()

    collisions_csv = Path(args.collisions_csv)
    users_parquet = Path(args.users_parquet)
    reference_parquet = Path(args.reference_parquet)
    raw_dir = Path(args.raw_dir)
    out_csv = Path(args.out)

    if not collisions_csv.exists():
        raise FileNotFoundError(f"Collision report not found: {collisions_csv.as_posix()}")
    if not users_parquet.exists():
        raise FileNotFoundError(f"Users parquet not found: {users_parquet.as_posix()}")

    collisions = pd.read_csv(collisions_csv)
    top_roles = (
        collisions[collisions["TemplateCount"] >= 2]
        .sort_values("Severity", ascending=False)
        .head(max(int(args.top), 1))
        .reset_index(drop=True)
    )
    if top_roles.empty:
        print("No roles with TemplateCount >= 2 found in collision report.")
        return

    users_df = filter_user_groups_df(pd.read_parquet(users_parquet))
    reference_df_raw, ref_source = _load_reference_df(reference_parquet, raw_dir)
    reference_df = filter_reference_df(reference_df_raw)

    ref_norm = reference_df.copy()
    ref_norm["JobTitleClean"] = ref_norm["JobTitle"].astype(str).str.lower().str.strip()
    ref_norm["DepartmentClean"] = ref_norm["Department"].astype(str).str.lower().str.strip()

    engine = AccessRecommendationEngine(min_confidence=float(args.min_confidence))

    rows: list[dict] = []

    for _, role in top_roles.iterrows():
        title_clean = _norm_text(role["JobTitleClean"])
        dept_clean = _norm_text(role["DepartmentClean"])

        role_ref = ref_norm[
            (ref_norm["JobTitleClean"] == title_clean)
            & (ref_norm["DepartmentClean"] == dept_clean)
        ].copy()
        if role_ref.empty:
            continue

        template_row = role_ref.iloc[0]
        narrowed_employee_type = _clean_text(template_row.get("EmployeeType"))
        narrowed_supervisor = _clean_text(template_row.get("Supervisor")) or None

        # Case 1: missing employee_type + missing supervisor context
        recs_ambiguous = engine.recommend_for_hire(
            users_df=users_df,
            reference_df=reference_df,
            title=title_clean,
            department=dept_clean,
            employee_type=None,
            supervisor=None,
            copy_from_netid=None,
            new_hire_netid=None,
        )

        # Case 2: narrowed context from one observed template
        recs_narrowed = engine.recommend_for_hire(
            users_df=users_df,
            reference_df=reference_df,
            title=title_clean,
            department=dept_clean,
            employee_type=narrowed_employee_type if narrowed_employee_type else "Full Time",
            supervisor=narrowed_supervisor,
            copy_from_netid=None,
            new_hire_netid=None,
        )

        for case_name, recs in (("ambiguous_context", recs_ambiguous), ("narrowed_context", recs_narrowed)):
            if recs.empty:
                rows.append(
                    {
                        "RowType": "detail",
                        "Case": case_name,
                        "JobTitleClean": title_clean,
                        "DepartmentClean": dept_clean,
                        "GroupName": "",
                        "FinalScore": 0.0,
                        "FinalDecision": "NO_ROWS",
                        "InReferenceSheet": False,
                        "ReferenceTemplateCount": 0,
                        "AmbiguousReferenceTemplate": False,
                        "Reason": "No recommendations returned.",
                        "StrongRecommendCount": "",
                        "SuggestCount": "",
                        "ManualReviewCount": "",
                        "LowConfidenceCount": "",
                        "AverageFinalScore": "",
                        "AmbiguousRowsCount": "",
                    }
                )
            else:
                for _, r in recs.iterrows():
                    rows.append(
                        {
                            "RowType": "detail",
                            "Case": case_name,
                            "JobTitleClean": title_clean,
                            "DepartmentClean": dept_clean,
                            "GroupName": _clean_text(r.get("GroupName")),
                            "FinalScore": float(r.get("FinalScore", 0.0)),
                            "FinalDecision": _clean_text(r.get("FinalDecision")),
                            "InReferenceSheet": bool(r.get("InReferenceSheet", False)),
                            "ReferenceTemplateCount": int(r.get("ReferenceTemplateCount", 0)),
                            "AmbiguousReferenceTemplate": bool(r.get("AmbiguousReferenceTemplate", False)),
                            "Reason": _clean_text(r.get("Reason")),
                            "StrongRecommendCount": "",
                            "SuggestCount": "",
                            "ManualReviewCount": "",
                            "LowConfidenceCount": "",
                            "AverageFinalScore": "",
                            "AmbiguousRowsCount": "",
                        }
                    )

            rows.append(_summary_row(title_clean, dept_clean, case_name, recs))

    report_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_csv, index=False)

    # Sanity check: collision-role ambiguity validation should exercise ambiguity rows.
    ambiguous_summaries = report_df[
        (report_df["RowType"] == "summary")
        & (report_df["Case"] == "ambiguous_context")
    ].copy()
    if not ambiguous_summaries.empty:
        all_zero_ambiguity = (
            ambiguous_summaries["AmbiguousRowsCount"]
            .fillna(0)
            .astype(int)
            .eq(0)
            .all()
        )
        if all_zero_ambiguity:
            msg = (
                "All ambiguous_context summaries have AmbiguousRowsCount=0; "
                "ambiguity path may not be exercised."
            )
            if args.fail_on_zero_ambiguity:
                raise RuntimeError(msg)
            print(f"WARNING: {msg}")

    print("Engine ambiguity validation complete")
    print(f"Reference source: {ref_source}")
    print(f"Roles evaluated: {report_df[['JobTitleClean','DepartmentClean']].drop_duplicates().shape[0]}")
    print(f"Wrote: {out_csv.as_posix()}")


if __name__ == "__main__":
    main()

