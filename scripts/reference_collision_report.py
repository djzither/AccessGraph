from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import pandas as pd

from DataLayer.rights_sheets_loader import RightsSheetsLoader


DEFAULT_REFERENCE_PARQUET = Path("data/processed/access_reference.parquet")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_CSV_OUT = Path("reports/reference_collisions.csv")


def _clean_text(value: object) -> str:
    """Match existing pipeline-style normalization: lowercase + strip."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).lower().strip()


def _stable_token(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def _as_set(values: Iterable[str]) -> set[str]:
    return {v for v in values if v}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


@dataclass(frozen=True)
class TemplateKey:
    employee_type: str
    job_title: str
    department: str
    supervisor: str

    def display(self) -> str:
        sup_token = _stable_token("SUP", self.supervisor) if self.supervisor else "SUP_<missing>"
        et = self.employee_type or "<missing>"
        return f"({et}, {sup_token})"


def load_reference_df(
    *,
    reference_parquet: Path = DEFAULT_REFERENCE_PARQUET,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> tuple[pd.DataFrame, str]:
    """
    Load the normalized reference dataframe.

    Preference order:
    1) data/processed/access_reference.parquet
    2) rebuild via RightsSheetsLoader(data/raw)
    """
    if reference_parquet.exists():
        df = pd.read_parquet(reference_parquet)
        return df, f"parquet:{reference_parquet.as_posix()}"

    if raw_dir.exists():
        df = RightsSheetsLoader(raw_path=raw_dir).load_reference_sheets()
        return df, f"rebuild:{raw_dir.as_posix()}"

    raise FileNotFoundError(
        "Could not load reference data. Expected either "
        f"{reference_parquet.as_posix()} or raw dir {raw_dir.as_posix()}."
    )


def normalize_reference_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["EmployeeType", "JobTitle", "Department", "Supervisor", "AccessName"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Reference dataframe missing required columns: {missing}")

    out = df.copy()
    out["EmployeeTypeClean"] = out["EmployeeType"].apply(_clean_text)
    out["JobTitleClean"] = out["JobTitle"].apply(_clean_text)
    out["DepartmentClean"] = out["Department"].apply(_clean_text)
    out["SupervisorClean"] = out["Supervisor"].apply(_clean_text)
    out["AccessNameClean"] = out["AccessName"].apply(_clean_text)

    # Drop empty identity rows deterministically (audit-friendly).
    out = out[out["JobTitleClean"] != ""]
    out = out[out["DepartmentClean"] != ""]
    out = out[out["AccessNameClean"] != ""]

    return out


def compute_collisions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], dict]]:
    """
    Returns:
    - summary_df: one row per (JobTitleClean, DepartmentClean)
    - details: dict keyed by (title, dept) with template access sets for drill-down
    """
    details: dict[tuple[str, str], dict] = {}
    rows: list[dict] = []

    # Build per-template access sets.
    template_cols = ["EmployeeTypeClean", "JobTitleClean", "DepartmentClean", "SupervisorClean"]
    grouped = df.groupby(["JobTitleClean", "DepartmentClean"], dropna=False, sort=True)

    for (title, dept), role_df in grouped:
        # Templates within this title+dept
        templates = {}
        for key_vals, tdf in role_df.groupby(template_cols, dropna=False, sort=True):
            key = TemplateKey(
                employee_type=key_vals[0],
                job_title=key_vals[1],
                department=key_vals[2],
                supervisor=key_vals[3],
            )
            access_set = _as_set(tdf["AccessNameClean"].tolist())
            templates[key] = access_set

        template_keys = list(templates.keys())
        template_count = len(template_keys)
        supervisor_count = len({k.supervisor for k in template_keys if k.supervisor})
        employee_type_count = len({k.employee_type for k in template_keys if k.employee_type})

        jaccards: list[float] = []
        min_pair = None
        min_j = 1.0
        max_j = 0.0

        for i in range(template_count):
            for j in range(i + 1, template_count):
                a = templates[template_keys[i]]
                b = templates[template_keys[j]]
                score = _jaccard(a, b)
                jaccards.append(score)
                if score < min_j:
                    min_j = score
                    min_pair = (template_keys[i], template_keys[j])
                if score > max_j:
                    max_j = score

        median_j = median(jaccards) if jaccards else 1.0
        min_j = min_j if jaccards else 1.0
        max_j = max_j if jaccards else 1.0

        # Severity used only for ranking (deterministic).
        severity = round((max(template_count - 1, 0)) * (1.0 - max_j), 6)

        most_divergent = ""
        a_only_sample = ""
        b_only_sample = ""
        if min_pair is not None:
            a_key, b_key = min_pair
            a_set = templates[a_key]
            b_set = templates[b_key]
            a_only = sorted(a_set - b_set)
            b_only = sorted(b_set - a_set)
            most_divergent = f"{a_key.display()} vs {b_key.display()}"
            a_only_sample = ", ".join(a_only[:10])
            b_only_sample = ", ".join(b_only[:10])

        rows.append(
            {
                "JobTitleClean": title,
                "DepartmentClean": dept,
                "TemplateCount": template_count,
                "SupervisorCount": supervisor_count,
                "EmployeeTypeCount": employee_type_count,
                "MinJaccard": round(min_j, 4),
                "MedianJaccard": round(median_j, 4),
                "MaxJaccard": round(max_j, 4),
                "Severity": severity,
                "MostDivergentPair": most_divergent,
                "AOnlySample": a_only_sample,
                "BOnlySample": b_only_sample,
            }
        )

        details[(title, dept)] = {
            "templates": templates,
            "min_pair": min_pair,
        }

    summary_df = pd.DataFrame(rows).sort_values(
        ["Severity", "TemplateCount", "SupervisorCount", "EmployeeTypeCount"],
        ascending=[False, False, False, False],
    )
    return summary_df.reset_index(drop=True), details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect ambiguous/duplicated access templates in normalized reference sheets."
    )
    parser.add_argument("--top", type=int, default=25, help="How many collisions to print.")
    parser.add_argument(
        "--reference-parquet",
        default=str(DEFAULT_REFERENCE_PARQUET),
        help="Preferred reference parquet path.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RAW_DIR),
        help="Raw directory used to rebuild reference sheets if parquet is missing.",
    )
    parser.add_argument(
        "--csv-out",
        default="",
        help="Optional CSV export path (default: reports/reference_collisions.csv).",
    )
    args = parser.parse_args()

    reference_parquet = Path(args.reference_parquet)
    raw_dir = Path(args.raw_dir)

    reference_df, source = load_reference_df(reference_parquet=reference_parquet, raw_dir=raw_dir)
    normalized = normalize_reference_df(reference_df)

    summary_df, _details = compute_collisions(normalized)

    total_roles = int(summary_df.shape[0])
    ambiguous_roles = int((summary_df["TemplateCount"] >= 2).sum())

    print("AccessGraph Reference Collision Report")
    print(f"Loaded reference rows: {len(reference_df):,} (normalized rows: {len(normalized):,})")
    print(f"Source: {source}")
    print(f"Distinct (JobTitleClean, DepartmentClean): {total_roles:,}")
    print(f"Ambiguous (>=2 templates): {ambiguous_roles:,} ({(ambiguous_roles / max(total_roles, 1)):.1%})")
    print()

    collisions = summary_df[summary_df["TemplateCount"] >= 2].copy()
    if collisions.empty:
        print("No ambiguous title+department combinations found (TemplateCount < 2 for all).")
    else:
        top_n = max(int(args.top), 1)
        print(f"Top {min(top_n, len(collisions))} ambiguous title+department combinations")
        print(
            collisions.head(top_n)[
                [
                    "Severity",
                    "TemplateCount",
                    "SupervisorCount",
                    "EmployeeTypeCount",
                    "MinJaccard",
                    "MedianJaccard",
                    "MaxJaccard",
                    "JobTitleClean",
                    "DepartmentClean",
                    "MostDivergentPair",
                    "AOnlySample",
                    "BOnlySample",
                ]
            ].to_string(index=False)
        )

    csv_out = args.csv_out.strip()
    if csv_out:
        out_path = Path(csv_out)
    else:
        out_path = DEFAULT_CSV_OUT if args.csv_out != "" else None

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print()
        print(f"Wrote CSV: {out_path.as_posix()}")


if __name__ == "__main__":
    main()

