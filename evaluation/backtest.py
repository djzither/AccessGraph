"""
AccessGraph -- Backtesting / Evaluation Pipeline
================================================

Purpose
-------
This evaluates AccessGraph in a way that better matches the real product goal:

    Can the system recover normal, role-based access without suggesting unsafe
    or obviously wrong permissions?

Why this version is better than pure random hiding
--------------------------------------------------
The old test randomly hid any AD group from a user. That is useful as a smoke
test, but it can unfairly punish the recommender for not recovering rare,
legacy, one-off, manager-only, or exception-based access.

This version separates hidden groups into:

    Recoverable groups:
        Groups that appear in at least `min_peer_support` other users after the
        evaluated user is removed from the dataset.

    Rare / unique groups:
        Groups with weak or no peer evidence. These are tracked separately and
        should usually not be expected as automatic recommendations.

Methodology
-----------
For each sampled user:

  1. Treat their actual AD groups as ground truth.
  2. Remove that user from the dataset to avoid leakage.
  3. Compute peer support for each of their groups.
  4. Hide only recoverable groups by default.
  5. Run AccessRecommendationEngine using title, department, and employee type.
  6. Measure whether hidden recoverable groups appear in top-K recommendations.
  7. Also measure precision, false positives, high-risk surfaced groups, and
     high-risk false positives.

Usage
-----
  python -m evaluation.backtest
  python -m evaluation.backtest --n 100 --hide 0.3 --k 3 5 10
  python -m evaluation.backtest --min-peer-support 3 --min-score 0.45
  python -m evaluation.backtest --output results.csv
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is importable when running as a script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine
from DataLayer.access_exclusions import filter_group_list, filter_reference_df, filter_recommendations_df, filter_user_groups_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_reference_df() -> pd.DataFrame:
    """Properly structured empty DataFrame for when reference sheets are absent."""
    return pd.DataFrame(columns=[
        "JobTitle",
        "Department",
        "EmployeeType",
        "Supervisor",
        "ReferenceEmployeeName",
        "AccessCategory",
        "AccessName",
        "AccessNameClean",
        "SourceFile",
    ])


def _as_group_list(value) -> list[str]:
    """Safely normalize GroupsList values."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return filter_group_list([str(g).strip() for g in value if str(g).strip()])

    # Handles numpy arrays without importing numpy directly
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
        if isinstance(value, list):
            return filter_group_list([str(g).strip() for g in value if str(g).strip()])

    if isinstance(value, float) and pd.isna(value):
        return []

    # Fallback for semicolon-delimited strings
    return filter_group_list([g.strip() for g in str(value).split(";") if g.strip()])

def _norm(value) -> str:
    """Normalize labels for safer metric checks."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


# ---------------------------------------------------------------------------
# BacktestRunner
# ---------------------------------------------------------------------------

class BacktestRunner:
    DEFAULT_K_VALUES = [3, 5, 10]

    def __init__(
        self,
        users_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        k_values: Optional[list[int]] = None,
        hide_fraction: float = 0.3,
        min_groups_required: int = 3,
        min_peer_support: int = 2,
        min_engine_confidence: float = 0.3,
        score_threshold: float = 0.35,
        seed: int = 42,
    ):
        self.users_df = filter_user_groups_df(users_df)
        self.reference_df = (
            filter_reference_df(reference_df)
            if reference_df is not None and not reference_df.empty
            else _empty_reference_df()
        )
        self.k_values = sorted(k_values or self.DEFAULT_K_VALUES)
        self.hide_fraction = hide_fraction
        self.min_groups_required = min_groups_required
        self.min_peer_support = min_peer_support
        self.min_engine_confidence = min_engine_confidence
        self.score_threshold = score_threshold
        self.seed = seed
        self.rng = random.Random(seed)

        self._validate_inputs()
        self.users_df["GroupsList"] = self.users_df["GroupsList"].apply(_as_group_list)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def eligible_users(self) -> pd.DataFrame:
        """Return users with enough total groups to evaluate."""
        df = self.users_df.copy()
        df["_group_count"] = df["GroupsList"].apply(len)
        eligible = df[df["_group_count"] >= self.min_groups_required]
        return eligible.drop(columns=["_group_count"]).reset_index(drop=True)

    def run_single(self, sam_account_name: str) -> Optional[dict]:
        """Run one leave-one-user-out backtest case."""
        rows = self.users_df[self.users_df["SamAccountName"] == sam_account_name]
        if rows.empty:
            return {"SamAccountName": sam_account_name, "Error": "User not found"}

        user = rows.iloc[0]
        ground_truth = set(_as_group_list(user["GroupsList"]))

        if len(ground_truth) < self.min_groups_required:
            return None

        # Remove evaluated user first to avoid leakage.
        test_df = self.users_df[self.users_df["SamAccountName"] != sam_account_name].copy()
        peer_counts = self._group_counts(test_df)

        recoverable_groups = sorted([
            g for g in ground_truth
            if peer_counts.get(g, 0) >= self.min_peer_support
        ])
        rare_groups = sorted([
            g for g in ground_truth
            if 0 < peer_counts.get(g, 0) < self.min_peer_support
        ])
        unique_groups = sorted([
            g for g in ground_truth
            if peer_counts.get(g, 0) == 0
        ])

        # This is the main improvement: hide only groups that the model has
        # some realistic chance of recovering from peer evidence.
        if not recoverable_groups:
            return None

        n_hide = max(1, int(len(recoverable_groups) * self.hide_fraction))
        n_hide = min(n_hide, len(recoverable_groups))
        hidden = set(self.rng.sample(recoverable_groups, n_hide))

        employee_type = "Full Time"
        if "EmployeeType" in user.index and pd.notna(user.get("EmployeeType")):
            employee_type = str(user["EmployeeType"]).strip()

        try:
            engine = AccessRecommendationEngine(
                min_confidence=self.min_engine_confidence
            )
            recs = engine.recommend_for_hire(
                users_df=test_df,
                reference_df=self.reference_df,
                title=str(user["Title"]),
                department=str(user["Department"]),
                employee_type=employee_type,
                # No new_hire_netid: simulating someone not yet in AD.
            )
        except Exception as exc:
            return {"SamAccountName": sam_account_name, "Error": str(exc)}

        return self._compute_metrics(
            sam_account_name=sam_account_name,
            ground_truth=ground_truth,
            hidden=hidden,
            recoverable_groups=set(recoverable_groups),
            rare_groups=set(rare_groups),
            unique_groups=set(unique_groups),
            recs=filter_recommendations_df(recs),
        )

    def run_batch(self, n_users: int = 50) -> pd.DataFrame:
        """Run the backtest for up to n_users sampled eligible users."""
        eligible = self.eligible_users()

        if eligible.empty:
            print("  No eligible users found. Try lowering --min-groups.")
            return pd.DataFrame()

        sample_size = min(n_users, len(eligible))
        sampled = eligible.sample(n=sample_size, random_state=self.seed)

        width = len(str(sample_size))
        print(f"  Eligible users     : {len(eligible):,} / {len(self.users_df):,} total")
        print(f"  Evaluating         : {sample_size}")
        print(f"  Min peer support   : {self.min_peer_support}")
        print(f"  Score threshold    : >= {self.score_threshold:.0%}\n")

        results = []
        errors = 0
        skipped = 0

        for i, (_, user) in enumerate(sampled.iterrows(), 1):
            sam = str(user["SamAccountName"])
            result = self.run_single(sam)

            if result is None:
                skipped += 1
                print(f"  [{i:{width}}/{sample_size}]  SKIP  {sam}  -- no recoverable groups")
            elif "Error" in result:
                errors += 1
                print(f"  [{i:{width}}/{sample_size}]  ERROR {sam}  -- {result['Error']}")
            else:
                k_last = self.k_values[-1]
                recall_str = f"RecoverableRecall@{k_last}={result.get(f'RecoverableRecall@{k_last}', 0):.0%}"
                print(
                    f"  [{i:{width}}/{sample_size}]  OK    {sam}"
                    f"  hidden={result['HiddenCount']}"
                    f"  recs={result['TotalRecs']}"
                    f"  {recall_str}"
                )
                results.append(result)

        if skipped:
            print(f"\n  {skipped} user(s) skipped because they had no recoverable groups.")
        if errors:
            print(f"  {errors} user(s) skipped due to engine errors.")

        return pd.DataFrame(results)

    def print_report(self, results: pd.DataFrame) -> None:
        """Print a formatted summary report."""
        if results.empty:
            print("No results to report.")
            return

        sep = "=" * 66
        print(f"\n{sep}")
        print("  AccessGraph -- Backtest Report")
        print(sep)
        print(f"  Users evaluated       : {len(results):,}")
        print(f"  Hide fraction         : {self.hide_fraction:.0%}")
        print(f"  Score threshold       : >= {self.score_threshold:.0%}")
        print(f"  K values              : {self.k_values}")
        print(f"  Min groups required   : {self.min_groups_required}")
        print(f"  Min peer support      : {self.min_peer_support}")
        print(sep)

        for k in self.k_values:
            print(f"\n  K = {k}")
            self._print_percent(results, f"Hit@{k}", f"Hit@{k}", ">= 1 hidden recoverable group in top-K")
            self._print_percent(results, f"RecoverableRecall@{k}", f"RecoverableRecall@{k}", "hidden recoverable groups found")
            self._print_percent(results, f"Precision@{k}", f"Precision@{k}", "top-K recs actually in user's full ground truth")
            self._print_percent(results, f"FalsePositiveRate@{k}", f"FalsePositiveRate@{k}", "top-K recs not in user's full ground truth")

        print()
        self._print_average(results, "AvgConfidence", "Avg recommendation confidence", percent=True)
        self._print_average(results, "HiddenCount", "Avg groups hidden per user")
        self._print_average(results, "RecoverableGroupCount", "Avg recoverable groups per user")
        self._print_average(results, "RareGroupCount", "Avg rare groups per user")
        self._print_average(results, "UniqueGroupCount", "Avg unique groups per user")
        self._print_average(results, "TotalRecs", "Avg recs returned, all")
        self._print_average(results, "RecsAboveThreshold", f"Avg recs above {self.score_threshold:.0%}")

        if "HighRiskFalsePositives" in results.columns:
            total = int(results["HighRiskFalsePositives"].sum())
            per_user = results["HighRiskFalsePositives"].mean()
            print(f"  High-risk false positives : {total} total  ({per_user:.2f} per user)")

        if "HighRiskSurfaced" in results.columns:
            total = int(results["HighRiskSurfaced"].sum())
            per_user = results["HighRiskSurfaced"].mean()
            print(f"  High-risk surfaced        : {total} total  ({per_user:.2f} per user)")

        if "ManualReviewRate" in results.columns:
            print(f"  Manual review rate        : {results['ManualReviewRate'].mean():.1%}")

        print(f"\n{sep}\n")

    # ------------------------------------------------------------------ #
    # Internal logic
    # ------------------------------------------------------------------ #

    def _validate_inputs(self) -> None:
        required = {"SamAccountName", "Title", "Department", "GroupsList"}
        missing = required - set(self.users_df.columns)
        if missing:
            raise ValueError(f"users_df is missing required columns: {sorted(missing)}")
        if not 0 < self.hide_fraction < 1:
            raise ValueError("hide_fraction must be between 0 and 1")
        if self.min_groups_required < 1:
            raise ValueError("min_groups_required must be at least 1")
        if self.min_peer_support < 1:
            raise ValueError("min_peer_support must be at least 1")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")

    @staticmethod
    def _group_counts(users_df: pd.DataFrame) -> Counter:
        counts = Counter()
        for groups in users_df["GroupsList"]:
            counts.update(set(_as_group_list(groups)))
        return counts

    def _compute_metrics(
        self,
        sam_account_name: str,
        ground_truth: set[str],
        hidden: set[str],
        recoverable_groups: set[str],
        rare_groups: set[str],
        unique_groups: set[str],
        recs: pd.DataFrame,
    ) -> dict:
        metrics = {
            "SamAccountName": sam_account_name,
            "TotalGroups": len(ground_truth),
            "HiddenCount": len(hidden),
            "RecoverableGroupCount": len(recoverable_groups),
            "RareGroupCount": len(rare_groups),
            "UniqueGroupCount": len(unique_groups),
            "TotalRecs": len(recs),
        }

        if recs.empty:
            return self._fill_zero_metrics(metrics)

        if "GroupName" not in recs.columns or "FinalScore" not in recs.columns:
            metrics["Error"] = "Recommendation output missing GroupName or FinalScore"
            return metrics

        recs_sorted = (
            recs[recs["FinalScore"] >= self.score_threshold]
            .sort_values("FinalScore", ascending=False)
            .reset_index(drop=True)
        )

        metrics["RecsAboveThreshold"] = len(recs_sorted)
        metrics["AvgConfidence"] = (
            round(float(recs_sorted["FinalScore"].mean()), 3)
            if not recs_sorted.empty else 0.0
        )

        for k in self.k_values:
            top_k_list = recs_sorted.head(k)["GroupName"].tolist()
            top_k = set(top_k_list)
            denom = len(top_k_list)

            hidden_hits = len(hidden & top_k)
            true_hits = len(ground_truth & top_k)
            false_hits = len(top_k - ground_truth)

            metrics[f"Hit@{k}"] = 1 if hidden_hits > 0 else 0
            metrics[f"RecoverableRecall@{k}"] = round(hidden_hits / len(hidden), 3) if hidden else 0.0

            # Precision should divide by actual returned top-K count, not always k.
            metrics[f"Precision@{k}"] = round(true_hits / denom, 3) if denom else 0.0
            metrics[f"FalsePositive@{k}"] = false_hits
            metrics[f"FalsePositiveRate@{k}"] = round(false_hits / denom, 3) if denom else 0.0

        self._add_safety_metrics(metrics, recs_sorted, ground_truth)
        return metrics

    def _fill_zero_metrics(self, metrics: dict) -> dict:
        for k in self.k_values:
            metrics[f"Hit@{k}"] = 0
            metrics[f"RecoverableRecall@{k}"] = 0.0
            metrics[f"Precision@{k}"] = 0.0
            metrics[f"FalsePositive@{k}"] = 0
            metrics[f"FalsePositiveRate@{k}"] = 0.0

        metrics["AvgConfidence"] = 0.0
        metrics["RecsAboveThreshold"] = 0
        metrics["HighRiskFalsePositives"] = 0
        metrics["HighRiskSurfaced"] = 0
        metrics["ManualReviewRate"] = 0.0
        return metrics

    @staticmethod
    def _add_safety_metrics(metrics: dict, recs_sorted: pd.DataFrame, ground_truth: set[str]) -> None:
        if recs_sorted.empty:
            metrics["HighRiskFalsePositives"] = 0
            metrics["HighRiskSurfaced"] = 0
            metrics["ManualReviewRate"] = 0.0
            return

        df = recs_sorted.copy()
        df["_is_true"] = df["GroupName"].isin(ground_truth)

        if "RiskLevel" in df.columns:
            df["_risk"] = df["RiskLevel"].apply(_norm)
        else:
            df["_risk"] = ""

        if "FinalDecision" in df.columns:
            df["_decision"] = df["FinalDecision"].apply(_norm)
        else:
            df["_decision"] = ""

        high_risk = df["_risk"].eq("high")
        manual_review = df["_decision"].eq("manual review")

        metrics["HighRiskSurfaced"] = int(high_risk.sum())
        metrics["HighRiskFalsePositives"] = int((high_risk & ~manual_review & ~df["_is_true"]).sum())
        metrics["ManualReviewRate"] = round(float(manual_review.mean()), 3)

    @staticmethod
    def _print_percent(results: pd.DataFrame, col: str, label: str, description: str) -> None:
        if col not in results.columns:
            return
        print(f"    {label:<24} {results[col].mean():>6.1%}   ({description})")

    @staticmethod
    def _print_average(results: pd.DataFrame, col: str, label: str, percent: bool = False) -> None:
        if col not in results.columns:
            return
        value = results[col].mean()
        if percent:
            print(f"  {label:<28}: {value:.1%}")
        else:
            print(f"  {label:<28}: {value:.1f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from DataLayer.cleaner import DataCleaner
    from DataLayer.rights_sheets_loader import RightsSheetsLoader

    parser = argparse.ArgumentParser(
        description="AccessGraph backtest pipeline -- evaluates recoverable access recommendations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/processed/clean_users.parquet",
        help="Path to cleaned user parquet file",
    )
    parser.add_argument(
        "--raw",
        default="data/raw",
        help="Path to raw data folder for reference sheets",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of users to evaluate",
    )
    parser.add_argument(
        "--hide",
        type=float,
        default=0.3,
        help="Fraction of recoverable groups to hide",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        help="K values for metrics",
    )
    parser.add_argument(
        "--min-groups",
        type=int,
        default=3,
        help="Skip users with fewer than this many total groups",
    )
    parser.add_argument(
        "--min-peer-support",
        type=int,
        default=2,
        help="Only hide groups held by at least this many other users",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Min confidence threshold passed to AccessRecommendationEngine",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="Ignore recommendations below this FinalScore when computing top-K metrics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save per-user results CSV",
    )
    args = parser.parse_args()

    print("Loading user data...")
    users_df = DataCleaner(args.data).load_cleaned()
    print(f"  {len(users_df):,} users loaded.")

    print("Loading reference sheets...")
    try:
        reference_df = RightsSheetsLoader(args.raw).load_reference_sheets()
        print(f"  {len(reference_df):,} reference entries loaded.")
    except Exception as exc:
        reference_df = _empty_reference_df()
        print(f"  Not found ({exc}) -- running without reference signal.")

    runner = BacktestRunner(
        users_df=users_df,
        reference_df=reference_df,
        k_values=args.k,
        hide_fraction=args.hide,
        min_groups_required=args.min_groups,
        min_peer_support=args.min_peer_support,
        min_engine_confidence=args.confidence,
        score_threshold=args.min_score,
        seed=args.seed,
    )

    print("\nRunning backtest...")
    results = runner.run_batch(n_users=args.n)
    runner.print_report(results)

    if args.output and not results.empty:
        out_path = Path(args.output)
        results.to_csv(out_path, index=False)
        print(f"Per-user results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
