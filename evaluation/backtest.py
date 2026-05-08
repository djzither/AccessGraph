"""
AccessGraph -- Lightweight Backtesting / Evaluation Pipeline
=============================================================

Methodology
-----------
For each sampled user we treat their actual AD groups as ground truth,
then simulate a "new hire" scenario:

  1. Randomly hide a fraction of their groups.
  2. Remove the user from the dataset entirely so they cannot
     contaminate cohort confidence scores.
  3. Run AccessRecommendationEngine with only their visible role
     context (title, department).
  4. Measure whether the hidden groups appear in the top-K
     recommendations ranked by FinalScore.

Metrics
-------
  Hit@K                  -- 1 if any hidden group appears in top-K recs.
  Recall@K               -- |hidden n top-K| / |hidden|.
  AvgConfidence          -- mean FinalScore across all recommendations.
  HighRiskFalsePositives -- high-risk groups recommended that are not
                            in the user's actual ground-truth groups.

Usage (CLI)
-----------
  python -m evaluation.backtest
  python -m evaluation.backtest --n 100 --hide 0.4 --k 3 5 10 --output results.csv
"""

import random
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is importable when running as a script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_reference_df() -> pd.DataFrame:
    """Properly-structured empty DataFrame for when reference sheets are absent."""
    return pd.DataFrame(columns=[
        "JobTitle", "Department", "EmployeeType", "Supervisor",
        "ReferenceEmployeeName", "AccessCategory", "AccessName",
        "AccessNameClean", "SourceFile",
    ])


# ---------------------------------------------------------------------------
# BacktestRunner
# ---------------------------------------------------------------------------

class BacktestRunner:
    """
    Runs the leave-some-out backtest against AccessRecommendationEngine.

    Parameters
    ----------
    users_df              : cleaned AD user DataFrame (GroupsList must be present).
    reference_df          : reference access sheet DataFrame, or None to skip signal.
    k_values              : list of K thresholds for Hit@K / Recall@K.
    hide_fraction         : fraction of each user's groups to hide (0 < x < 1).
    min_groups_required   : skip users with fewer groups than this.
    min_engine_confidence : min_confidence passed to AccessRecommendationEngine.
    seed                  : random seed for reproducibility.
    """

    DEFAULT_K_VALUES = [3, 5, 10]

    def __init__(
        self,
        users_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        k_values: Optional[list] = None,
        hide_fraction: float = 0.3,
        min_groups_required: int = 3,
        min_engine_confidence: float = 0.3,
        score_threshold: float = 0.35,
        seed: int = 42,
    ):
        self.users_df = users_df.copy()
        self.reference_df = (
            reference_df
            if (reference_df is not None and not reference_df.empty)
            else _empty_reference_df()
        )
        self.k_values = sorted(k_values or self.DEFAULT_K_VALUES)
        self.hide_fraction = hide_fraction
        self.min_groups_required = min_groups_required
        self.min_engine_confidence = min_engine_confidence
        self.score_threshold = score_threshold
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def eligible_users(self) -> pd.DataFrame:
        """Return users who have enough groups to make hiding meaningful."""
        df = self.users_df.copy()
        df["_gc"] = df["GroupsList"].apply(len)
        eligible = df[df["_gc"] >= self.min_groups_required].drop(columns=["_gc"])
        return eligible.reset_index(drop=True)

    def run_single(self, sam_account_name: str) -> Optional[dict]:
        """
        Run the backtest for one user.

        Returns a metrics dict, or None if the user is ineligible.
        Returns a dict with an 'Error' key if the engine fails.
        """
        rows = self.users_df[self.users_df["SamAccountName"] == sam_account_name]
        if rows.empty:
            return {"SamAccountName": sam_account_name, "Error": "User not found"}

        user = rows.iloc[0]
        ground_truth: set = set(user["GroupsList"])

        if len(ground_truth) < self.min_groups_required:
            return None  # silently skip ineligible users

        # -- 1. Hide a random fraction of the user's groups ----------------
        sorted_groups = sorted(ground_truth)
        n_hide = max(1, int(len(sorted_groups) * self.hide_fraction))
        hidden: set = set(self.rng.sample(sorted_groups, n_hide))

        # -- 2. Remove user from dataset (simulate new hire not yet in AD) -
        test_df = self.users_df[
            self.users_df["SamAccountName"] != sam_account_name
        ].copy()

        # -- 3. Determine employee type (column absent in this dataset) -----
        employee_type = "Full Time"
        if "EmployeeType" in user.index and pd.notna(user.get("EmployeeType")):
            employee_type = str(user["EmployeeType"]).strip()

        # -- 4. Run the recommendation engine ------------------------------
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
                # No new_hire_netid: simulating someone not yet in the system.
            )
        except Exception as exc:
            return {"SamAccountName": sam_account_name, "Error": str(exc)}

        # -- 5. Compute and return metrics ---------------------------------
        return self._compute_metrics(
            sam_account_name=sam_account_name,
            hidden=hidden,
            ground_truth=ground_truth,
            recs=recs,
        )

    def run_batch(self, n_users: int = 50) -> pd.DataFrame:
        """
        Run the backtest for up to n_users randomly sampled eligible users.

        Returns a DataFrame with one row per successfully evaluated user.
        """
        eligible = self.eligible_users()

        if eligible.empty:
            print("  No eligible users found (try lowering --min-groups).")
            return pd.DataFrame()

        sample_size = min(n_users, len(eligible))
        sampled = eligible.sample(n=sample_size, random_state=42)

        width = len(str(sample_size))
        print(f"  Eligible users : {len(eligible):,} / {len(self.users_df):,} total")
        print(f"  Evaluating     : {sample_size}\n")

        results = []
        errors = 0

        for i, (_, user) in enumerate(sampled.iterrows(), 1):
            sam = str(user["SamAccountName"])
            result = self.run_single(sam)

            if result is None:
                continue
            elif "Error" in result:
                errors += 1
                print(f"  [{i:{width}}/{sample_size}]  SKIP  {sam}  -- {result['Error']}")
            else:
                k_last = self.k_values[-1]
                recall_str = f"Recall@{k_last}={result.get(f'Recall@{k_last}', 0):.0%}"
                print(
                    f"  [{i:{width}}/{sample_size}]  OK    {sam}"
                    f"  hidden={result['HiddenCount']}"
                    f"  recs={result['TotalRecs']}"
                    f"  {recall_str}"
                )
                results.append(result)

        if errors:
            print(f"\n  {errors} user(s) skipped due to engine errors.")

        return pd.DataFrame(results)

    def print_report(self, results: pd.DataFrame) -> None:
        """Print a formatted summary report to stdout."""
        if results.empty:
            print("No results to report.")
            return

        sep = "=" * 56
        print(f"\n{sep}")
        print("  AccessGraph -- Backtest Report")
        print(sep)
        print(f"  Users evaluated  : {len(results):,}")
        print(f"  Hide fraction    : {self.hide_fraction:.0%}")
        print(f"  Score threshold  : >= {self.score_threshold:.0%}  (recs below this ignored)")
        print(f"  K values         : {self.k_values}")
        print(f"  Min groups req.  : {self.min_groups_required}")
        print(sep)

        for k in self.k_values:
            hit_col  = f"Hit@{k}"
            rec_col  = f"Recall@{k}"
            prec_col = f"Precision@{k}"
            if hit_col not in results.columns:
                continue
            hit_rate  = results[hit_col].mean()
            recall    = results[rec_col].mean()
            precision = results[prec_col].mean() if prec_col in results.columns else 0.0
            print(f"\n  K = {k}")
            print(f"    Hit@{k:<4}       {hit_rate:>6.1%}   (>= 1 hidden group in top-{k})")
            print(f"    Recall@{k:<2}      {recall:>6.1%}   (fraction of hidden groups found)")
            print(f"    Precision@{k:<2}   {precision:>6.1%}   (of top-{k} recs, % user actually has)")

        print()

        if "AvgConfidence" in results.columns:
            print(f"  Avg recommendation confidence      : {results['AvgConfidence'].mean():.1%}")

        if "HighRiskFalsePositives" in results.columns:
            total = int(results["HighRiskFalsePositives"].sum())
            per_u = results["HighRiskFalsePositives"].mean()
            print(f"  High-risk false positives          : {total} total  ({per_u:.2f} per user)")

        if "HiddenCount" in results.columns:
            print(f"  Avg groups hidden per user         : {results['HiddenCount'].mean():.1f}")

        if "TotalRecs" in results.columns:
            print(f"  Avg recs returned (all)            : {results['TotalRecs'].mean():.1f}")

        if "RecsAboveThreshold" in results.columns:
            print(f"  Avg recs above {self.score_threshold:.0%} threshold         : {results['RecsAboveThreshold'].mean():.1f}")

        print(f"\n{sep}\n")

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _compute_metrics(
        self,
        sam_account_name: str,
        hidden: set,
        ground_truth: set,
        recs: pd.DataFrame,
    ) -> dict:
        """Compute all metrics for one user evaluation."""
        metrics: dict = {
            "SamAccountName": sam_account_name,
            "TotalGroups":    len(ground_truth),
            "HiddenCount":    len(hidden),
            "TotalRecs":      len(recs),
        }

        # Zero out everything if the engine returned no recommendations.
        if recs.empty:
            for k in self.k_values:
                metrics[f"Hit@{k}"]       = 0
                metrics[f"Recall@{k}"]    = 0.0
                metrics[f"Precision@{k}"] = 0.0
            metrics["AvgConfidence"]          = 0.0
            metrics["RecsAboveThreshold"]     = 0
            metrics["HighRiskFalsePositives"] = 0
            return metrics

        # Filter to recommendations the engine is confident about, then sort.
        # Recommendations below score_threshold are ignored for all metrics —
        # they represent noise the engine itself isn't sure about.
        recs_sorted = (
            recs[recs["FinalScore"] >= self.score_threshold]
            .sort_values("FinalScore", ascending=False)
            .reset_index(drop=True)
        )
        metrics["RecsAboveThreshold"] = len(recs_sorted)

        # Hit@K, Recall@K, Precision@K
        for k in self.k_values:
            top_k: set = set(recs_sorted.head(k)["GroupName"].tolist())

            # Recall: how many hidden groups did we find?
            hidden_hits = len(hidden & top_k)
            metrics[f"Hit@{k}"]    = 1 if hidden_hits > 0 else 0
            metrics[f"Recall@{k}"] = round(hidden_hits / len(hidden), 3) if hidden else 0.0

            # Precision: of top-K recommendations, how many are in the user's
            # full ground truth (all groups, not just hidden ones)?
            true_hits = len(ground_truth & top_k)
            metrics[f"Precision@{k}"] = round(true_hits / k, 3) if k > 0 else 0.0

        # Average FinalScore across all recommendations.
        metrics["AvgConfidence"] = round(float(recs_sorted["FinalScore"].mean()), 3)

        # High-risk false positives: high-risk groups that the engine recommends
        # (without a Manual Review flag) but which the user does not actually have.
        if "RiskLevel" in recs_sorted.columns and "FinalDecision" in recs_sorted.columns:
            hrfp_mask = (
                (recs_sorted["RiskLevel"] == "High")
                & (recs_sorted["FinalDecision"] != "Manual Review")
                & (~recs_sorted["GroupName"].isin(ground_truth))
            )
            metrics["HighRiskFalsePositives"] = int(hrfp_mask.sum())
        else:
            metrics["HighRiskFalsePositives"] = 0

        return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    from DataLayer.cleaner import DataCleaner
    from DataLayer.rights_sheets_loader import RightsSheetsLoader

    parser = argparse.ArgumentParser(
        description="AccessGraph backtest pipeline -- evaluates recommendation recall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", default="data/processed/clean_users.parquet",
        help="Path to cleaned user parquet file",
    )
    parser.add_argument(
        "--raw", default="data/raw",
        help="Path to raw data folder (for reference sheets)",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of users to evaluate",
    )
    parser.add_argument(
        "--hide", type=float, default=0.3,
        help="Fraction of each user's groups to hide (0 < x < 1)",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[3, 5, 10],
        help="K values for Hit@K / Recall@K",
    )
    parser.add_argument(
        "--min-groups", type=int, default=3,
        help="Skip users with fewer than this many groups",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Min AD confidence threshold passed to the engine",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.35,
        help="Ignore recommendations below this FinalScore when computing metrics",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output", default=None,
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
        min_engine_confidence=args.confidence,
        score_threshold=args.min_score,
        seed=args.seed,
    )

    print(f"\nRunning backtest...")
    results = runner.run_batch(n_users=args.n)

    runner.print_report(results)

    if args.output and not results.empty:
        out_path = Path(args.output)
        results.to_csv(out_path, index=False)
        print(f"Per-user results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
