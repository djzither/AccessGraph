from pathlib import Path
import pandas as pd

from DeterministicLayer.rules_recommender import RulesRecommender
from MLLayer.recommender import MLRecommender


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_users.parquet"

def classify_review_reason(group_name: str) -> str:
    g = str(group_name).lower()

    if "region" in g or "fsy" in g:
        return "Review: region/location-specific"
    if "storage" in g or "room" in g:
        return "Review: physical/location access"
    if "software" in g:
        return "Likely software access"
    if ".administrators" in g or "admin" in g:
        return "Review: admin-level access"

    return "Normal review"

def add_user_evidence_columns(
    final_results: pd.DataFrame,
    users: pd.DataFrame,
    target_user: str,
) -> pd.DataFrame:
    final_results = final_results.copy()

    users_by_id = users.set_index("SamAccountName")

    if target_user not in users_by_id.index:
        return final_results

    for idx, row in final_results.iterrows():
        group = row["GroupName"]

        nearest_users = str(row.get("NearestUsers", "")).split(",")
        nearest_users = [u.strip() for u in nearest_users if u.strip()]

        holders = []

        for user in nearest_users:
            if user not in users_by_id.index:
                final_results.loc[idx, f"Has_{user}"] = False
                continue

            rights = set(users_by_id.loc[user, "GroupsList"])
            has_group = group in rights

            final_results.loc[idx, f"Has_{user}"] = has_group

            if has_group:
                holders.append(user)

        target_rights = set(users_by_id.loc[target_user, "GroupsList"])

        final_results.loc[idx, "TargetAlreadyHas"] = group in target_rights
        final_results.loc[idx, "UsersWithThisRight"] = ", ".join(holders)

    return final_results


def add_supervisor_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["IsSupervisor"] = df["Title"].astype(str).str.contains(
        "supervisor|manager|director|lead|head|chief",
        case=False,
        na=False,
    )

    return df


def combine_recommendations(
    deterministic_results: pd.DataFrame,
    ml_results: pd.DataFrame,
) -> pd.DataFrame:

    deterministic_results = deterministic_results.copy()
    ml_results = ml_results.copy()

    if deterministic_results.empty and ml_results.empty:
        return pd.DataFrame()

    if deterministic_results.empty:
        ml_results["Confidence"] = 0
        ml_results["FinalScore"] = ml_results["MLConfidence"]
        ml_results["FinalSource"] = "ML only"
        return ml_results.sort_values("FinalScore", ascending=False)

    if ml_results.empty:
        deterministic_results["MLConfidence"] = 0
        deterministic_results["FinalScore"] = deterministic_results["Confidence"]
        deterministic_results["FinalSource"] = "Rules only"
        return deterministic_results.sort_values("FinalScore", ascending=False)

    combined = deterministic_results.merge(
        ml_results,
        on="GroupName",
        how="outer",
        suffixes=("_Rules", "_ML"),
    )

    combined["Confidence"] = combined["Confidence"].fillna(0)
    combined["MLConfidence"] = combined["MLConfidence"].fillna(0)

    combined["FinalScore"] = combined.apply(
        lambda row: (
            0.65 * row["Confidence"] + 0.35 * row["MLConfidence"]
            if row["Confidence"] > 0 and row["MLConfidence"] > 0
            else row["Confidence"]
            if row["Confidence"] > 0
            else row["MLConfidence"]
        ),
        axis=1,
    )
    combined["ScoreBreakdown"] = (
            "Rules: " + combined["Confidence"].round(2).astype(str)
            + " | ML: " + combined["MLConfidence"].round(2).astype(str)
    )

    combined["FinalSource"] = combined.apply(
        lambda row: (
            "Rules + ML"
            if row["Confidence"] > 0 and row["MLConfidence"] > 0
            else "Rules only"
            if row["Confidence"] > 0
            else "ML only"
        ),
        axis=1,
    )

    return combined.sort_values(
        ["FinalScore", "Confidence", "MLConfidence"],
        ascending=False,
    )

def get_ml_results_with_fallback(
    users: pd.DataFrame,
    sam_account_name: str,
    title: str,
    department: str,
) -> pd.DataFrame:

    ml = MLRecommender(users)

    attempts = [
        {
            "name": "Exact department",
            "department": department,
            "top_n_users": 5,
            "min_support": 1,
        },
        {
            "name": "Broader CE pool",
            "department": "CE",
            "top_n_users": 10,
            "min_support": 2,
        },
    ]

    for attempt in attempts:
        print(f"\nTrying ML pool: {attempt['name']}")

        try:
            results = ml.recommend_for_user(
                sam_account_name=sam_account_name,
                department=attempt["department"],
                top_n_users=attempt["top_n_users"],
                min_support=attempt["min_support"],
                include_supervisors=False,
            )

            if not results.empty:
                print(f"✔ Found results using: {attempt['name']}")
                results["MLPoolUsed"] = attempt["name"]
                return results

        except Exception as e:
            print(f"ML attempt failed for {attempt['name']}: {e}")

    print("❌ No ML results found in any pool")
    return pd.DataFrame()

def assign_final_decision(row):
    score = row.get("FinalScore", 0)
    ml_conf = row.get("MLConfidence", 0)
    group = str(row["GroupName"]).lower()

    if "admin" in group or "administrator" in group:
        return "REVIEW (admin access)"

    if score >= 0.8 and ml_conf >= 0.6:
        return "AUTO APPROVE"

    if score >= 0.5:
        return "REVIEW"

    return "LOW CONFIDENCE"


def main():
    print("Looking for file at:", DATA_PATH)

    if not DATA_PATH.exists():
        print("Could not find clean_users.parquet.")
        return

    users = pd.read_parquet(DATA_PATH)
    users = add_supervisor_flag(users)

    print(f"Loaded users: {len(users)}")

    sam_account_name = input("Enter SamAccountName: ").strip()

    target_user = users[users["SamAccountName"] == sam_account_name]

    if target_user.empty:
        print(f"User {sam_account_name} not found.")
        return

    # 🔥 AUTO PULL ROLE INFO
    title = str(target_user.iloc[0]["Title"])
    department = str(target_user.iloc[0]["Department"])

    same_department = users[
        users["Department"].astype(str).str.lower().str.strip()
        == department.lower().strip()
        ]

    same_role = same_department[
        same_department["Title"].astype(str).str.lower().str.strip()
        == title.lower().strip()
        ]

    print("\nPool debug:")
    print(f"Same department users: {len(same_department)}")
    print(f"Same title + department users: {len(same_role)}")

    print("\nDetected user info:")
    print(f"Title: {title}")
    print(f"Department: {department}")

    target_rights = set(target_user.iloc[0]["GroupsList"])

    # -----------------------------
    # Deterministic
    # -----------------------------
    rules = RulesRecommender(min_confidence=0.6)

    deterministic_results = rules.recommend_for_new_user(
        users_df=users,
        title=title,
        department=department,
    )

    if not deterministic_results.empty:
        deterministic_results = deterministic_results[
            ~deterministic_results["GroupName"].isin(target_rights)
        ]

    # -----------------------------
    # ML
    # -----------------------------
    ml_results = get_ml_results_with_fallback(
        users=users,
        sam_account_name=sam_account_name,
        title=title,
        department=department,
    )

    # -----------------------------
    # ML Noise Filtering (VERY IMPORTANT)
    # -----------------------------
    NOISE_PATTERNS = [
        "student",
        "sophomore",
        "freshman",
        "junior",
        "senior",
        "spouse",
        "post baccalaureate",
        "non degree",
        "registered_student",
        "slca-std",
        "deprovision",
        "provisioned",
        "o365elig",
        "zoom_access_student",
        "aerstd",
        "test",
    ]
    if not ml_results.empty:
        ml_results = ml_results[
            ~ml_results["GroupName"].astype(str).str.lower().apply(
                lambda g: any(pattern in g for pattern in NOISE_PATTERNS)
            )
        ]

    # -----------------------------
    # Combine
    # -----------------------------
    final_results = combine_recommendations(
        deterministic_results=deterministic_results,
        ml_results=ml_results,
    )

    final_results = add_user_evidence_columns(
        final_results=final_results,
        users=users,
        target_user=sam_account_name,
    )

    # -----------------------------
    # Add review classification
    # -----------------------------
    if not final_results.empty:
        final_results["ReviewReason"] = final_results["GroupName"].apply(
            classify_review_reason
        )

    final_results["FinalDecision"] = final_results.apply(
        assign_final_decision,
        axis=1,
    )

    if final_results.empty:
        print("\nNo recommendations found.")
        return

    columns_to_show = [
        col for col in [
            "GroupName",
            "FinalScore",
            "FinalSource",
            "ScoreBreakdown",
            "Confidence",
            "MLConfidence",
            "RiskLevel",
            "Decision",
            "ReviewReason",  # ← ADD THIS
            "UserCountWithGroup",
            "TotalUsersInRole",
            "MLSupportCount",
            "TargetAlreadyHas",
            "UsersWithThisRight",
            "MLComparedUsers",
            "NearestUsers",
            "FinalDecision"
        ]
        if col in final_results.columns
    ]

    print("\nFinal combined recommendations:")
    print(final_results[columns_to_show].round(3).head(30))


if __name__ == "__main__":
    main()
