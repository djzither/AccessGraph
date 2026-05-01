from pathlib import Path
import pandas as pd


AD_RIGHTS_PATH = Path(r"C:\Users\djzit\Downloads\ce_ad_user_rights_all.xlsx")
FULL_TIME_ACCESS_PATH = Path(r"C:\Users\djzit\Downloads\full_time_employee_access_list.xlsx")


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).lower().strip()


def split_groups(group_string):
    if pd.isna(group_string):
        return []

    return [
        g.strip()
        for g in str(group_string).split(";")
        if g.strip()
        and "Cannot find an object with identity" not in g
    ]


def classify_score(score):
    if score == 1.0:
        return "Auto Assign"
    if score >= 0.8:
        return "Strong Recommend"
    if score >= 0.6:
        return "Suggest"
    if score >= 0.4:
        return "Low Confidence"
    return "Ignore"


def load_ad_rights(path):
    df = pd.read_excel(path)

    df["TitleClean"] = df["Title"].apply(clean_text)
    df["DepartmentClean"] = df["Department"].apply(clean_text)
    df["GroupsList"] = df["Groups"].apply(split_groups)

    return df


def recommend_for_new_employee(ad_df, new_employee):
    title_clean = clean_text(new_employee["Title"])
    department_clean = clean_text(new_employee["Department"])

    similar_users = ad_df[
        (ad_df["TitleClean"] == title_clean)
        & (ad_df["DepartmentClean"] == department_clean)
    ].copy()

    if similar_users.empty:
        raise ValueError("No similar users found for this title and department.")

    total_users = similar_users["SamAccountName"].nunique()

    exploded = similar_users.explode("GroupsList")
    exploded = exploded[exploded["GroupsList"].notna()]
    exploded = exploded[exploded["GroupsList"] != ""]

    recommendations = (
        exploded.groupby("GroupsList")
        .agg(
            UserCountWithGroup=("SamAccountName", "nunique"),
            UsersWithAccess=(
                "DisplayName",
                lambda names: ", ".join(sorted(set(names.astype(str)))),
            ),
        )
        .reset_index()
    )

    recommendations["TotalUsersInRole"] = total_users
    recommendations["Score"] = (
        recommendations["UserCountWithGroup"] / recommendations["TotalUsersInRole"]
    )

    recommendations["Decision"] = recommendations["Score"].apply(classify_score)

    recommendations = recommendations.sort_values(
        by=["Score", "UserCountWithGroup", "GroupsList"],
        ascending=[False, False, True],
    )

    return similar_users, recommendations


def main():
    ad_df = load_ad_rights(AD_RIGHTS_PATH)

    new_employee = {
        "DisplayName": "Derek2",
        "Title": "Computing Specialist",
        "Department": "CE IT Helpdesk",
    }

    similar_users, recommendations = recommend_for_new_employee(
        ad_df,
        new_employee,
    )

    print("\nNew employee:")
    print(pd.DataFrame([new_employee]).to_string(index=False))

    print("\nUsers being compared:")
    print(
        similar_users[
            ["SamAccountName", "DisplayName", "Title", "Department"]
        ].to_string(index=False)
    )

    print("\nRecommended rights:")
    print(
        recommendations[
            [
                "GroupsList",
                "UserCountWithGroup",
                "TotalUsersInRole",
                "Score",
                "Decision",
                "UsersWithAccess",
            ]
        ].to_string(index=False)
    )

    output_path = Path("data/output/jane_doe_recommendations.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recommendations.to_csv(output_path, index=False)

    print(f"\nSaved recommendations to: {output_path}")


if __name__ == "__main__":
    main()