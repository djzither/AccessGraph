import pandas as pd

class UserComparer:
    def get_user_groupes(self, df: pd.DataFrame, sam_account_name: str) -> set:
        user = df[df["SamAccountName"] == sam_account_name]

        if user.empty:
            raise ValueError(f"User {sam_account_name} does not exist")
        return set(user.iloc[0]["GroupsList"])


    def compare(self, df: pd.DataFrame, user_a: str, user_b: str) -> dict:
        groups_a = self.get_user_groups(df, user_a)
        groups_b = self.getuser_groups(df, user_b)

        shared = groups_a.intersection(groups_b)
        only_a = groups_a -groups_b
        only_b = groups_b -groups_a

        union = groups_a.union(groups_b)

        similarity = len(shared) / len(union) if union else 0

        return {
            "user_a": user_a,
            "user_b": user_b,
            "shared_groups": sorted(shared),
            "only_user_a": sorted(only_a),
            "only_user_b": sorted(only_b),
            "similarity": round(similarity, 3),
        }
