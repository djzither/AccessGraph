import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityModel:
    def fit(self, users_df: pd.DataFrame):
        self.users_df = users_df.copy().reset_index(drop=True)

        all_users = self.users_df["SamAccountName"].astype(str).unique()

        exploded = (
            self.users_df[["SamAccountName", "GroupsList"]]
            .explode("GroupsList")
            .dropna(subset=["GroupsList"])
            .reset_index(drop=True)
        )

        exploded["GroupsList"] = exploded["GroupsList"].astype(str).str.strip()
        exploded = exploded[exploded["GroupsList"] != ""]

        if exploded.empty:
            self.matrix = pd.DataFrame(index=all_users)
            return self

        self.matrix = (
            pd.crosstab(
                index=exploded["SamAccountName"],
                columns=exploded["GroupsList"],
            )
            .astype(int)
        )

        # Important: keep users with zero groups
        self.matrix = self.matrix.reindex(all_users, fill_value=0)

        return self

    def similar_users(self, sam_account_name: str, top_n: int = 5) -> pd.DataFrame:
        sam_account_name = str(sam_account_name)

        if sam_account_name not in self.matrix.index:
            raise ValueError(f"{sam_account_name} not found in comparison pool")

        if self.matrix.shape[1] == 0:
            return pd.DataFrame(columns=["SamAccountName", "MLSimilarityScore"])

        target_vector = self.matrix.loc[[sam_account_name]]
        scores = cosine_similarity(target_vector, self.matrix)[0]

        results = pd.DataFrame({
            "SamAccountName": self.matrix.index,
            "MLSimilarityScore": scores,
        })

        results = results[results["SamAccountName"] != sam_account_name]

        return results.sort_values(
            "MLSimilarityScore",
            ascending=False,
        ).head(top_n)