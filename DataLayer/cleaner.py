from pathlib import Path
import re
import shutil
import pandas as pd


class DataCleaner:
    AD_GROUP_PREFIX_PATTERN = re.compile(r"^[A-Za-z]\.")

    def __init__(self, processed_path: str = "data/processed/clean_users.parquet"):
        self.processed_path = Path(processed_path)
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)

    def clean_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def process_groups(group_str):
            if pd.isna(group_str):
                return []

            groups = [g.strip() for g in str(group_str).split(";")]

            cleaned = [
                g for g in groups
                if g
                and not g.startswith("Cannot find an object")
                and "Cannot find an object with identity" not in g
                and not self.AD_GROUP_PREFIX_PATTERN.match(g)
            ]

            return cleaned

        df["GroupsList"] = df["Groups"].apply(process_groups)
        df["CleanGroupCount"] = df["GroupsList"].apply(len)

        return df

    def save_cleaned(self, df: pd.DataFrame) -> None:
        if self.processed_path.exists():
            if self.processed_path.is_dir():
                shutil.rmtree(self.processed_path)
            else:
                self.processed_path.unlink()

        df.to_parquet(self.processed_path, index=False)

    def load_cleaned(self) -> pd.DataFrame:
        if not self.processed_path.exists():
            raise FileNotFoundError(f"No cleaned file found at: {self.processed_path}")

        return pd.read_parquet(self.processed_path)

    def clean_and_save(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = self.clean_groups(df)
        self.save_cleaned(cleaned_df)
        return cleaned_df
