from pathlib import Path
import pandas as pd

class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise ValueError(f"Base Path Doesn't Exist: {self.base_path}")

    def _get_path(self, file_name: str) -> Path:
        path = self.base_path / file_name

        if not path.exists():
            raise ValueError(f"Path Doesn't Exist: {path}")
        return path

    def load_file(self, file_name: str, sheet_name=0) -> pd.DataFrame:
        path = self._get_path(file_name)
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path, sheet_name=sheet_name)

        raise ValueError(f"unsupported file format: {file_name}")




