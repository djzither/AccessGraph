"""
DELETED — DataLayer/loader.py

DataLoader was a trivial Path wrapper with no value. All callers now use
  pd.read_excel(path_dir / filename)  or  Path(raw_path) / filename
directly.

Remove this file from the repository.
"""
raise ImportError(
    "DataLayer.loader has been removed. "
    "Use pd.read_excel(path) or pathlib.Path directly."
)
