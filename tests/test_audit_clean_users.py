from pathlib import Path
import pandas as pd

from DataLayer.audit_clean_users import audit_clean_users


def test_audit_clean_users_runs(tmp_path: Path, capsys):
    p = tmp_path / "clean_users.parquet"
    df = pd.DataFrame(
        [
            {"SamAccountName": "u1", "Title": "T1", "Department": "D1", "GroupsList": ["G1", "G2"]},
            {"SamAccountName": "u2", "Title": "T2", "Department": "D2", "GroupsList": []},
        ]
    )
    df.to_parquet(p, index=False)

    audit_clean_users(p)
    out = capsys.readouterr().out
    assert "Total users: 2" in out
    assert "Zero-group users: 1" in out

