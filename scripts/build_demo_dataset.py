"""
scripts/build_demo_dataset.py
─────────────────────────────────────────────────────────────────────────────
Build a fully sanitized demo dataset from the real processed parquets.

Reads (real, sensitive):
    data/processed/clean_users.parquet
    data/processed/access_reference.parquet

Writes (sanitized, safe to ship as demo):
    data/demo_processed/sanitized_clean_users.parquet
    data/demo_processed/sanitized_access_reference.parquet

Writes private mapping CSVs (PII — must NOT be checked in; .gitignored):
    private/demo_mapping/user_map.csv
    private/demo_mapping/person_name_map.csv
    private/demo_mapping/title_map.csv
    private/demo_mapping/department_map.csv
    private/demo_mapping/permission_map.csv

Design constraints (preserve recommendation behavior end-to-end):

* Deterministic relabeling only: same real value → same fake value everywhere,
  across runs and across the two parquets.
* Row counts, per-user permission set sizes, and joinable couplings are
  preserved exactly. We do not add, drop, shuffle, or reassign rows.
* Joinable couplings preserved:
    - clean_users.GroupsList ↔ access_reference.AccessName  (canonical key)
    - clean_users.Title       ↔ access_reference.JobTitle    (normalized text)
    - clean_users.Department  ↔ access_reference.Department  (normalized text)
    - clean_users.Manager     ↔ access_reference.Supervisor  (person name)
    - access_reference.ReferenceEmployeeName ↔ DisplayName  (person name)
* Functional values preserved literally:
    - "a.FULL TIME STAFF"   (workforce_type.FULL_TIME_STAFF_AD_GROUP)
    - EmployeeType / EmployeeTypeCanonical
    - SourceFile (informational)
    - AccessCategory (drives door-access filter via "HCEB Doors")
* Behavior-preserving substring guarantees:
    - Sensitive keywords (admin/payroll/finance/hr/superuser/domain/security/
      privileged/owner) embedded in fake permission names so PermissionFilter
      RiskLevel still triggers the same way.
    - "hceb " / "hcen " door prefixes preserved on permissions so
      PermissionFilter.is_door_access still drops the same rows.
    - Variant prefixes (m./i./dce./dce-/dce ) preserved on permissions so
      AccessRecommendationEngine.choose_group_name picks the same labels.
    - "fsy" substring preserved in fake titles/departments so
      AccessRecommendationEngine._is_fsy_role still triggers.
    - The four EXCLUDED_FSY_TITLES (used by the Streamlit app to filter rows)
      are preserved literally so that exclusion logic still works in demo mode.
* Forbidden output: never produces "crm" or "salesforce" substrings.

Usage:
    python -m scripts.build_demo_dataset
    python -m scripts.build_demo_dataset --salt my-custom-salt
    python -m scripts.build_demo_dataset --users-in ... --ref-in ... \
        --users-out ... --ref-out ... --mapping-dir ...
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Make project imports work when invoked as a script (python scripts/build_demo_dataset.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataLayer.permission_normalization import (  # noqa: E402
    normalize_groups_input,
    normalize_single_permission,
)


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_USERS_IN = Path("data/processed/clean_users.parquet")
DEFAULT_REF_IN = Path("data/processed/access_reference.parquet")
DEFAULT_USERS_OUT = Path("data/demo_processed/sanitized_clean_users.parquet")
DEFAULT_REF_OUT = Path("data/demo_processed/sanitized_access_reference.parquet")
DEFAULT_MAPPING_DIR = Path("private/demo_mapping")
DEFAULT_SALT = "accessgraph-demo-2026"

# Optional NetIDs to record in a **private** CSV only (`private/demo_mapping/` is gitignored).
# Never written under `data/demo_processed/`.
DEMO_LOOKUP_NETIDS: tuple[str, ...] = ("djzither", "ag877")


# ─── Behavior-preserving constants (kept in sync with engine internals) ──────

# Literal group string driving workforce classification (see DataLayer/workforce_type.py).
PRESERVED_GROUP_LITERALS = {"a.FULL TIME STAFF"}

# Titles the Streamlit app filters out via exact normalized match (ProductLayer/app.py).
# We keep them literal in the demo so the same users get filtered.
PRESERVED_TITLE_LITERALS_NORMALIZED = {
    "ce fsy us coordinator",
    "ce fsy us assistant coordinator",
    "ce fsy us wellness coordinator",
    "ce fsy us counselor",
}

# Sensitive keyword vocabulary from DeterministicLayer/permission_filter.py — embedding
# any present keyword in the fake string keeps RiskLevel and Manual-Review behavior intact.
SENSITIVE_KEYWORDS = (
    "admin", "payroll", "finance", "hr", "superuser",
    "domain", "security", "privileged", "owner",
)

# Door-access prefixes from DeterministicLayer/permission_filter.py:is_door_access.
DOOR_GROUP_PREFIXES = ("hceb ", "hcen ")

# Prefixes AccessRecommendationEngine._normalize_group_name strips when matching AD vs reference.
VARIANT_PREFIXES = ("m.", "i.", "dce.", "dce-", "dce ")

# Substrings that must never appear in sanitized output.
# - "crm" / "salesforce": access_exclusions drops these entirely from real
#   data; this is a defensive guard against ever generating one.
# - "byu": catches any residual leak of real BYU usernames or domain strings
#   (e.g. "[email protected]", "DC=BYU,DC=EDU", "BYUFaculty"). Safe to add
#   because all generated names use hex digests (0-9, a-f) plus fixed stems
#   like "perm_/u_/Role/Dept/Demo User/demo.local", none of which contain "byu".
FORBIDDEN_SUBSTRINGS = ("crm", "salesforce", "byu")

# Synthetic identity fields — derived deterministically from the fake SAM.
# DO NOT use a real organizational domain here.
DEMO_UPN_DOMAIN = "demo.local"
DEMO_DN_SUFFIX = "OU=People,DC=demo,DC=local"

# Substring AccessRecommendationEngine._is_fsy_role looks for in title/department.
PRESERVED_ROLE_SUBSTRINGS = ("fsy",)

# Categories that drive functional behavior in the engine — preserve literally.
PRESERVED_CATEGORY_LITERALS_LOWER = {"hceb doors"}


# ─── Sanitizer ───────────────────────────────────────────────────────────────


class Sanitizer:
    """Deterministic, idempotent pseudonymizer for AccessGraph demo data."""

    def __init__(self, salt: str = DEFAULT_SALT):
        self.salt = salt
        # Per-key memoization (drives the mapping CSV exports).
        self.user_id_map: dict[str, str] = {}             # raw_lower -> fake
        self.user_id_originals: dict[str, str] = {}       # raw_lower -> original (for CSV)
        self.person_name_map: dict[str, str] = {}         # raw_lower -> fake
        self.person_name_originals: dict[str, str] = {}
        self.title_map: dict[str, str] = {}               # normalized -> fake
        self.department_map: dict[str, str] = {}          # normalized -> fake
        self.permission_canonical_map: dict[str, str] = {}  # canonical_key -> fake_base
        self.permission_raw_map: dict[str, str] = {}      # raw -> fake (full, with prefixes)
        self.category_map: dict[str, str] = {}            # raw -> fake (mostly identity)

    # ── Hash helpers ─────────────────────────────────────────────────────────

    def _hash_key(self, key: str, length: int = 6) -> str:
        digest = hashlib.sha256(f"{self.salt}|{key}".encode("utf-8")).hexdigest()
        return digest[:length]

    @staticmethod
    def _scrub_forbidden(text: str) -> str:
        """Defensive guard: ensure forbidden substrings never reach the output."""
        out = text
        lowered = out.lower()
        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden in lowered:
                pattern = re.compile(re.escape(forbidden), re.IGNORECASE)
                out = pattern.sub("xxx", out)
                lowered = out.lower()
        return out

    # ── Person / user identifiers ────────────────────────────────────────────

    def fake_user_id(self, raw_sam: object) -> str:
        if raw_sam is None:
            return ""
        text = str(raw_sam).strip()
        if not text or text.lower() == "nan":
            return ""
        key = text.lower()
        if key in self.user_id_map:
            return self.user_id_map[key]
        fake = "u_" + self._hash_key(f"user|{key}", 8)
        fake = self._scrub_forbidden(fake)
        self.user_id_map[key] = fake
        self.user_id_originals[key] = text
        return fake

    def fake_person_name(self, raw_name: object) -> str:
        if raw_name is None:
            return ""
        text = str(raw_name).strip()
        if not text or text.lower() == "nan":
            return ""
        key = text.lower()
        if key in self.person_name_map:
            return self.person_name_map[key]
        fake = "Demo User " + self._hash_key(f"person|{key}", 6).upper()
        fake = self._scrub_forbidden(fake)
        self.person_name_map[key] = fake
        self.person_name_originals[key] = text
        return fake

    def fake_name(self, raw_name: object) -> str:
        """
        Sanitize the AD ``Name`` attribute. AD's Name is usually the same as
        DisplayName (the CN) but in some exports it can be the SamAccountName.
        We try both existing maps before falling back to fake_person_name so
        that "Name == DisplayName" yields the same fake as DisplayName, and
        "Name == SamAccountName" yields the same fake as SamAccountName —
        preserving referential consistency.
        """
        if raw_name is None:
            return ""
        text = str(raw_name).strip()
        if not text or text.lower() == "nan":
            return ""
        key = text.lower()
        if key in self.user_id_map:
            return self.user_id_map[key]
        if key in self.person_name_map:
            return self.person_name_map[key]
        return self.fake_person_name(raw_name)

    @staticmethod
    def _build_demo_upn(fake_local_part: str) -> str:
        """Build a synthetic UPN: ``<fake_sam>@demo.local``."""
        if not fake_local_part:
            return ""
        return f"{fake_local_part}@{DEMO_UPN_DOMAIN}"

    @staticmethod
    def _build_demo_dn(fake_cn: str) -> str:
        """Build a synthetic DN: ``CN=<fake_sam>,OU=People,DC=demo,DC=local``."""
        if not fake_cn:
            return ""
        return f"CN={fake_cn},{DEMO_DN_SUFFIX}"

    def fake_upn_from_raw(self, raw_upn: object) -> str:
        """
        Fallback used only when SamAccountName is not available on the row.
        Strips the local part of a real UPN (e.g. ``[email protected]``) and
        looks it up in the user_id_map; if absent, registers a fresh fake.
        Always emits a synthetic UPN — the real domain is never preserved.
        """
        if raw_upn is None:
            return ""
        text = str(raw_upn).strip()
        if not text or text.lower() == "nan":
            return ""
        local_part = text.split("@", 1)[0]
        return self._build_demo_upn(self.fake_user_id(local_part))

    def fake_dn_from_raw(self, raw_dn: object) -> str:
        """
        Fallback used only when SamAccountName is not available on the row.
        Extracts the first ``CN=...`` component and remaps it via the user or
        person map (else generates a fresh deterministic fake from the DN
        hash). The real OU/DC suffix is always discarded.
        """
        if raw_dn is None:
            return ""
        text = str(raw_dn).strip()
        if not text or text.lower() == "nan":
            return ""
        match = re.match(r"\s*CN\s*=\s*([^,]+)", text, flags=re.IGNORECASE)
        if not match:
            fake = "u_" + self._hash_key(f"dn|{text.lower()}", 8)
        else:
            cn_value = match.group(1).strip()
            cn_key = cn_value.lower()
            if cn_key in self.user_id_map:
                fake = self.user_id_map[cn_key]
            elif cn_key in self.person_name_map:
                fake = self.person_name_map[cn_key]
            else:
                fake = "u_" + self._hash_key(f"dn-cn|{cn_key}", 8)
        return self._build_demo_dn(fake)

    # ── Role text (Title / Department) ───────────────────────────────────────

    @staticmethod
    def _normalize_role_text(value: object) -> str:
        """Mirrors AccessRecommendationEngine._normalize_role_text exactly."""
        text = str(value).lower().strip()
        for old, new in [("&", " and "), (",", " "), ("/", " "), ("-", " ")]:
            text = text.replace(old, new)
        return " ".join(text.split())

    def _role_substring_suffix(self, normalized: str) -> str:
        markers = [sub.upper() for sub in PRESERVED_ROLE_SUBSTRINGS if sub in normalized]
        return (" " + " ".join(markers)) if markers else ""

    def fake_title(self, raw_title: object) -> str:
        if raw_title is None:
            return ""
        text = str(raw_title).strip()
        if not text or text.lower() == "nan":
            return ""
        normalized = self._normalize_role_text(text)
        if not normalized:
            return ""
        if normalized in PRESERVED_TITLE_LITERALS_NORMALIZED:
            self.title_map.setdefault(normalized, text)
            return text
        if normalized in self.title_map:
            return self.title_map[normalized]
        digest = self._hash_key(f"title|{normalized}", 6).upper()
        fake = self._scrub_forbidden(f"Role {digest}{self._role_substring_suffix(normalized)}")
        self.title_map[normalized] = fake
        return fake

    def fake_department(self, raw_dept: object) -> str:
        if raw_dept is None:
            return ""
        text = str(raw_dept).strip()
        if not text or text.lower() == "nan":
            return ""
        normalized = self._normalize_role_text(text)
        if not normalized:
            return ""
        if normalized in self.department_map:
            return self.department_map[normalized]
        digest = self._hash_key(f"dept|{normalized}", 6).upper()
        fake = self._scrub_forbidden(f"Dept {digest}{self._role_substring_suffix(normalized)}")
        self.department_map[normalized] = fake
        return fake

    # ── Permissions / groups ────────────────────────────────────────────────

    @staticmethod
    def _split_variant_prefix(text: str) -> tuple[str, str]:
        """Strip a variant prefix (m./i./dce./dce-/dce ) preserving original case."""
        lowered = text.lower()
        for prefix in VARIANT_PREFIXES:
            if lowered.startswith(prefix):
                return text[: len(prefix)], text[len(prefix):]
        return "", text

    @staticmethod
    def _split_door_prefix(text: str) -> tuple[str, str]:
        lowered = text.lower()
        for prefix in DOOR_GROUP_PREFIXES:
            if lowered.startswith(prefix):
                return text[: len(prefix)], text[len(prefix):]
        return "", text

    @staticmethod
    def _canonical_permission_key(text: object) -> str:
        """Mirrors AccessRecommendationEngine._normalize_group_name exactly."""
        base = normalize_single_permission(text)
        s = str(base).lower().strip() if base else ""
        for prefix in ("m.", "i.", "dce.", "dce-", "dce "):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        return re.sub(r"[\s._-]+", "", s)

    def fake_permission(self, raw: object) -> str:
        text = normalize_single_permission(raw)
        if text is None or not text:
            return ""
        if text in PRESERVED_GROUP_LITERALS:
            canonical = self._canonical_permission_key(text)
            self.permission_canonical_map.setdefault(canonical, text)
            self.permission_raw_map.setdefault(text, text)
            return text
        if text in self.permission_raw_map:
            return self.permission_raw_map[text]

        var_prefix, after_variant = self._split_variant_prefix(text)
        door_prefix, _body = self._split_door_prefix(after_variant)

        canonical_key = self._canonical_permission_key(text)
        if not canonical_key:
            return ""

        if canonical_key in self.permission_canonical_map:
            base_fake = self.permission_canonical_map[canonical_key]
        else:
            digest = self._hash_key(f"perm|{canonical_key}", 6)
            base_fake = f"perm_{digest}"
            lowered = text.lower()
            kw_present = [kw for kw in SENSITIVE_KEYWORDS if kw in lowered]
            if kw_present:
                base_fake += "_" + "_".join(kw_present)
            base_fake = self._scrub_forbidden(base_fake)
            self.permission_canonical_map[canonical_key] = base_fake

        fake_full = self._scrub_forbidden(f"{var_prefix}{door_prefix}{base_fake}")
        self.permission_raw_map[text] = fake_full
        return fake_full

    # ── Category (mostly preserved) ─────────────────────────────────────────

    def fake_category(self, raw_category: object) -> str | None:
        """
        Categories drive functional behavior in two places:
          * is_door_access checks for "hceb doors" substring
          * AccessCategory is also exported into recommendations for explainability
        We therefore preserve their literal strings — they are functional column
        names from the spreadsheets ("AD Rights", "HCEB Doors", "Cvent", etc.),
        not personal data, and changing them risks breaking downstream behavior.
        """
        if raw_category is None:
            return None
        text = str(raw_category).strip()
        if not text or text.lower() == "nan":
            return None
        self.category_map.setdefault(text, text)
        return text

    # ── DataFrame transforms ─────────────────────────────────────────────────

    def transform_users(self, users_in: pd.DataFrame) -> pd.DataFrame:
        df = users_in.copy()

        # Order matters: SamAccountName + DisplayName must be remapped first so
        # that derived identity columns (Name, UPN, DN) can reuse those maps
        # for referential consistency.
        if "SamAccountName" in df.columns:
            df["SamAccountName"] = df["SamAccountName"].apply(self.fake_user_id)
        if "DisplayName" in df.columns:
            df["DisplayName"] = df["DisplayName"].apply(self.fake_person_name)
        if "Manager" in df.columns:
            df["Manager"] = df["Manager"].apply(self.fake_person_name)
        if "Title" in df.columns:
            df["Title"] = df["Title"].apply(self.fake_title)
        if "Department" in df.columns:
            df["Department"] = df["Department"].apply(self.fake_department)

        # AD identity columns that previously leaked real netids / @byu.edu /
        # CN=...,OU=People,DC=byu,DC=edu strings. We rebuild them from the
        # already-transformed fake SamAccountName so the new values are
        # entirely synthetic and tie back to the same user.
        if "Name" in df.columns:
            df["Name"] = df["Name"].apply(self.fake_name)
        if "UserPrincipalName" in df.columns:
            if "SamAccountName" in df.columns:
                df["UserPrincipalName"] = df["SamAccountName"].apply(self._build_demo_upn)
            else:
                df["UserPrincipalName"] = df["UserPrincipalName"].apply(self.fake_upn_from_raw)
        if "DistinguishedName" in df.columns:
            if "SamAccountName" in df.columns:
                df["DistinguishedName"] = df["SamAccountName"].apply(self._build_demo_dn)
            else:
                df["DistinguishedName"] = df["DistinguishedName"].apply(self.fake_dn_from_raw)

        if "GroupsList" in df.columns:
            def _map_groups(value: object) -> list[str]:
                tokens = normalize_groups_input(value)
                out: list[str] = []
                for token in tokens:
                    fake = self.fake_permission(token)
                    if fake:
                        out.append(fake)
                return out
            df["GroupsList"] = df["GroupsList"].apply(_map_groups)
            if "CleanGroupCount" in df.columns:
                df["CleanGroupCount"] = df["GroupsList"].apply(len)

        # Drop the raw 'Groups' string column if it leaked through — it contains
        # the unfiltered original AD group dump and must not appear in demo data.
        for sensitive_col in ("Groups",):
            if sensitive_col in df.columns:
                df = df.drop(columns=[sensitive_col])

        return df

    def transform_reference(self, ref_in: pd.DataFrame) -> pd.DataFrame:
        df = ref_in.copy()

        if "JobTitle" in df.columns:
            df["JobTitle"] = df["JobTitle"].apply(self.fake_title)
        if "Department" in df.columns:
            df["Department"] = df["Department"].apply(self.fake_department)
        if "Supervisor" in df.columns:
            df["Supervisor"] = df["Supervisor"].apply(self.fake_person_name)
        if "ReferenceEmployeeName" in df.columns:
            df["ReferenceEmployeeName"] = df["ReferenceEmployeeName"].apply(self.fake_person_name)
        if "AccessCategory" in df.columns:
            df["AccessCategory"] = df["AccessCategory"].apply(self.fake_category)
        if "AccessName" in df.columns:
            df["AccessName"] = df["AccessName"].apply(self.fake_permission)

        return df

    # ── Mapping CSV exports ──────────────────────────────────────────────────

    def export_mappings(self, mapping_dir: Path) -> dict[str, Path]:
        mapping_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}

        outputs["user_map"] = self._write_csv(
            mapping_dir / "user_map.csv",
            [
                {"real_sam_account": self.user_id_originals.get(k, k), "fake_sam_account": v}
                for k, v in sorted(self.user_id_map.items())
            ],
        )
        outputs["person_name_map"] = self._write_csv(
            mapping_dir / "person_name_map.csv",
            [
                {"real_person_name": self.person_name_originals.get(k, k), "fake_person_name": v}
                for k, v in sorted(self.person_name_map.items())
            ],
        )
        outputs["title_map"] = self._write_csv(
            mapping_dir / "title_map.csv",
            [
                {"real_title_normalized": k, "fake_title": v}
                for k, v in sorted(self.title_map.items())
            ],
        )
        outputs["department_map"] = self._write_csv(
            mapping_dir / "department_map.csv",
            [
                {"real_department_normalized": k, "fake_department": v}
                for k, v in sorted(self.department_map.items())
            ],
        )
        outputs["permission_map"] = self._write_csv(
            mapping_dir / "permission_map.csv",
            [
                {"real_permission_raw": k, "fake_permission": v}
                for k, v in sorted(self.permission_raw_map.items())
            ],
        )
        return outputs

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> Path:
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        return path


# ─── Private demo lookup (real NetIDs — never under demo_processed) ───────────


def emit_demo_lookup_examples(
    sanitizer: Sanitizer,
    users_in: pd.DataFrame,
    users_out: pd.DataFrame,
    mapping_dir: Path,
    *,
    netids: tuple[str, ...] = DEMO_LOOKUP_NETIDS,
) -> Path | None:
    """
    Write ``demo_lookup_examples.csv`` under ``mapping_dir`` (gitignored) and print a summary.

    Lets you resolve sanitized SamAccountName / Title / Department for specific real NetIDs
    after a build. Real identifiers stay only in ``private/demo_mapping/``.
    """
    mapping_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for nid in netids:
        key = nid.strip().lower()
        if "SamAccountName" not in users_in.columns:
            fake_sam = sanitizer.fake_user_id(nid)
            rows.append(
                {
                    "lookup_netid": nid,
                    "found_in_clean_users_parquet": False,
                    "sanitized_netid": fake_sam,
                    "real_title": "",
                    "sanitized_title": "",
                    "real_department": "",
                    "sanitized_department": "",
                    "note": "SamAccountName column missing from input",
                }
            )
            continue

        mask = users_in["SamAccountName"].astype(str).str.strip().str.lower() == key
        if mask.any():
            idx = users_in.index[mask][0]
            rin = users_in.loc[idx]
            rout = users_out.loc[idx]
            rows.append(
                {
                    "lookup_netid": nid,
                    "found_in_clean_users_parquet": True,
                    "sanitized_netid": str(rout.get("SamAccountName", "") or ""),
                    "real_title": str(rin.get("Title", "") or ""),
                    "sanitized_title": str(rout.get("Title", "") or ""),
                    "real_department": str(rin.get("Department", "") or ""),
                    "sanitized_department": str(rout.get("Department", "") or ""),
                    "note": "",
                }
            )
        else:
            fake_sam = sanitizer.fake_user_id(nid)
            rows.append(
                {
                    "lookup_netid": nid,
                    "found_in_clean_users_parquet": False,
                    "sanitized_netid": fake_sam,
                    "real_title": "",
                    "sanitized_title": "",
                    "real_department": "",
                    "sanitized_department": "",
                    "note": "NetID not present in source clean_users row set",
                }
            )

    out_path = mapping_dir / "demo_lookup_examples.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")

    _print_section("Demo identity lookup (private demo_lookup_examples.csv)")
    print(
        "  Real NetIDs below appear only under gitignored private/demo_mapping — "
        "not in demo_processed."
    )
    for r in rows:
        nid = r["lookup_netid"]
        sam_o = r["sanitized_netid"]
        print(f"  • {nid}")
        print(f"      sanitized NetID (SamAccountName): {sam_o}")
        if r["found_in_clean_users_parquet"]:
            print(f"      Title:     {r['real_title']!r} → {r['sanitized_title']!r}")
            print(f"      Department:{r['real_department']!r} → {r['sanitized_department']!r}")
        else:
            tail = str(r.get("note") or "").strip()
            if tail:
                print(f"      ({tail})")
    print(f"\n  Wrote: {out_path.resolve()}")

    return out_path


# ─── Validation helpers ──────────────────────────────────────────────────────


def _flatten_groups(values) -> list[str]:
    out: list[str] = []
    for v in values:
        out.extend(normalize_groups_input(v))
    return out


def _print_section(title: str) -> None:
    bar = "-" * max(8, min(78, len(title) + 2))
    print(f"\n{bar}\n {title}\n{bar}")


def _check_no_forbidden(label: str, values) -> int:
    hits = 0
    for v in values:
        if v is None:
            continue
        text = str(v).lower()
        if any(forbidden in text for forbidden in FORBIDDEN_SUBSTRINGS):
            hits += 1
    print(f"  forbidden-substring hits in {label}: {hits}")
    return hits


def _scan_columns_for_real_leakage(
    label: str,
    df: pd.DataFrame,
    columns: list[str],
    real_values: set[str],
) -> int:
    """Spot-check that no real-name / real-title strings leaked verbatim.

    Allowance: literally-preserved values (a.FULL TIME STAFF, the four FSY titles)
    are filtered out of the comparison set, so they are not counted as 'leaks'.
    """
    leaks = 0
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str)
        for val in series.unique():
            if val in real_values:
                leaks += 1
    print(f"  verbatim real-value hits in {label}: {leaks}")
    return leaks


def validate(
    *,
    users_in: pd.DataFrame,
    users_out: pd.DataFrame,
    ref_in: pd.DataFrame,
    ref_out: pd.DataFrame,
    sanitizer: "Sanitizer | None" = None,
) -> bool:
    ok = True

    _print_section("Row counts")
    print(f"  users:      input={len(users_in):>6} output={len(users_out):>6}")
    print(f"  reference:  input={len(ref_in):>6} output={len(ref_out):>6}")
    if len(users_in) != len(users_out):
        print("  FAIL: user row count changed.")
        ok = False
    if len(ref_in) != len(ref_out):
        print("  FAIL: reference row count changed.")
        ok = False

    _print_section("Schema audit")
    sanitized_user_cols = {
        "SamAccountName", "DisplayName", "Manager", "Title", "Department",
        "GroupsList", "CleanGroupCount",
        # AD identity columns rebuilt from the fake SamAccountName / person map.
        "Name", "UserPrincipalName", "DistinguishedName",
    }
    sanitized_ref_cols = {
        "JobTitle", "Department", "Supervisor", "ReferenceEmployeeName",
        "AccessCategory", "AccessName",
    }
    explicitly_dropped_user_cols = {"Groups"}
    user_cols_in = set(users_in.columns)
    user_cols_out = set(users_out.columns)
    ref_cols_in = set(ref_in.columns)
    ref_cols_out = set(ref_out.columns)
    user_passthrough = sorted(
        user_cols_out - sanitized_user_cols - explicitly_dropped_user_cols
    )
    ref_passthrough = sorted(ref_cols_out - sanitized_ref_cols)
    print(f"  users.columns_in:      {sorted(user_cols_in)}")
    print(f"  users.columns_out:     {sorted(user_cols_out)}")
    print(f"  users.passthrough:     {user_passthrough}")
    print(f"  reference.columns_in:  {sorted(ref_cols_in)}")
    print(f"  reference.columns_out: {sorted(ref_cols_out)}")
    print(f"  reference.passthrough: {ref_passthrough}")
    if user_passthrough or ref_passthrough:
        print(
            "  NOTE: Pass-through columns above were copied verbatim. Audit "
            "them and extend Sanitizer.transform_* if any contain sensitive "
            "data."
        )

    _print_section("Cardinality preservation")
    # SamAccountName / DisplayName / Manager / Supervisor / ReferenceEmployeeName
    # / AccessName / AccessCategory: should be 1:1 at the raw level (same number
    # of distinct values before and after).
    # Title / Department / JobTitle: keyed by normalized text on purpose, so the
    # AFTER count must equal the count of distinct *normalized* BEFORE values
    # (and never exceed BEFORE — that would mean fake collisions split reals).

    def _check_strict(label: str, before: int, after: int) -> None:
        nonlocal ok
        marker = "OK" if before == after else "FAIL"
        print(f"  {label}: unique before={before:>5} after={after:>5}  [{marker}]")
        if before != after:
            ok = False

    def _check_normalized(label: str, before_raw, after_raw, normalizer) -> None:
        """
        Compare distinct *normalized* values on BOTH sides. The sanitizer
        deliberately preserves variant prefixes (m./i./dce./dce-/dce ) and
        door prefixes (hceb /hcen ) on top of a canonical-key-derived base,
        so two distinct real raws sharing a canonical key produce two
        distinct fake raws sharing a canonical key. We must therefore
        normalize both sides — comparing canonical(before) to raw(after)
        would be asymmetric and falsely fail whenever prefix variants exist.
        """
        nonlocal ok
        before_norm = len({normalizer(v) for v in before_raw if v not in (None, "")})
        after_norm = len({normalizer(v) for v in after_raw if v not in (None, "")})
        marker = "OK" if before_norm == after_norm else "FAIL"
        print(
            f"  {label}: unique-normalized-before={before_norm:>5} "
            f"unique-normalized-after={after_norm:>5}  [{marker}]"
        )
        if before_norm != after_norm:
            ok = False

    if "SamAccountName" in users_in.columns and "SamAccountName" in users_out.columns:
        _check_strict("users.SamAccountName",
                      users_in["SamAccountName"].nunique(dropna=True),
                      users_out["SamAccountName"].nunique(dropna=True))
    if "DisplayName" in users_in.columns and "DisplayName" in users_out.columns:
        _check_normalized(
            "users.DisplayName",
            users_in["DisplayName"].dropna().astype(str).tolist(),
            users_out["DisplayName"].dropna().astype(str).tolist(),
            normalizer=lambda v: v.strip().lower(),
        )
    if "Manager" in users_in.columns and "Manager" in users_out.columns:
        _check_normalized(
            "users.Manager",
            users_in["Manager"].dropna().astype(str).tolist(),
            users_out["Manager"].dropna().astype(str).tolist(),
            normalizer=lambda v: v.strip().lower(),
        )
    if "Title" in users_in.columns and "Title" in users_out.columns:
        _check_normalized(
            "users.Title",
            users_in["Title"].dropna().astype(str).tolist(),
            users_out["Title"].dropna().astype(str).tolist(),
            normalizer=Sanitizer._normalize_role_text,
        )
    if "Department" in users_in.columns and "Department" in users_out.columns:
        _check_normalized(
            "users.Department",
            users_in["Department"].dropna().astype(str).tolist(),
            users_out["Department"].dropna().astype(str).tolist(),
            normalizer=Sanitizer._normalize_role_text,
        )

    for col, normalizer in [
        ("JobTitle", Sanitizer._normalize_role_text),
        ("Department", Sanitizer._normalize_role_text),
        ("Supervisor", lambda v: v.strip().lower()),
        ("ReferenceEmployeeName", lambda v: v.strip().lower()),
    ]:
        if col in ref_in.columns and col in ref_out.columns:
            _check_normalized(
                f"reference.{col}",
                ref_in[col].dropna().astype(str).tolist(),
                ref_out[col].dropna().astype(str).tolist(),
                normalizer=normalizer,
            )
    # AccessName: keyed by canonical permission key — two raw spellings that
    # the engine already treats as the same permission (e.g. "foo bar" /
    # "foo.bar") correctly map to the same fake. Validate that the count of
    # distinct *canonical keys* is preserved.
    if "AccessName" in ref_in.columns and "AccessName" in ref_out.columns:
        _check_normalized(
            "reference.AccessName",
            ref_in["AccessName"].dropna().astype(str).tolist(),
            ref_out["AccessName"].dropna().astype(str).tolist(),
            normalizer=Sanitizer._canonical_permission_key,
        )
    if "AccessCategory" in ref_in.columns and "AccessCategory" in ref_out.columns:
        # Categories are preserved literally — strict 1:1 expected.
        _check_strict(
            "reference.AccessCategory",
            ref_in["AccessCategory"].nunique(dropna=True),
            ref_out["AccessCategory"].nunique(dropna=True),
        )

    _print_section("Permission token preservation")
    if "GroupsList" in users_in.columns and "GroupsList" in users_out.columns:
        in_tokens = _flatten_groups(users_in["GroupsList"])
        out_tokens = _flatten_groups(users_out["GroupsList"])
        print(f"  total user permission tokens: input={len(in_tokens):>6} output={len(out_tokens):>6}")
        if len(in_tokens) != len(out_tokens):
            print("    FAIL: total user permission token count changed.")
            ok = False
        # Two raw tokens with the same canonical key collapse to one fake by
        # design (engine treats them as the same permission), so we compare the
        # count of distinct canonical keys, not raw strings.
        in_unique = len({Sanitizer._canonical_permission_key(t) for t in in_tokens if t})
        out_unique = len({Sanitizer._canonical_permission_key(t) for t in out_tokens if t})
        in_unique_raw = len(set(in_tokens))
        out_unique_raw = len(set(out_tokens))
        print(
            f"  unique user permissions (canonical): "
            f"input={in_unique:>6} output={out_unique:>6}  "
            f"(raw: input={in_unique_raw} output={out_unique_raw})"
        )
        if in_unique != out_unique:
            print("    FAIL: unique user permission count (canonical) changed.")
            ok = False

    _print_section("'a.FULL TIME STAFF' literal preservation")
    if "GroupsList" in users_in.columns:
        before_present = any(
            "a.FULL TIME STAFF" in normalize_groups_input(g)
            for g in users_in["GroupsList"]
        )
        after_present = any(
            "a.FULL TIME STAFF" in normalize_groups_input(g)
            for g in users_out["GroupsList"]
        )
        print(f"  present in users (input):  {before_present}")
        print(f"  present in users (output): {after_present}")
        if before_present != after_present:
            print("    FAIL: a.FULL TIME STAFF presence flipped.")
            ok = False

    _print_section("Canonical-key bijection (real -> fake)")
    # Every real canonical permission key must map to exactly one fake
    # canonical key. Multiple raw fake spellings per canonical are fine
    # (they preserve variant / door prefixes for choose_group_name and
    # is_door_access), but they MUST all canonicalize to the same fake key,
    # or downstream matching against reference AccessName breaks.
    #
    # We query the sanitizer's own internal raw→fake map rather than
    # positional row alignment — that map is the source of truth and is
    # immune to any token being dropped during transform.
    if sanitizer is None:
        print("  (skipped — sanitizer instance not passed to validate())")
    else:
        real_to_fake_canonicals: dict[str, set[str]] = {}
        for real_raw, fake_raw in sanitizer.permission_raw_map.items():
            rk = Sanitizer._canonical_permission_key(real_raw)
            if not rk:
                continue
            fk = Sanitizer._canonical_permission_key(fake_raw)
            real_to_fake_canonicals.setdefault(rk, set()).add(fk)
        split = {
            rk: sorted(fks)
            for rk, fks in real_to_fake_canonicals.items()
            if len(fks) > 1
        }
        print(
            f"  permission_raw_map: real_canonical_keys={len(real_to_fake_canonicals):>5} "
            f"split_into_multiple_fake_canonicals={len(split):>5}"
        )
        if split:
            sample = dict(list(split.items())[:5])
            print(
                "    FAIL: at least one real canonical permission key maps "
                "to multiple fake canonical keys."
            )
            print(f"    sample (real_canonical -> [fake_canonicals]): {sample}")
            ok = False
        else:
            print("    OK: every real canonical key maps to exactly one fake canonical key.")

    _print_section("GroupsList <-> AccessName overlap (canonical key)")
    if {"GroupsList"}.issubset(users_in.columns) and {"AccessName"}.issubset(ref_in.columns):
        def canonical_set(perms: list[str]) -> set[str]:
            return {Sanitizer._canonical_permission_key(p) for p in perms if p}
        users_in_keys = canonical_set(_flatten_groups(users_in["GroupsList"]))
        users_out_keys = canonical_set(_flatten_groups(users_out["GroupsList"]))
        ref_in_keys = canonical_set(ref_in["AccessName"].dropna().astype(str).tolist())
        ref_out_keys = canonical_set(ref_out["AccessName"].dropna().astype(str).tolist())
        in_overlap = len(users_in_keys & ref_in_keys)
        out_overlap = len(users_out_keys & ref_out_keys)
        print(f"  user∩reference canonical keys: input={in_overlap:>5} output={out_overlap:>5}")
        if in_overlap != out_overlap:
            print("    FAIL: GroupsList <-> AccessName overlap changed.")
            ok = False

    _print_section("Forbidden-substring scan (must all be 0)")
    # Scan EVERY column of both output frames — not a hardcoded list — so any
    # passthrough column that contains a residual real value (real netid,
    # @byu.edu, DC=BYU,...) gets caught regardless of column name.
    forbidden_hits = 0
    for col in users_out.columns:
        if col == "GroupsList":
            forbidden_hits += _check_no_forbidden(
                f"users.{col}", _flatten_groups(users_out[col])
            )
        else:
            forbidden_hits += _check_no_forbidden(
                f"users.{col}", users_out[col].dropna()
            )
    for col in ref_out.columns:
        forbidden_hits += _check_no_forbidden(
            f"reference.{col}", ref_out[col].dropna()
        )
    if forbidden_hits > 0:
        print("  FAIL: forbidden substrings present in output.")
        ok = False

    _print_section("Verbatim real-value leakage spot-check")
    real_user_names = set(
        v for v in users_in.get("DisplayName", pd.Series(dtype=str)).dropna().astype(str).tolist() if v
    )
    # Names from the AD ``Name`` attribute too — they may be display-name-like
    # OR sam-like, so we union them in.
    real_user_names |= set(
        v for v in users_in.get("Name", pd.Series(dtype=str)).dropna().astype(str).tolist() if v
    )
    real_sams = set(
        v for v in users_in.get("SamAccountName", pd.Series(dtype=str)).dropna().astype(str).tolist() if v
    )
    real_upns = set(
        v for v in users_in.get("UserPrincipalName", pd.Series(dtype=str)).dropna().astype(str).tolist() if v
    )
    real_dns = set(
        v for v in users_in.get("DistinguishedName", pd.Series(dtype=str)).dropna().astype(str).tolist() if v
    )
    real_titles = set(
        v for v in users_in.get("Title", pd.Series(dtype=str)).dropna().astype(str).tolist() if v
    )
    # Allow literally-preserved titles through.
    real_titles_for_check = {
        v for v in real_titles
        if Sanitizer._normalize_role_text(v) not in PRESERVED_TITLE_LITERALS_NORMALIZED
    }
    real_departments = set(
        v for v in users_in.get("Department", pd.Series(dtype=str)).dropna().astype(str).tolist() if v
    )
    real_permissions = set(_flatten_groups(users_in.get("GroupsList", pd.Series(dtype=object))))
    real_permissions_for_check = real_permissions - PRESERVED_GROUP_LITERALS

    leaks = 0
    leaks += _scan_columns_for_real_leakage(
        "users.DisplayName/Manager/Name",
        users_out,
        ["DisplayName", "Manager", "Name"],
        real_user_names,
    )
    leaks += _scan_columns_for_real_leakage(
        "users.SamAccountName", users_out, ["SamAccountName"], real_sams,
    )
    leaks += _scan_columns_for_real_leakage(
        "users.UserPrincipalName", users_out, ["UserPrincipalName"], real_upns,
    )
    leaks += _scan_columns_for_real_leakage(
        "users.DistinguishedName", users_out, ["DistinguishedName"], real_dns,
    )
    leaks += _scan_columns_for_real_leakage(
        "users.Title", users_out, ["Title"], real_titles_for_check,
    )
    leaks += _scan_columns_for_real_leakage(
        "users.Department", users_out, ["Department"], real_departments,
    )
    if "GroupsList" in users_out.columns:
        out_perms = set(_flatten_groups(users_out["GroupsList"]))
        verbatim_perm_leaks = len(out_perms & real_permissions_for_check)
        print(f"  verbatim real-permission hits in users.GroupsList: {verbatim_perm_leaks}")
        leaks += verbatim_perm_leaks
    if leaks > 0:
        print("  FAIL: verbatim real values present in sanitized output.")
        ok = False

    _print_section("RESULT")
    print("  PASS" if ok else "  FAIL")
    return ok


# ─── CLI / entry point ───────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users-in", default=str(DEFAULT_USERS_IN))
    parser.add_argument("--ref-in", default=str(DEFAULT_REF_IN))
    parser.add_argument("--users-out", default=str(DEFAULT_USERS_OUT))
    parser.add_argument("--ref-out", default=str(DEFAULT_REF_OUT))
    parser.add_argument("--mapping-dir", default=str(DEFAULT_MAPPING_DIR))
    parser.add_argument("--salt", default=DEFAULT_SALT,
                        help="Hash salt for deterministic mappings (change to re-randomize).")
    return parser.parse_args()


def build_demo_dataset(
    *,
    users_in_path: Path,
    ref_in_path: Path,
    users_out_path: Path,
    ref_out_path: Path,
    mapping_dir: Path,
    salt: str = DEFAULT_SALT,
) -> bool:
    print(f"[load] users:     {users_in_path}")
    print(f"[load] reference: {ref_in_path}")
    users_in = pd.read_parquet(users_in_path)
    ref_in = pd.read_parquet(ref_in_path)
    print(f"  users rows:     {len(users_in):,}")
    print(f"  reference rows: {len(ref_in):,}")

    sanitizer = Sanitizer(salt=salt)
    print("[sanitize] transforming users…")
    users_out = sanitizer.transform_users(users_in)
    print("[sanitize] transforming reference…")
    ref_out = sanitizer.transform_reference(ref_in)

    users_out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {users_out_path}")
    users_out.to_parquet(users_out_path, index=False)
    print(f"[write] {ref_out_path}")
    ref_out.to_parquet(ref_out_path, index=False)

    print(f"[write] mapping CSVs in {mapping_dir}")
    written = sanitizer.export_mappings(mapping_dir)
    for label, path in written.items():
        print(f"  {label}: {path}")

    emit_demo_lookup_examples(sanitizer, users_in, users_out, mapping_dir)

    ok = validate(
        users_in=users_in,
        users_out=users_out,
        ref_in=ref_in,
        ref_out=ref_out,
        sanitizer=sanitizer,
    )
    return ok


def main() -> int:
    args = parse_args()
    ok = build_demo_dataset(
        users_in_path=Path(args.users_in),
        ref_in_path=Path(args.ref_in),
        users_out_path=Path(args.users_out),
        ref_out_path=Path(args.ref_out),
        mapping_dir=Path(args.mapping_dir),
        salt=args.salt,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
