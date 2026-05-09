from __future__ import annotations

import pandas as pd

# Canonical labels for recommendation explainability (conservative defaults).
PATTERN_BASELINE = "Baseline Access"
PATTERN_COMMON = "Common Access"
PATTERN_SUBROLE = "Subrole Access"
PATTERN_RARE = "Rare Access"
PATTERN_UNIQUE = "Unique Access"
PATTERN_POSSIBLE_EXTRA = "Possible Extra Access"
PATTERN_HIGH_RISK = "High Risk"
PATTERN_UNKNOWN = "Unknown / Ambiguous"


def subrole_map_from_subgroup_df(sub_df: pd.DataFrame) -> dict[str, bool]:
    """Map permission string (GroupName) -> True if subgroup detector says Subrole Access."""
    if sub_df is None or sub_df.empty:
        return {}
    out: dict[str, bool] = {}
    for _, r in sub_df.iterrows():
        key = str(r.get("permission", "")).strip()
        if not key:
            continue
        out[key] = str(r.get("subgroup_assessment", "")).strip() == "Subrole Access"
    return out


def _safe_int(x: object, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def label_access_pattern(
    row: pd.Series,
    *,
    has_subrole_evidence: bool,
) -> tuple[str, str, str]:
    """
    Return (AccessPattern, AmbiguityReason, ReviewQuestion).

    Rules are conservative and cohort-frequency-first; subgroup evidence
    comes from DataLayer.subgroup_detection (co-occurrence lift).
    """
    group = str(row.get("GroupName", "")).strip() or "(unknown)"
    risk = str(row.get("RiskLevel", "Low")).strip()
    total = _safe_int(row.get("TotalUsersInRole"), 0)
    count = _safe_int(row.get("UserCountWithGroup"), 0)
    in_ref = bool(row.get("InReferenceSheet", False))
    ambiguous_ref = bool(row.get("AmbiguousReferenceTemplate", False))
    copy_has = bool(row.get("CopyFromUserHasIt", False))
    ml_conf = _safe_float(row.get("MLConfidence"), 0.0)
    global_rate = _safe_float(row.get("GlobalGroupRate"), 0.0)
    ad_conf = _safe_float(row.get("ADConfidence"), 0.0)

    def q_template(kind: str) -> str:
        if kind == "baseline":
            return (
                f"Should `{group}` remain standard for every hire in this role, "
                "or has the baseline access set changed?"
            )
        if kind == "common":
            return (
                f"Should `{group}` be default for most hires in this role, "
                "or only for specific functions?"
            )
        if kind == "subrole":
            return (
                f"Does this hire belong to the function or team that typically "
                f"needs `{group}`?"
            )
        if kind == "possible_extra":
            return (
                f"Does this hire need `{group}` for their specific duties "
                "(not assumed for all peers in this role)?"
            )
        if kind == "rare":
            return (
                f"Is `{group}` justified given uncommon peer usage, "
                "or could it be legacy or over-assigned access?"
            )
        if kind == "unique":
            return (
                f"Only one peer in the cohort holds `{group}` — is this a one-off grant, "
                "or should a sub-template capture this access?"
            )
        if kind == "high_risk":
            return (
                "Has elevated or sensitive access been explicitly approved for this hire "
                "and role under your governance process?"
            )
        if kind == "unknown":
            return (
                f"Can you confirm whether `{group}` applies, given weak or conflicting "
                "automated signals?"
            )
        return f"Should `{group}` be part of access for this hire?"

    # 1) High-risk permission wording — always review regardless of support.
    if risk == "High":
        reason = (
            "Permission matches elevated-risk heuristics; manual review is required "
            f"regardless of cohort support (AD={ad_conf:.0%}, cohort {count}/{max(total, 1)})."
        )
        if global_rate >= 0.5:
            reason += f" Note: high global prevalence ({global_rate:.0%}) does not remove review obligation."
        return PATTERN_HIGH_RISK, reason, q_template("high_risk")

    # 2) No cohort denominator
    if total <= 0:
        reason = (
            "No AD comparison cohort size; peer frequency bands cannot be applied reliably."
        )
        return PATTERN_UNKNOWN, reason, q_template("unknown")

    support = count / total

    # 3) Zero holders in cohort
    if count <= 0:
        if in_ref:
            reason = (
                "Reference lists this access but the AD comparison cohort has no holders; "
                "frequency is unknown for this slice."
            )
            if ambiguous_ref:
                reason += " Reference template is also ambiguous (multiple variants)."
            return PATTERN_UNKNOWN, reason, q_template("unknown")
        if copy_has:
            reason = (
                "No cohort holders; permission appears only via copy-from template — "
                "treat as non-baseline unless duties require it."
            )
            return PATTERN_POSSIBLE_EXTRA, reason, q_template("possible_extra")
        if ml_conf >= 0.5:
            reason = (
                f"ML similarity suggests the permission (ml={ml_conf:.0%}) but AD cohort "
                "shows zero holders; evidence is mixed."
            )
            return PATTERN_UNKNOWN, reason, q_template("unknown")
        reason = "No AD cohort holders and no reference/copy signal strong enough to classify."
        return PATTERN_RARE, reason, q_template("rare")

    # 4) Ambiguous reference + weak cohort support (conservative)
    if ambiguous_ref and in_ref and support < 0.5 and not has_subrole_evidence:
        reason = (
            "Reference template is ambiguous and cohort support is below 50% without "
            f"subgroup co-occurrence evidence (AD {count}/{total})."
        )
        return PATTERN_UNKNOWN, reason, q_template("unknown")

    # 5) Frequency bands
    if support >= 0.9:
        reason = (
            f"Strong cohort support: {count}/{total} users (≥90%) in the AD comparison cohort; "
            f"ADConfidence={ad_conf:.0%}."
        )
        if global_rate >= 0.8:
            reason += f" Also very common org-wide ({global_rate:.0%}); confirm least-privilege."
        return PATTERN_BASELINE, reason, q_template("baseline")

    if support >= 0.5:
        reason = (
            f"Majority but not universal cohort support: {count}/{total} (50–89%); "
            f"ADConfidence={ad_conf:.0%}."
        )
        return PATTERN_COMMON, reason, q_template("common")

    # Single holder — treat as unique even if share falls in the 10–49% band (e.g. 1/10).
    if count == 1 and total >= 2:
        reason = (
            f"Only one user in the cohort ({count}/{total}) holds this permission; "
            "may be individual grant or missing sub-template."
        )
        return PATTERN_UNIQUE, reason, q_template("unique")

    if support >= 0.1:
        if has_subrole_evidence:
            reason = (
                f"Minority cohort share {count}/{total} (10–49%) with subgroup co-occurrence "
                "pattern (holders share distinguishing permissions vs non-holders)."
            )
            return PATTERN_SUBROLE, reason, q_template("subrole")
        reason = (
            f"Minority cohort share {count}/{total} (10–49%) without a clear subgroup bundle "
            "in automated analysis."
        )
        if copy_has:
            reason += " Copy-from user has the permission — verify duty-based need."
        return PATTERN_POSSIBLE_EXTRA, reason, q_template("possible_extra")

    # support < 0.1 (and count != 1 already implies count == 0 handled above; here count >= 2)
    reason = (
        f"Very low cohort support: {count}/{total} (<10%); ADConfidence={ad_conf:.0%}."
    )
    return PATTERN_RARE, reason, q_template("rare")


def apply_access_pattern_columns(
    merged: pd.DataFrame,
    sub_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add AccessPattern, AmbiguityReason, ReviewQuestion to recommendations."""
    if merged.empty:
        out = merged.copy()
        out["AccessPattern"] = pd.Series(dtype=str)
        out["AmbiguityReason"] = pd.Series(dtype=str)
        out["ReviewQuestion"] = pd.Series(dtype=str)
        return out

    subrole_map = subrole_map_from_subgroup_df(sub_df if sub_df is not None else pd.DataFrame())

    patterns: list[str] = []
    reasons: list[str] = []
    questions: list[str] = []

    for _, row in merged.iterrows():
        gname = str(row.get("GroupName", "")).strip()
        has_sub = bool(subrole_map.get(gname, False))
        p, r, q = label_access_pattern(row, has_subrole_evidence=has_sub)
        patterns.append(p)
        reasons.append(r)
        questions.append(q)

    out = merged.copy()
    out["AccessPattern"] = patterns
    out["AmbiguityReason"] = reasons
    out["ReviewQuestion"] = questions
    return out
