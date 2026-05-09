from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from DataLayer.access_exclusions import filter_group_list
from DeterministicLayer.permission_filter import PermissionFilter


def _filtered_groups_for_row(groups: object, permission_filter: PermissionFilter) -> set[str]:
    """Parse GroupsList and drop excluded / noisy / malformed entries."""
    raw = filter_group_list(groups)
    out: set[str] = set()
    for g in raw:
        text = str(g).strip()
        if not text or permission_filter.should_ignore(text):
            continue
        out.add(text)
    return out


def iter_user_group_sets(
    users_df: pd.DataFrame,
    *,
    groups_column: str = "GroupsList",
    id_column: str = "SamAccountName",
    permission_filter: PermissionFilter | None = None,
) -> Iterable[tuple[str, set[str]]]:
    pf = permission_filter or PermissionFilter()
    if groups_column not in users_df.columns:
        return
    id_present = id_column in users_df.columns
    for _, row in users_df.iterrows():
        uid = str(row[id_column]).strip() if id_present else ""
        if not uid or uid.lower() == "nan":
            continue
        yield uid, _filtered_groups_for_row(row.get(groups_column), pf)


def global_permission_counts(
    users_df: pd.DataFrame,
    *,
    groups_column: str = "GroupsList",
    id_column: str = "SamAccountName",
    permission_filter: PermissionFilter | None = None,
) -> tuple[int, Counter]:
    """
    Return (n_users_with_any_id, Counter of permission -> number of users holding it).
    """
    pf = permission_filter or PermissionFilter()
    counts: Counter = Counter()
    n_users = 0
    for _uid, groups in iter_user_group_sets(
        users_df,
        groups_column=groups_column,
        id_column=id_column,
        permission_filter=pf,
    ):
        n_users += 1
        counts.update(groups)
    return n_users, counts


_EMPTY_COOC_COLUMNS = [
    "co_permission",
    "users_with_target",
    "users_with_b",
    "users_with_both",
    "p_b_given_a",
    "p_a_given_b",
    "jaccard",
    "lift",
    "overlap_pct",
    "example_users_overlap",
]


def _empty_cooc_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COOC_COLUMNS)


@dataclass(frozen=True)
class CooccurrenceState:
    """Single scan of users: reuse for many target permissions without re-reading rows."""

    pairs: list[tuple[str, set[str]]]
    global_counts: Counter
    n_users: int


def build_cooccurrence_state(
    users_df: pd.DataFrame,
    *,
    groups_column: str = "GroupsList",
    id_column: str = "SamAccountName",
    permission_filter: PermissionFilter | None = None,
) -> CooccurrenceState:
    pf = permission_filter or PermissionFilter()
    pairs: list[tuple[str, set[str]]] = list(
        iter_user_group_sets(
            users_df,
            groups_column=groups_column,
            id_column=id_column,
            permission_filter=pf,
        )
    )
    global_counts: Counter = Counter()
    for _uid, groups in pairs:
        global_counts.update(groups)
    return CooccurrenceState(pairs=pairs, global_counts=global_counts, n_users=len(pairs))


def cooccurrence_from_state(
    state: CooccurrenceState,
    target_permission: str,
    *,
    top_n: int = 20,
    max_example_users: int = 5,
) -> pd.DataFrame:
    """
    Same output schema as cooccurrence_with_target, using a pre-built CooccurrenceState.
    """
    target = str(target_permission).strip()
    if not target:
        return _empty_cooc_df()

    cooc_counts: Counter = Counter()
    examples_for_b: dict[str, list[str]] = {}

    for uid, groups in state.pairs:
        if target not in groups:
            continue
        for b in groups:
            if b == target:
                continue
            cooc_counts[b] += 1
            if b not in examples_for_b:
                examples_for_b[b] = []
            if len(examples_for_b[b]) < max_example_users:
                examples_for_b[b].append(uid)

    users_a = state.global_counts.get(target, 0)
    if users_a == 0:
        return _empty_cooc_df()

    n_users = state.n_users
    global_counts = state.global_counts
    rows: list[dict] = []
    for b, both in cooc_counts.items():
        ub = global_counts.get(b, 0)
        p_b_a = both / users_a if users_a else 0.0
        p_a_b = both / ub if ub else 0.0
        union = users_a + ub - both
        jaccard = both / union if union > 0 else 0.0
        if n_users > 0 and users_a > 0 and ub > 0:
            p_joint = both / n_users
            p_a = users_a / n_users
            p_b = ub / n_users
            lift = p_joint / (p_a * p_b) if (p_a * p_b) > 0 else 0.0
        else:
            lift = 0.0

        ex = examples_for_b.get(b, [])
        rows.append(
            {
                "co_permission": b,
                "users_with_target": users_a,
                "users_with_b": ub,
                "users_with_both": both,
                "p_b_given_a": round(p_b_a, 4),
                "p_a_given_b": round(p_a_b, 4),
                "jaccard": round(jaccard, 4),
                "lift": round(lift, 4),
                "overlap_pct": round(100.0 * p_b_a, 2),
                "example_users_overlap": ", ".join(ex),
            }
        )

    if not rows:
        return _empty_cooc_df()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["users_with_both", "lift", "jaccard"],
        ascending=[False, False, False],
    ).head(top_n)
    return out.reset_index(drop=True)


def cooccurrence_with_target(
    users_df: pd.DataFrame,
    target_permission: str,
    *,
    top_n: int = 20,
    groups_column: str = "GroupsList",
    id_column: str = "SamAccountName",
    max_example_users: int = 5,
    permission_filter: PermissionFilter | None = None,
) -> pd.DataFrame:
    """
    For a target permission A, summarize co-occurrence with every other permission B.

    Columns:
    - co_permission (B)
    - users_with_target (|A|)
    - users_with_b (|B|)
    - users_with_both (|A ∩ B|)
    - p_b_given_a  P(B|A)
    - p_a_given_b  P(A|B)
    - jaccard
    - lift  (P(A∩B) / (P(A)*P(B))) = users_both * N / (users_a * users_b)
    - overlap_pct  same as P(B|A) as percentage (handy for reporting)
    - example_users_overlap  sample NetIDs with both A and B
    """
    state = build_cooccurrence_state(
        users_df,
        groups_column=groups_column,
        id_column=id_column,
        permission_filter=permission_filter,
    )
    return cooccurrence_from_state(
        state,
        target_permission,
        top_n=top_n,
        max_example_users=max_example_users,
    )
