import sys
from pathlib import Path

# Ensure the project root (one level above ProductLayer/) is on sys.path
# so that DataLayer, DeterministicLayer, MLLayer, ProductLayer are all importable
# regardless of where `streamlit run` is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from DataLayer.access_exclusions import filter_reference_df, filter_recommendations_df, filter_user_groups_df
from DataLayer.cleaner import DataCleaner
from DataLayer.rights_sheets_loader import RightsSheetsLoader
from DataLayer.permission_cooccurrence import build_cooccurrence_state, cooccurrence_from_state
from DataLayer.subgroup_detection import analyze_recommendation_subgroups
from DeterministicLayer.access_pattern_analyzer import AccessPatternAnalyzer
from DeterministicLayer.privilege_audit import PrivilegeAuditAnalyzer
from ProductLayer.AccessRecommendationEngine import AccessRecommendationEngine


DEFAULT_CLEAN_DATA_PATH = Path("data/processed/clean_users.parquet")
DEFAULT_RAW_DATA_PATH = Path("data/raw")

# Decision label → (icon, background colour)
DECISION_BADGE: dict[str, tuple[str, str]] = {
    "Auto Assign":           ("🟢", "#d4edda"),
    "Strong Recommend":      ("🟢", "#d4edda"),
    "Suggest":               ("🟡", "#fff3cd"),
    "Low Confidence":        ("🟡", "#fff3cd"),
    "Manual Review":         ("🔴", "#f8d7da"),
    "Possible Extra Access": ("🟠", "#fde9d4"),
    "Ignore":                ("⚪", "#f0f0f0"),
}

EXCLUDED_FSY_TITLES = {
    "ce fsy us coordinator",
    "ce fsy us assistant coordinator",
    "ce fsy us wellness coordinator",
    "ce fsy us counselor",
}


# ---------------------------------------------------------------------------
# Data loaders (cached so they only run once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_users(clean_data_path: str, data_mtime: float) -> pd.DataFrame:
    cleaner = DataCleaner(processed_path=clean_data_path)
    return filter_user_groups_df(cleaner.load_cleaned())


@st.cache_data
def load_reference(raw_data_path: str) -> pd.DataFrame:
    try:
        loader = RightsSheetsLoader(raw_path=raw_data_path)
        return filter_reference_df(loader.load_reference_sheets())
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_reference_df() -> pd.DataFrame:
    """Properly-structured empty DataFrame to pass when reference sheets are missing."""
    return pd.DataFrame(columns=[
        "JobTitle", "Department", "EmployeeType", "Supervisor",
        "ReferenceEmployeeName", "AccessCategory", "AccessName",
        "AccessNameClean", "SourceFile",
    ])


def _badge(decision: str) -> str:
    icon, color = DECISION_BADGE.get(decision, ("⚪", "#f0f0f0"))
    return (
        f'<span style="background:{color};padding:2px 8px;border-radius:4px;'
        f'font-size:0.85em;white-space:nowrap">{icon}&nbsp;{decision}</span>'
    )


def _fmt_pct(val: float) -> str:
    return f"{val:.0%}" if val > 0 else "—"


def _normalize_title_for_exclusion(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace(".", "")
    text = " ".join(text.split())
    text = text.replace("councilor", "counselor")
    return text


def _filter_excluded_fsy_roles(users_df: pd.DataFrame) -> pd.DataFrame:
    filtered = users_df.copy()
    title_norm = filtered["Title"].astype(str).apply(_normalize_title_for_exclusion)
    return filtered[~title_norm.isin(EXCLUDED_FSY_TITLES)].copy()


def _render_cooccurrence_table(co_state, gname: str, co_top_n: int) -> None:
    """Display top co-permissions for a target group (explainability only)."""
    co_df = cooccurrence_from_state(
        co_state,
        gname,
        top_n=int(co_top_n),
        max_example_users=5,
    )
    if co_df.empty:
        st.caption("No strong co-occurring permissions found.")
        return
    view = co_df[
        [
            "co_permission",
            "users_with_both",
            "p_b_given_a",
            "jaccard",
            "lift",
            "example_users_overlap",
        ]
    ].copy()
    view["P(co|target)"] = view["p_b_given_a"].apply(lambda x: f"{float(x):.1%}")
    view = view.drop(columns=["p_b_given_a"])
    view = view.rename(
        columns={
            "co_permission": "Co-permission",
            "users_with_both": "Users with both",
            "jaccard": "Jaccard",
            "lift": "Lift",
            "example_users_overlap": "Example users",
        }
    )
    st.dataframe(view, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 1 — New Hire Onboarding
# ---------------------------------------------------------------------------

def render_onboarding_tab(users_df: pd.DataFrame, reference_df: pd.DataFrame) -> None:
    st.subheader("New Hire Access Recommendations")
    st.caption(
        "The engine merges four signals: reference sheet, AD peer frequency, "
        "ML cosine similarity, and copy-from user."
    )

    users_for_recs = _filter_excluded_fsy_roles(users_df)

    titles = sorted(users_for_recs["Title"].dropna().astype(str).unique())
    departments = sorted(users_for_recs["Department"].dropna().astype(str).unique())
    all_netids = [""] + sorted(
        users_for_recs["SamAccountName"].dropna().astype(str).unique()
    ) if "SamAccountName" in users_for_recs.columns else [""]

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        title = st.selectbox("Job Title", titles, key="ob_title")
    with col2:
        department = st.selectbox("Department", departments, key="ob_dept")
    with col3:
        employee_type = st.selectbox(
            "Employee Type", ["Full Time", "Student"], key="ob_emptype"
        )

    with st.expander("Advanced options (supervisor / copy-from / confidence)"):
        adv1, adv2 = st.columns(2)
        with adv1:
            supervisor = st.text_input("Supervisor name (optional)", key="ob_supervisor")
            copy_from = st.selectbox(
                "Copy permissions from existing user (optional)",
                all_netids,
                key="ob_copyfrom",
            )
        with adv2:
            new_hire_netid = st.selectbox(
                "New hire NetID — if already in AD, enables ML similarity",
                all_netids,
                key="ob_newhire",
            )
            min_confidence = st.slider(
                "Minimum AD confidence",
                min_value=0.10,
                max_value=1.00,
                value=0.40,
                step=0.05,
                key="ob_conf",
            )

    show_subgroup_diagnostics = st.checkbox(
        "Show subgroup diagnostics (explainability only; does not change scores)",
        value=False,
        key="ob_subgroup",
    )
    show_cooccurrence = st.checkbox(
        "Show permission co-occurrence insights (explainability only; does not change scores)",
        value=False,
        key="ob_cooc",
    )
    co_top_n = 8
    co_row_cap = 40
    if show_cooccurrence:
        cco1, cco2 = st.columns(2)
        with cco1:
            co_top_n = st.slider(
                "Co-permissions per recommendation",
                min_value=5,
                max_value=10,
                value=8,
                key="ob_co_top",
            )
        with cco2:
            co_row_cap = st.slider(
                "Max recommendations with detail expanders",
                min_value=10,
                max_value=80,
                value=40,
                key="ob_co_rows",
            )

    # Cohort preview before running
    cohort_size = len(
        users_for_recs[
            (users_for_recs["Title"] == title) & (users_for_recs["Department"] == department)
        ]
    )

    run = st.button("Generate Recommendations", type="primary", key="ob_run")

    if not run:
        st.info(
            f"**{cohort_size}** existing users match this title + department. "
            "Click **Generate Recommendations** to run the hybrid engine."
        )
        return

    engine = AccessRecommendationEngine(min_confidence=min_confidence)
    ref = reference_df if not reference_df.empty else _empty_reference_df()

    with st.spinner("Running hybrid recommendation engine…"):
        try:
            recs = engine.recommend_for_hire(
                users_df=users_for_recs,
                reference_df=ref,
                title=title,
                department=department,
                employee_type=employee_type,
                supervisor=supervisor or None,
                copy_from_netid=copy_from or None,
                new_hire_netid=new_hire_netid or None,
            )
        except Exception as exc:
            st.error(f"Engine error: {exc}")
            return

    if recs.empty:
        st.warning("No recommendations found for this role combination.")
        return

    recs = filter_recommendations_df(recs)
    if recs.empty:
        st.warning("No recommendations found for this role combination.")
        return

    # ── Metrics row ──────────────────────────────────────────────────────────
    decision_counts = recs["FinalDecision"].value_counts()
    metric_cols = st.columns(len(DECISION_BADGE))
    for i, (label, (icon, _)) in enumerate(DECISION_BADGE.items()):
        with metric_cols[i]:
            st.metric(f"{icon} {label}", int(decision_counts.get(label, 0)))

    st.divider()

    # ── Results table ─────────────────────────────────────────────────────────
    display_cols = [
        "GroupName", "AccessPattern", "FinalDecision", "FinalScore", "RiskLevel",
        "InReferenceSheet", "ADConfidence", "MLConfidence",
        "UserCountWithGroup", "TotalUsersInRole",
        "AmbiguityReason", "ReviewQuestion", "Reason",
    ]
    display_cols = [c for c in display_cols if c in recs.columns]
    display = recs[display_cols].copy()

    display["FinalScore"] = display["FinalScore"].apply(lambda x: f"{x:.1%}")
    if "ADConfidence" in display.columns:
        display["ADConfidence"] = display["ADConfidence"].apply(_fmt_pct)
    if "MLConfidence" in display.columns:
        display["MLConfidence"] = display["MLConfidence"].apply(_fmt_pct)
    if "InReferenceSheet" in display.columns:
        display["InReferenceSheet"] = display["InReferenceSheet"].apply(
            lambda x: "✅" if x else "—"
        )

    display = display.rename(columns={
        "GroupName":          "Permission",
        "AccessPattern":      "Access pattern",
        "FinalDecision":      "Decision",
        "FinalScore":         "Score",
        "RiskLevel":          "Risk",
        "InReferenceSheet":   "Ref Sheet",
        "ADConfidence":       "AD Conf",
        "MLConfidence":       "ML Conf",
        "UserCountWithGroup": "Users With",
        "TotalUsersInRole":   "Role Total",
        "AmbiguityReason":    "Why this label",
        "ReviewQuestion":     "Review question",
    })

    st.dataframe(display, use_container_width=True, hide_index=True)

    # --- Explainability: subgroup (engine cohort) + co-occurrence (dataset-wide, one index build)
    sub_df = pd.DataFrame()
    subgroup_by_perm: dict = {}
    cohort_size = 0

    if show_subgroup_diagnostics:
        with st.spinner("Computing subgroup diagnostics (engine AD cohort)…"):
            try:
                reference_recs = engine._get_reference_recommendations(
                    reference_df=ref,
                    title=title,
                    department=department,
                    employee_type=employee_type,
                    supervisor=supervisor or None,
                    users_df=users_for_recs,
                    copy_from_netid=copy_from or None,
                )
                comparison_cohort = engine._select_ad_comparison_cohort(
                    users_df=users_for_recs,
                    title=title,
                    department=department,
                    reference_recs=reference_recs,
                    employee_type=employee_type,
                    copy_from_netid=copy_from or None,
                )
                cohort_size = len(comparison_cohort)
                sub_df = analyze_recommendation_subgroups(
                    comparison_cohort=comparison_cohort,
                    recommendations_df=recs,
                )
            except Exception as exc:
                st.warning(f"Subgroup diagnostics unavailable: {exc}")

    if not sub_df.empty:
        subgroup_by_perm = {str(row["permission"]): row for _, row in sub_df.iterrows()}

    co_state = None
    if show_cooccurrence:
        with st.spinner("Indexing users for co-occurrence (one pass over cleaned users)…"):
            try:
                co_state = build_cooccurrence_state(users_for_recs)
            except Exception as exc:
                st.warning(f"Co-occurrence indexing failed: {exc}")

    has_subsection = (show_subgroup_diagnostics and bool(subgroup_by_perm)) or (
        show_cooccurrence and co_state is not None
    )
    if show_subgroup_diagnostics and sub_df.empty and not (show_cooccurrence and co_state is not None):
        st.caption("Subgroup diagnostics: no rows returned (empty cohort or unavailable).")

    if has_subsection:
        st.divider()
        st.markdown("#### Explainability")
        cap_bits = []
        if show_subgroup_diagnostics and subgroup_by_perm:
            cap_bits.append(
                f"Subgroup: engine AD comparison cohort **({cohort_size} users)**."
            )
        if show_cooccurrence and co_state is not None:
            cap_bits.append(
                f"Co-occurrence: **{co_state.n_users}** users in loaded dataset (FSY-excluded slice)."
            )
        st.caption(
            " ".join(cap_bits)
            + " Does **not** change scores, FinalDecision, or FinalScore."
        )

    if not has_subsection:
        return

    recs_sorted = recs.sort_values(
        by=["FinalScore", "GroupName"],
        ascending=[False, True],
    )
    row_cap = int(co_row_cap) if show_cooccurrence else len(recs_sorted)
    iter_recs = recs_sorted.head(row_cap)

    for _, rrow in iter_recs.iterrows():
        gname = str(rrow["GroupName"])
        has_sub = show_subgroup_diagnostics and gname in subgroup_by_perm
        has_co = bool(show_cooccurrence and co_state is not None)

        if show_subgroup_diagnostics and not show_cooccurrence:
            if not has_sub:
                continue
        elif show_cooccurrence and not show_subgroup_diagnostics:
            if not has_co:
                continue
        else:
            if not has_sub and not has_co:
                continue

        if has_sub:
            diag = subgroup_by_perm[gname]
            assessment = str(diag.get("subgroup_assessment", "Rare Access"))
            expander_label = f"{gname} — {assessment}"
        else:
            expander_label = f"{gname} — co-occurrence"

        with st.expander(expander_label, expanded=False):
            if has_sub:
                specialized = assessment == "Subrole Access"
                with_users = diag.get("users_with_permission") or []
                without_users = diag.get("users_without_permission") or []
                with_shared = diag.get("with_shared_permissions") or []
                without_shared = diag.get("without_shared_permissions") or []
                indicators = diag.get("strongest_subgroup_indicators") or []

                st.markdown("### Subgroup")
                st.markdown(
                    "**Specialized subgroup (permission pattern):** "
                    + (
                        "Yes — holders share other permissions at high rate vs non-holders "
                        "(suggests a functional sub-slice)."
                        if specialized
                        else "No — pattern fits rare or broad cohort access (no strong subrole lift)."
                    )
                )
                st.markdown(f"**Subgroup assessment:** `{assessment}`")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"**Users with this permission:** {len(with_users)} "
                        f"— `{', '.join(with_users)}`"
                        if with_users
                        else "**Users with this permission:** 0"
                    )
                with c2:
                    st.markdown(
                        f"**Users without:** {len(without_users)} "
                        f"— `{', '.join(without_users)}`"
                        if without_users
                        else "**Users without:** 0"
                    )
                st.markdown("**Shared permissions among holders** (high support within “with” slice):")
                if with_shared:
                    st.markdown("\n".join(f"- `{p}`" for p in with_shared))
                else:
                    st.caption("None above default frequency threshold.")
                st.markdown("**Contrast — commonly shared by users without this permission:**")
                if without_shared:
                    st.markdown("\n".join(f"- `{p}`" for p in without_shared))
                else:
                    st.caption("None above default frequency threshold.")
                st.markdown("**Strongest subgroup indicators** (lift = rate_with − rate_without):")
                if indicators:
                    st.dataframe(
                        pd.DataFrame(indicators),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No indicators passed default lift/support thresholds.")

            if has_co:
                st.markdown("### Permission co-occurrence")
                _render_cooccurrence_table(co_state, gname, co_top_n)


# ---------------------------------------------------------------------------
# Tab 2 — Privilege Audit
# ---------------------------------------------------------------------------

def render_privilege_audit_tab(users_df: pd.DataFrame) -> None:
    st.subheader("Privilege Audit")
    st.caption(
        "Identifies users whose AD group count significantly exceeds their "
        "role-peer median — a signal of privilege creep or unreviewed historical access."
    )

    threshold = st.slider(
        "Flag users who have ≥ N× their role-peer median group count",
        min_value=1.2,
        max_value=4.0,
        value=1.5,
        step=0.1,
        key="pa_threshold",
    )

    analyzer = PrivilegeAuditAnalyzer(threshold_multiplier=threshold)

    with st.spinner("Analysing privilege distribution…"):
        flagged = analyzer.get_flagged_users(users_df)
        summary = analyzer.get_role_summary(users_df)

    # ── Metrics row ───────────────────────────────────────────────────────────
    total_users = len(users_df)
    flagged_count = len(flagged)
    pct_flagged = flagged_count / total_users * 100 if total_users else 0
    avg_ratio = flagged["OverprivilegeRatio"].mean() if not flagged.empty else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Users", f"{total_users:,}")
    m2.metric("Flagged", f"{flagged_count:,}")
    m3.metric("% Flagged", f"{pct_flagged:.1f}%")
    m4.metric("Avg Ratio (flagged)", f"{avg_ratio:.2f}×")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 🚩 Flagged Users")
        if flagged.empty:
            st.success(
                f"No users exceed {threshold}× their role-peer median "
                f"(minimum 2 peers required)."
            )
        else:
            display = flagged.copy()
            display["RoleMedian"] = display["RoleMedian"].apply(lambda x: f"{x:.1f}")
            display["OverprivilegeRatio"] = display["OverprivilegeRatio"].apply(
                lambda x: f"{x:.2f}×"
            )
            display = display.rename(columns={
                "SamAccountName":    "User",
                "GroupCount":        "Groups",
                "RoleMedian":        "Role Median",
                "ExtraGroupCount":   "Extra",
                "OverprivilegeRatio":"Ratio",
                "RolePeerCount":     "Peers",
            })
            st.dataframe(display, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("#### 📊 Role Group-Count Summary")
        disp_summary = summary.copy()
        disp_summary["MedianGroups"] = disp_summary["MedianGroups"].apply(
            lambda x: f"{x:.1f}"
        )
        st.dataframe(disp_summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3 — Orphaned Access
# ---------------------------------------------------------------------------

def render_orphaned_access_tab(users_df: pd.DataFrame) -> None:
    st.subheader("Orphaned & Unique Access")
    st.caption(
        "Finds users holding AD groups that are absent or extremely rare in their "
        "cohort — a signal of orphaned access from previous roles, one-off grants, "
        "or data entry errors."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        scope_label = st.radio(
            "Comparison scope",
            ["Department", "Role (Title + Department)"],
            index=1,
            horizontal=True,
            key="oa_scope",
        )
    with ctrl2:
        min_peers = st.slider(
            "Minimum cohort size (skip tiny teams)",
            min_value=2,
            max_value=20,
            value=5,
            key="oa_minpeers",
        )
    with ctrl3:
        st.write("")  # vertical spacer
        run_oa = st.button("Run Analysis", key="oa_run")

    if not run_oa:
        st.info("Configure the options above and click **Run Analysis**.")
        return

    scope_key = "department" if scope_label == "Department" else "role"
    analyzer = AccessPatternAnalyzer()

    with st.spinner("Scanning for orphaned and rare access…"):
        result = analyzer.find_orphaned_access(
            users_df,
            scope=scope_key,
            min_peer_count=min_peers,
        )

    group_keys = ["Department"] if scope_key == "department" else ["Title", "Department"]
    cohort_sizes = users_df.groupby(group_keys).size()
    small_cohort_count = int((cohort_sizes < min_peers).sum())
    analyzed_cohort_count = int((cohort_sizes >= min_peers).sum())

    if result.empty:
        if analyzed_cohort_count == 0:
            st.warning(
                f"No cohorts met the minimum size of {min_peers}. "
                "Result is inconclusive due to insufficient cohort data."
            )
        else:
            st.success("No unique or orphaned access patterns detected with these settings.")
        if small_cohort_count > 0:
            st.caption(
                f"Skipped {small_cohort_count} cohort(s) below minimum size "
                f"({min_peers}); analyzed {analyzed_cohort_count} cohort(s)."
            )
        return

    unique_total = int(result["UniqueGroupCount"].sum())
    rare_total = int(result["RareGroupCount"].sum())

    ma, mb, mc = st.columns(3)
    ma.metric("Users Flagged", len(result))
    mb.metric("Total Unique Groups", unique_total)
    mc.metric("Total Rare Groups", rare_total)
    if small_cohort_count > 0:
        st.caption(
            f"Skipped {small_cohort_count} cohort(s) below minimum size "
            f"({min_peers}); analyzed {analyzed_cohort_count} cohort(s)."
        )

    st.divider()
    st.markdown("#### Flagged Users")

    def _split_groups(raw: str) -> list[str]:
        return [g.strip() for g in raw.split(",") if g.strip()] if raw else []

    for _, user_row in result.iterrows():
        unique_list = _split_groups(user_row["UniqueGroups"])
        rare_list = _split_groups(user_row["RareGroups"])

        label = (
            f"{user_row['SamAccountName']}  |  "
            f"{user_row['Title']} — {user_row['Department']}  |  "
            f"🔴 {int(user_row['UniqueGroupCount'])} unique   "
            f"🟡 {int(user_row['RareGroupCount'])} rare"
        )

        with st.expander(label):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"**🔴 Unique groups** — only this user "
                    f"in cohort of {int(user_row['CohortSize'])}"
                )
                if unique_list:
                    for g in unique_list:
                        st.markdown(f"- `{g}`")
                else:
                    st.caption("None.")

            with col2:
                st.markdown(
                    f"**🟡 Rare groups** — held by <10% "
                    f"of cohort ({int(user_row['CohortSize'])} peers)"
                )
                if rare_list:
                    for g in rare_list:
                        st.markdown(f"- `{g}`")
                else:
                    st.caption("None.")

            st.markdown("**Cohort peers:**")
            peers = [
                p.strip()
                for p in str(user_row.get("CohortMembers", "")).split(",")
                if p.strip()
            ]
            if peers:
                st.caption("  ·  ".join(peers))
            else:
                st.caption("No peers recorded.")


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AccessGraph", layout="wide", page_icon="🔐")

    st.title("🔐 AccessGraph")
    st.caption(
        "AI-assisted access governance · hybrid RBAC + ML · explainable recommendations"
    )

    # Sidebar — data paths
    with st.sidebar:
        st.header("Data Sources")
        clean_path = st.text_input(
            "Cleaned user data (parquet)",
            value=str(DEFAULT_CLEAN_DATA_PATH),
        )
        raw_path = st.text_input(
            "Raw data folder",
            value=str(DEFAULT_RAW_DATA_PATH),
        )

    # Load user data
    try:
        mtime = Path(clean_path).stat().st_mtime
        users_df = load_users(clean_path, mtime)
    except Exception as exc:
        st.error(f"Could not load user data: {exc}")
        st.stop()

    required_cols = {"Title", "Department", "GroupsList"}
    missing_cols = required_cols - set(users_df.columns)
    if missing_cols:
        st.error(f"Dataset is missing required columns: {sorted(missing_cols)}")
        st.stop()

    # Load reference sheets (optional — degrades gracefully)
    reference_df = load_reference(raw_path)
    if reference_df.empty:
        st.sidebar.warning("⚠️ Reference sheets not found — reference signal disabled.")
    else:
        st.sidebar.success(
            f"✅ Reference sheet: {len(reference_df):,} access entries loaded."
        )

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🧑‍💼 New Hire Onboarding",
        "⚠️ Privilege Audit",
        "🔍 Orphaned Access",
    ])

    with tab1:
        render_onboarding_tab(users_df, reference_df)

    with tab2:
        render_privilege_audit_tab(users_df)

    with tab3:
        render_orphaned_access_tab(users_df)


if __name__ == "__main__":
    main()
