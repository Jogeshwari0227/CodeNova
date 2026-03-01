import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import math

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduInsight AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg:         #0d0f14;
    --panel:      #13161e;
    --card:       #1a1e2b;
    --border:     #252a3a;
    --accent:     #4f8ef7;
    --accent2:    #a78bfa;
    --green:      #34d399;
    --orange:     #fb923c;
    --red:        #f87171;
    --text:       #e2e8f0;
    --muted:      #64748b;
    --mono:       'Space Mono', monospace;
    --sans:       'Syne', sans-serif;
}

/* ── Global ── */
.stApp { background: var(--bg) !important; }
html, body, [class*="css"] { font-family: var(--sans) !important; color: var(--text) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* ── Inputs ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
}
.stSlider [data-testid="stSlider"] div div div div {
    background: var(--accent) !important;
}
.stTextInput > div > div,
.stNumberInput > div > div,
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--sans) !important;
    font-weight: 700 !important;
    padding: 10px 28px !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 12px !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--mono) !important; font-size: 2rem !important; }

/* ── Custom Cards ── */
.edu-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.edu-card-accent {
    border-left: 4px solid var(--accent);
}
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-green  { background: rgba(52,211,153,.15); color: var(--green);  border: 1px solid rgba(52,211,153,.3); }
.badge-orange { background: rgba(251,146,60,.15);  color: var(--orange); border: 1px solid rgba(251,146,60,.3); }
.badge-red    { background: rgba(248,113,113,.15); color: var(--red);    border: 1px solid rgba(248,113,113,.3); }
.badge-blue   { background: rgba(79,142,247,.15);  color: var(--accent); border: 1px solid rgba(79,142,247,.3); }

.score-ring {
    font-family: var(--mono);
    font-size: 3.5rem;
    font-weight: 700;
    text-align: center;
    line-height: 1;
}
.section-header {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f8ef7 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 6px;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* Matplotlib dark patch */
figure { background: transparent !important; }

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────────────────────────
# Initialize ALL keys here so they're always available regardless of merge order
_defaults = {
    'logged_in':         False,
    'portal_view':       False,
    'role':              None,     # 'admin' after login, None when logged out
    'prediction_result': None,
}
for _key, _val in _defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ... Imports (st, pd, joblib, etc.)

# 1. LOAD DATA AT THE TOP (Global Scope)
# This ensures df_raw is available to EVERY part of the app
@st.cache_data
def load_global_data():
    return pd.read_csv('data/StudentPerformanceFactors.csv')

df_raw = load_global_data()

# 2. LOAD MODEL
model = joblib.load('model/model.pkl')

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-size:1.6rem; font-weight:800; font-family:Syne,sans-serif;'>
            🎓 EduInsight <span style='color:#4f8ef7;'>AI</span>
        </div>
        <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>
            Attendance & Performance Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    # st.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)
    # st.markdown("""
    # <div class='edu-card' style='padding:14px; margin-bottom:12px;'>
    #     <div style='font-size:0.85rem; color:#64748b;'>Problem Statement</div>
    #     <div style='font-size:0.95rem; font-weight:600; margin-top:4px;'>PS 05 — Attendance & Performance Correlation</div>
    # </div>
    # """, unsafe_allow_html=True)

    if st.session_state['logged_in']:
        st.markdown("<div class='section-header' style='margin-top:20px;'>Account</div>", unsafe_allow_html=True)
        st.markdown("<div class='badge badge-green'>● Logged In as Admin</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['role'] = None
            st.rerun()

    st.markdown("---")
    st.markdown("<div class='section-header'>Public Access</div>", unsafe_allow_html=True)
    if st.button("🏫 Student Portal", use_container_width=True):
        st.session_state['portal_view'] = not st.session_state.get('portal_view', False)
        st.rerun()
    st.markdown("""<div style='color:#64748b; font-size:0.72rem; margin-top:6px;'>View defaulters list & attendance recovery calculator</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#64748b; font-size:0.75rem; line-height:1.6;'>
    Built with RandomForest ML<br>
    Dataset: 6,607 student records<br>
    Features: Attendance, Study Hours,<br>
    &nbsp;&nbsp;Previous Scores, Motivation
    </div>
    """, unsafe_allow_html=True)


# ─── STUDENT PORTAL (PUBLIC) ─────────────────────────────────────────────────────
if st.session_state.get('portal_view', False):

    st.markdown("""
    <div style='display:flex; align-items:center; gap:16px; margin-bottom:6px;'>
        <div class='hero-title'>Student Portal</div>
        <span class='badge badge-green' style='font-size:10px;'>PUBLIC</span>
    </div>
    <div class='hero-sub'>Attendance Defaulters List & Recovery Calculator — No login required</div>
    <br>
    """, unsafe_allow_html=True)

    # ── KPIs ──
    threshold = 75
    defaulters_df = df_raw[df_raw['Attendance'] < threshold].copy().reset_index()
    defaulters_df.rename(columns={'index': 'student_id'}, inplace=True)
    defaulters_df['student_id'] = defaulters_df['student_id'] + 1  # 1-indexed

    total_students = len(df_raw)
    n_defaulters   = len(defaulters_df)
    avg_def_att    = defaulters_df['Attendance'].mean() if n_defaulters > 0 else 0

    pk1, pk2, pk3 = st.columns(3)
    pk1.metric("🎓 Total Students",      f"{total_students:,}")
    pk2.metric("🚨 Defaulters (<75%)",   f"{n_defaulters:,}", delta=f"{n_defaulters/total_students*100:.1f}% of class", delta_color="inverse")
    pk3.metric("📉 Avg Defaulter Attendance", f"{avg_def_att:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters ──
    st.markdown("<div class='section-header'>🔍 Search & Filter</div>", unsafe_allow_html=True)
    fc1, fc2, fc3, fc4, fc5 = st.columns([1.4, 1, 1, 1, 1])

    with fc1:
        search_query = st.text_input(
            "🔎 Search by Name or ID",
            placeholder="e.g. Deepak Bhatt  or  42",
            help="Type a student name (partial match supported) or enter an exact Student ID number",
            label_visibility="visible"
        )
    with fc2:
        gender_opts = ["All"] + sorted(df_raw['Gender'].dropna().unique().tolist())
        filter_gender = st.selectbox("Gender", gender_opts)
    with fc3:
        att_max = st.slider("Max Attendance %", 40, 74, 74)
    with fc4:
        sort_by = st.selectbox("Sort By", ["Attendance ↑ (worst first)", "Attendance ↓", "Exam Score ↑", "Exam Score ↓"])
    with fc5:
        show_all = st.checkbox("Show all students", value=False,
                               help="When checked, shows the full student list — not just defaulters")

    # Build base dataframe — defaulters only OR all students
    base_df = df_raw.copy().reset_index()
    base_df.rename(columns={'index': 'student_id'}, inplace=True)
    base_df['student_id'] = base_df['student_id'] + 1

    # Detect whether the loaded CSV has a Name column
    HAS_NAME = 'Name' in base_df.columns

    if not show_all:
        filtered = base_df[base_df['Attendance'] < 75].copy()
        filtered = filtered[filtered['Attendance'] <= att_max]
    else:
        filtered = base_df[base_df['Attendance'] <= att_max].copy()

    if filter_gender != "All":
        filtered = filtered[filtered['Gender'] == filter_gender]

    # Smart search — detect if query is numeric (ID) or text (name)
    query = search_query.strip()
    search_matched = False
    if query:
        if query.isdigit():
            # Search by exact Student ID
            id_result = filtered[filtered['student_id'] == int(query)]
            if len(id_result) > 0:
                filtered = id_result
                search_matched = True
            else:
                st.warning(f"No student found with ID **{query}** in the current filtered list.")
                filtered = filtered.iloc[0:0]
        else:
            # Search by name — only if Name column exists in the CSV
            if HAS_NAME:
                name_result = filtered[filtered['Name'].str.contains(query, case=False, na=False)]
                if len(name_result) > 0:
                    filtered = name_result
                    search_matched = True
                else:
                    st.warning(f"No student found matching **\"{query}\"**. Try a partial name or check spelling.")
                    filtered = filtered.iloc[0:0]
            else:
                st.error("⚠️ Name search unavailable — your CSV does not have a **Name** column. "
                         "Please replace `data/StudentPerformanceFactors.csv` with the updated file that includes names, then restart the app.")
                filtered = filtered.iloc[0:0]

    sort_map = {
        "Attendance ↑ (worst first)": ('Attendance', True),
        "Attendance ↓": ('Attendance', False),
        "Exam Score ↑": ('Exam_Score', True),
        "Exam Score ↓": ('Exam_Score', False),
    }
    sort_col, sort_asc = sort_map[sort_by]
    filtered = filtered.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    # Result count label
    list_label = "students" if show_all else "defaulters"
    match_note = f" matching <b style='color:#4f8ef7;'>'{query}'</b>" if query and search_matched else ""
    st.markdown(
        f"<div style='color:#64748b; font-size:0.85rem; margin-bottom:8px;'>"
        f"Showing <b style='color:#f87171;'>{len(filtered)}</b> {list_label}{match_note}</div>",
        unsafe_allow_html=True
    )

    # ── Defaulters Table ──
    st.markdown("<div class='section-header'>📋 Defaulters List</div>", unsafe_allow_html=True)

    def severity_badge(att):
        if att < 60:
            return "🔴 Critical"
        elif att < 65:
            return "🟠 High Risk"
        elif att < 75:
            return "🟡 At Risk"
        else:
            return "🟢 OK"

    has_name = 'Name' in filtered.columns
    cols_to_show = ['student_id'] + (['Name'] if has_name else []) + ['Gender', 'Attendance', 'Exam_Score', 'Hours_Studied', 'Motivation_Level']
    cols_to_show = [c for c in cols_to_show if c in filtered.columns]

    display_df = filtered[cols_to_show].copy()
    rename_map = {
        'student_id': 'ID', 'Name': 'Name', 'Gender': 'Gender',
        'Attendance': 'Attendance %', 'Exam_Score': 'Exam Score',
        'Hours_Studied': 'Hours Studied/Week', 'Motivation_Level': 'Motivation'
    }
    display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns}, inplace=True)
    display_df['Status'] = display_df['Attendance %'].apply(severity_badge)

    st.dataframe(display_df, use_container_width=True, height=340, hide_index=True)

    # ── Download button ──
    csv_export = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download List (CSV)", csv_export,
                       file_name="defaulters_list.csv", mime="text/csv")

    st.markdown("<br>", unsafe_allow_html=True)

    # # ── Safety Margin / Recovery Calculator ──────────────────────────────────────
    # st.markdown("---")
    # st.markdown("""
    # <div class='section-header'>🛡️ Attendance Recovery Calculator</div>
    # <div style='color:#64748b; font-size:0.85rem; margin-bottom:16px;'>
    #     Enter your lecture details to see exactly how many classes you need to attend to exit the defaulters list,
    #     and optionally plan ahead towards a personal target.
    # </div>
    # """, unsafe_allow_html=True)

    # rc1, rc2 = st.columns([1, 1.4], gap="large")

    # with rc1:
    #     st.markdown("<div class='edu-card' style='padding:28px;'>", unsafe_allow_html=True)
    #     st.markdown("<div class='section-header'>Enter Your Details</div>", unsafe_allow_html=True)

    #     total_lectures  = st.number_input("📚 Total Lectures Conducted", min_value=1, max_value=500, value=100,
    #                                       help="Total number of lectures held so far this semester")
    #     attended        = st.number_input("✅ Lectures Attended by You", min_value=0, max_value=500, value=65,
    #                                       help="Number of lectures you have actually attended")
    #     future_lectures = st.number_input("📅 Remaining Scheduled Lectures", min_value=0, max_value=300, value=30,
    #                                       help="How many more lectures are left in the semester")

    #     st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    #     # Mandatory institutional threshold — fixed, always shown
    #     THRESHOLD = 75
    #     st.markdown(f"""
    #     <div style='padding:10px 14px; background:#13161e; border-radius:10px;
    #                 border:1px solid #facc1540; margin-bottom:10px;'>
    #         <div style='font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:1px;'>🏛️ Institutional Minimum (Defaulters List)</div>
    #         <div style='font-size:1rem; font-weight:800; color:#facc15; margin-top:2px;'>{THRESHOLD}% — Fixed, cannot be changed</div>
    #     </div>
    #     """, unsafe_allow_html=True)

    #     # Personal target slider — fully adjustable, separate goal
    #     personal_target = st.slider(
    #         "🎯 My Personal Target %",
    #         min_value=50, max_value=100, value=85,
    #         help="Set a personal attendance goal beyond the minimum. Must be ≥ 75% to be meaningful."
    #     )
    #     if personal_target < THRESHOLD:
    #         st.warning(f"⚠️ Your personal target ({personal_target}%) is below the institutional minimum ({THRESHOLD}%). You'll still be on the defaulters list even if you hit this target.")

    #     # Clamp attended to total_lectures
    #     attended = min(int(attended), int(total_lectures))
    #     st.markdown("</div>", unsafe_allow_html=True)

    # with rc2:
    #     total_lectures  = int(total_lectures)
    #     future_lectures = int(future_lectures)
    #     attended        = int(attended)

    #     current_pct      = (attended / total_lectures * 100) if total_lectures > 0 else 0
    #     max_possible_att = (attended + future_lectures) / (total_lectures + future_lectures) * 100 \
    #                        if (total_lectures + future_lectures) > 0 else current_pct

    #     def calc_lectures_needed(target_pct, attended, total):
    #         """How many consecutive lectures must a student attend to reach target_pct?"""
    #         frac = target_pct / 100
    #         shortfall = frac * total - attended
    #         if shortfall <= 0:
    #             return 0
    #         return max(0, math.ceil(shortfall / (1 - frac)))

    #     def calc_can_miss(target_pct, attended, total, future):
    #         """How many of the remaining future lectures can the student miss and still end >= target?"""
    #         frac = target_pct / 100
    #         min_needed = max(0, math.ceil(frac * (total + future) - attended))
    #         return max(0, future - min_needed)

    #     # ── Compute for both targets ──
    #     thresh_needed   = calc_lectures_needed(THRESHOLD, attended, total_lectures)
    #     personal_needed = calc_lectures_needed(personal_target, attended, total_lectures)

    #     thresh_safe   = current_pct >= THRESHOLD
    #     personal_safe = current_pct >= personal_target

    #     thresh_possible   = thresh_needed <= future_lectures
    #     personal_possible = personal_needed <= future_lectures

    #     # ── SECTION A: Mandatory 75% status ──────────────────────────────────────
    #     st.markdown(f"""
    #     <div style='font-size:0.72rem; font-weight:700; letter-spacing:2px; text-transform:uppercase;
    #                 color:#facc15; margin-bottom:8px;'>① Mandatory Threshold — {THRESHOLD}%</div>
    #     """, unsafe_allow_html=True)

    #     if thresh_safe:
    #         thresh_can_miss = calc_can_miss(THRESHOLD, attended, total_lectures, future_lectures)
    #         thresh_min_future = max(0, future_lectures - thresh_can_miss)
    #         st.markdown(f"""
    #         <div class='edu-card' style='border-left:4px solid #34d399; padding:22px; margin-bottom:16px;'>
    #             <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;'>
    #                 <div>
    #                     <div style='font-size:2.2rem; font-weight:800; color:#34d399; font-family:"Space Mono",monospace; line-height:1;'>{current_pct:.1f}%</div>
    #                     <div style='color:#64748b; font-size:0.8rem; margin-top:2px;'>Current Attendance</div>
    #                 </div>
    #                 <span class='badge badge-green'>✅ NOT A DEFAULTER</span>
    #             </div>
    #             <div style='margin-top:16px; background:#13161e; border-radius:10px; padding:14px; text-align:center;'>
    #                 <div style='color:#64748b; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>Safety Buffer — Lectures You Can Still Miss</div>
    #                 <div style='font-size:2.8rem; font-weight:800; color:#34d399; font-family:"Space Mono",monospace; line-height:1;'>{thresh_can_miss}</div>
    #                 <div style='color:#64748b; font-size:0.78rem; margin-top:4px;'>Must attend at least <b style='color:#e2e8f0;'>{thresh_min_future}</b> of {future_lectures} remaining lectures</div>
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)
    #     elif not thresh_possible:
    #         st.markdown(f"""
    #         <div class='edu-card' style='border-left:4px solid #f87171; padding:22px; margin-bottom:16px;'>
    #             <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;'>
    #                 <div>
    #                     <div style='font-size:2.2rem; font-weight:800; color:#f87171; font-family:"Space Mono",monospace; line-height:1;'>{current_pct:.1f}%</div>
    #                     <div style='color:#64748b; font-size:0.8rem; margin-top:2px;'>Current Attendance</div>
    #                 </div>
    #                 <span class='badge badge-red'>🚨 RECOVERY IMPOSSIBLE</span>
    #             </div>
    #             <div style='margin-top:14px; display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
    #                 <div style='background:#13161e; border-radius:10px; padding:12px; text-align:center;'>
    #                     <div style='color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:1px;'>Lectures Required</div>
    #                     <div style='font-size:1.5rem; font-weight:800; color:#f87171;'>{thresh_needed}</div>
    #                 </div>
    #                 <div style='background:#13161e; border-radius:10px; padding:12px; text-align:center;'>
    #                     <div style='color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:1px;'>Lectures Available</div>
    #                     <div style='font-size:1.5rem; font-weight:800; color:#f87171;'>{future_lectures}</div>
    #                 </div>
    #             </div>
    #             <div style='margin-top:12px; color:#64748b; font-size:0.76rem;'>
    #                 Best case (attend all): <b style='color:#fb923c;'>{max_possible_att:.1f}%</b> — still below {THRESHOLD}%.
    #                 Shortfall of <b style='color:#f87171;'>{thresh_needed - future_lectures} lectures</b>.
    #                 Apply for condonation / medical exemption.
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)
    #     else:
    #         pct_after_thresh = (attended + thresh_needed) / (total_lectures + thresh_needed) * 100
    #         left_after_thresh = future_lectures - thresh_needed
    #         left_color = "#34d399" if left_after_thresh > 0 else "#fb923c"
    #         st.markdown(f"""
    #         <div class='edu-card' style='border-left:4px solid #fb923c; padding:22px; margin-bottom:16px;'>
    #             <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;'>
    #                 <div>
    #                     <div style='font-size:2.2rem; font-weight:800; color:#fb923c; font-family:"Space Mono",monospace; line-height:1;'>{current_pct:.1f}%</div>
    #                     <div style='color:#64748b; font-size:0.8rem; margin-top:2px;'>Current Attendance</div>
    #                 </div>
    #                 <span class='badge badge-orange'>⚠️ DEFAULTER — FIXABLE</span>
    #             </div>
    #             <div style='margin-top:14px; background:#13161e; border-radius:10px; padding:16px; text-align:center;'>
    #                 <div style='color:#64748b; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>Consecutive Lectures to Exit Defaulters List</div>
    #                 <div style='font-size:3.5rem; font-weight:800; color:#4f8ef7; font-family:"Space Mono",monospace; line-height:1;'>{thresh_needed}</div>
    #             </div>
    #             <div style='margin-top:10px; display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
    #                 <div style='background:#13161e; border-radius:10px; padding:12px; text-align:center;'>
    #                     <div style='color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:1px;'>Attendance After</div>
    #                     <div style='font-size:1.3rem; font-weight:700; color:#34d399;'>{pct_after_thresh:.1f}%</div>
    #                     <div style='color:#64748b; font-size:0.68rem;'>≥ {THRESHOLD}% ✓</div>
    #                 </div>
    #                 <div style='background:#13161e; border-radius:10px; padding:12px; text-align:center;'>
    #                     <div style='color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:1px;'>Lectures Left After</div>
    #                     <div style='font-size:1.3rem; font-weight:700; color:{left_color};'>{left_after_thresh}</div>
    #                     <div style='color:#64748b; font-size:0.68rem;'>of {future_lectures} remaining</div>
    #                 </div>
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)

    #     # ── SECTION B: Personal Target ────────────────────────────────────────────
    #     target_label_color = "#a78bfa" if personal_target >= THRESHOLD else "#f87171"
    #     st.markdown(f"""
    #     <div style='font-size:0.72rem; font-weight:700; letter-spacing:2px; text-transform:uppercase;
    #                 color:{target_label_color}; margin-bottom:8px; margin-top:4px;'>② Personal Target — {personal_target}%</div>
    #     """, unsafe_allow_html=True)

    #     if personal_safe:
    #         personal_can_miss = calc_can_miss(personal_target, attended, total_lectures, future_lectures)
    #         personal_min_future = max(0, future_lectures - personal_can_miss)
    #         st.markdown(f"""
    #         <div class='edu-card' style='border-left:4px solid #a78bfa; padding:22px;'>
    #             <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;'>
    #                 <div style='color:#a78bfa; font-size:1rem; font-weight:700;'>Already at {personal_target}% ✓</div>
    #                 <span class='badge badge-blue'>ON TARGET</span>
    #             </div>
    #             <div style='margin-top:12px; background:#13161e; border-radius:10px; padding:14px; text-align:center;'>
    #                 <div style='color:#64748b; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>Can Still Miss (staying above {personal_target}%)</div>
    #                 <div style='font-size:2.4rem; font-weight:800; color:#a78bfa; font-family:"Space Mono",monospace; line-height:1;'>{personal_can_miss}</div>
    #                 <div style='color:#64748b; font-size:0.75rem; margin-top:4px;'>Must attend at least <b style='color:#e2e8f0;'>{personal_min_future}</b> of {future_lectures} remaining</div>
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)
    #     elif not personal_possible:
    #         pct_best = max_possible_att
    #         st.markdown(f"""
    #         <div class='edu-card' style='border-left:4px solid #a78bfa; padding:22px;'>
    #             <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;'>
    #                 <div style='color:#a78bfa; font-size:1rem; font-weight:700;'>Target {personal_target}% not reachable this semester</div>
    #                 <span class='badge badge-blue'>OUT OF REACH</span>
    #             </div>
    #             <div style='margin-top:10px; color:#64748b; font-size:0.78rem;'>
    #                 Best case (attend all remaining): <b style='color:#fb923c;'>{pct_best:.1f}%</b>.
    #                 You need <b style='color:#a78bfa;'>{personal_needed}</b> lectures but only <b style='color:#e2e8f0;'>{future_lectures}</b> remain.
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)
    #     else:
    #         pct_after_personal = (attended + personal_needed) / (total_lectures + personal_needed) * 100
    #         left_after_personal = future_lectures - personal_needed
    #         left_color_p = "#34d399" if left_after_personal > 0 else "#fb923c"
    #         # Extra note if personal target < 75 (they'll hit their target but still be a defaulter)
    #         below_thresh_note = ""
    #         if personal_target < THRESHOLD:
    #             below_thresh_note = f"<div style='margin-top:10px; padding:8px 12px; background:#f8717120; border-radius:8px; color:#f87171; font-size:0.75rem;'>⚠️ Even after reaching {personal_target}%, you'll still be on the defaulters list. You need {thresh_needed} lectures to clear {THRESHOLD}%.</div>"
    #         st.markdown(f"""
    #         <div class='edu-card' style='border-left:4px solid #a78bfa; padding:22px;'>
    #             <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;'>
    #                 <div style='color:#a78bfa; font-size:1rem; font-weight:700;'>Target: {personal_target}%</div>
    #                 <span class='badge badge-blue'>REACHABLE</span>
    #             </div>
    #             <div style='margin-top:12px; background:#13161e; border-radius:10px; padding:14px; text-align:center;'>
    #                 <div style='color:#64748b; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>Lectures Needed for Personal Target</div>
    #                 <div style='font-size:3rem; font-weight:800; color:#a78bfa; font-family:"Space Mono",monospace; line-height:1;'>{personal_needed}</div>
    #             </div>
    #             <div style='margin-top:10px; display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
    #                 <div style='background:#13161e; border-radius:10px; padding:12px; text-align:center;'>
    #                     <div style='color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:1px;'>Attendance After</div>
    #                     <div style='font-size:1.3rem; font-weight:700; color:#a78bfa;'>{pct_after_personal:.1f}%</div>
    #                 </div>
    #                 <div style='background:#13161e; border-radius:10px; padding:12px; text-align:center;'>
    #                     <div style='color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:1px;'>Lectures Left After</div>
    #                     <div style='font-size:1.3rem; font-weight:700; color:{left_color_p};'>{left_after_personal}</div>
    #                 </div>
    #             </div>
    #             {below_thresh_note}
    #         </div>
    #         """, unsafe_allow_html=True)

    #     # ── Visual gauge: all three lines on one chart ──
    #     st.markdown(f"<div class='section-header' style='margin-top:20px;'>Attendance Progress</div>", unsafe_allow_html=True)

    #     rows = ['Current']
    #     values = [current_pct]
    #     bar_colors_gauge = ['#fb923c' if not thresh_safe else '#34d399']

    #     if not thresh_safe and thresh_possible:
    #         pct_after_thresh_gauge = (attended + thresh_needed) / (total_lectures + thresh_needed) * 100
    #         rows.append(f'After {thresh_needed} lectures (→{THRESHOLD}%)')
    #         values.append(pct_after_thresh_gauge)
    #         bar_colors_gauge.append('#4f8ef7')

    #     if personal_target != THRESHOLD and not personal_safe and personal_possible:
    #         pct_after_personal_gauge = (attended + personal_needed) / (total_lectures + personal_needed) * 100
    #         rows.append(f'After {personal_needed} lectures (→{personal_target}%)')
    #         values.append(pct_after_personal_gauge)
    #         bar_colors_gauge.append('#a78bfa')

    #     fig_p, ax_p = plt.subplots(figsize=(8, max(1.4, len(rows) * 0.8)), facecolor='#1a1e2b')
    #     ax_p.set_facecolor('#1a1e2b')
    #     for i, (row, val, col) in enumerate(zip(rows, values, bar_colors_gauge)):
    #         ax_p.barh([row], [val], color=col, height=0.45, label=f'{row}: {val:.1f}%')
    #         ax_p.barh([row], [100 - val], left=val, color='#252a3a', height=0.45)
    #         ax_p.text(val + 0.8, i, f'{val:.1f}%', va='center', color='#e2e8f0', fontsize=8)

    #     ax_p.axvline(THRESHOLD, color='#facc15', linewidth=2, linestyle='--', label=f'{THRESHOLD}% Institutional Min')
    #     if personal_target != THRESHOLD:
    #         ax_p.axvline(personal_target, color='#a78bfa', linewidth=1.5, linestyle=':', label=f'{personal_target}% Personal Target')
    #     ax_p.set_xlim(0, 105)
    #     ax_p.set_xlabel('Attendance %', color='#64748b', fontsize=9)
    #     ax_p.tick_params(colors='#64748b', labelsize=8)
    #     for spine in ax_p.spines.values(): spine.set_visible(False)
    #     ax_p.legend(facecolor='#252a3a', edgecolor='#252a3a', labelcolor='#e2e8f0', fontsize=7.5, loc='lower right')
    #     plt.tight_layout()
    #     st.pyplot(fig_p)
    #     plt.close()

    # ── Back button ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Main Page"):
        st.session_state['portal_view'] = False
        st.rerun()

    st.stop()   # Don't render rest of app


# ─── LOGIN PAGE ──────────────────────────────────────────────────────────────────
if not st.session_state['logged_in']:
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='edu-card' style='border: 1px solid #252a3a; padding: 40px;'>
            <div class='hero-title'>EduInsight AI</div>
            <div class='hero-sub'>Educator Portal — Secure Login</div>
            <hr style='margin: 24px 0; border-color:#252a3a;'>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("""<div class='edu-card' style='padding:32px;'>""", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Credentials</div>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔐 Sign In", use_container_width=True):
                if username == "admin" and password == "samsung123":
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = 'admin'
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align:center; color:#64748b; font-size:0.78rem; margin-top:12px;'>
            Demo credentials: admin / samsung123
            </div>
            """, unsafe_allow_html=True)


# ─── MAIN DASHBOARD ──────────────────────────────────────────────────────────────
else:
    # Load data & model
    @st.cache_data
    def load_data():
        return pd.read_csv('data/StudentPerformanceFactors.csv')

    @st.cache_resource
    def load_model():
        return joblib.load('model/model.pkl')

    df_raw = load_data()
    model  = load_model()

    # ── Header ──
    st.markdown("""
    <div class='hero-title'>EduInsight AI Dashboard</div>
    <div class='hero-sub'>AI-Powered Attendance & Performance Analytics Platform</div>
    <br>
    """, unsafe_allow_html=True)

    # ── Top KPIs ──
    avg_att   = df_raw['Attendance'].mean()
    avg_score = df_raw['Exam_Score'].mean()
    corr_val  = df_raw[['Attendance', 'Exam_Score']].corr().iloc[0, 1]
    total_stu = len(df_raw)
    at_risk   = len(df_raw[df_raw['Attendance'] < 75])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📊 Total Students",   f"{total_stu:,}")
    k2.metric("📅 Avg Attendance",   f"{avg_att:.1f}%")
    k3.metric("🏆 Avg Exam Score",   f"{avg_score:.1f}")
    k4.metric("🔗 Correlation",      f"{corr_val:.3f}")
    k5.metric("🚨 At-Risk (<75%)",   f"{at_risk:,}", delta=f"-{at_risk/total_stu*100:.1f}% of class", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["📈 Correlation Analysis", "🤖 Grade Estimator", "📊 Data Explorer"])


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — CORRELATION ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.4, 1], gap="large")

        with c1:
            st.markdown("<div class='section-header'>Attendance vs Exam Score — Trend Line</div>", unsafe_allow_html=True)
            sample = df_raw.sample(800, random_state=42)

            fig, ax = plt.subplots(figsize=(9, 5), facecolor='#1a1e2b')
            ax.set_facecolor('#1a1e2b')

            ax.scatter(sample['Attendance'], sample['Exam_Score'],
                       alpha=0.25, s=18, color='#4f8ef7', linewidths=0)

            # Trend line
            z = np.polyfit(sample['Attendance'], sample['Exam_Score'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sample['Attendance'].min(), sample['Attendance'].max(), 200)
            ax.plot(x_line, p(x_line), color='#a78bfa', linewidth=2.5, label='Trend Line')

            # 75% threshold line
            ax.axvline(75, color='#f87171', linewidth=1.5, linestyle='--', alpha=0.7, label='75% Attendance Threshold')

            ax.set_xlabel('Attendance (%)', color='#64748b', fontsize=11)
            ax.set_ylabel('Exam Score', color='#64748b', fontsize=11)
            ax.set_title('', color='#e2e8f0')
            ax.tick_params(colors='#64748b')
            for spine in ax.spines.values(): spine.set_edgecolor('#252a3a')
            ax.legend(facecolor='#252a3a', edgecolor='#252a3a', labelcolor='#e2e8f0', fontsize=10)
            ax.grid(True, alpha=0.08, color='white')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown("<div class='section-header'>Statistical Summary</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='edu-card edu-card-accent'>
                <div class='section-header'>Correlation</div>
                <div class='score-ring' style='color:#4f8ef7;'>{corr_val:.3f}</div>
                <div style='text-align:center; color:#64748b; font-size:0.8rem; margin-top:8px;'>
                    Moderate–Strong positive correlation
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Attendance buckets
            bins_label = ['<60%', '60–75%', '75–85%', '85–100%']
            bins_cut   = pd.cut(df_raw['Attendance'], bins=[0, 60, 75, 85, 100])
            group_means = df_raw.groupby(bins_cut, observed=True)['Exam_Score'].mean().values

            st.markdown("<div class='section-header' style='margin-top:16px;'>Avg Score by Attendance Band</div>", unsafe_allow_html=True)
            colors_bar = ['#f87171', '#fb923c', '#facc15', '#34d399']
            fig2, ax2 = plt.subplots(figsize=(5, 2.8), facecolor='#1a1e2b')
            ax2.set_facecolor('#1a1e2b')
            bars = ax2.bar(bins_label, group_means, color=colors_bar, width=0.55, edgecolor='none', zorder=3)
            for bar, val in zip(bars, group_means):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f'{val:.1f}', ha='center', va='bottom', color='#e2e8f0', fontsize=9)
            ax2.set_ylabel('Avg Score', color='#64748b', fontsize=9)
            ax2.tick_params(colors='#64748b', labelsize=8)
            for spine in ax2.spines.values(): spine.set_edgecolor('#252a3a')
            ax2.grid(axis='y', alpha=0.08, color='white', zorder=0)
            ax2.set_ylim(0, max(group_means) + 8)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Distribution Analysis</div>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)

        with d1:
            fig3, ax3 = plt.subplots(figsize=(4.5, 3), facecolor='#1a1e2b')
            ax3.set_facecolor('#1a1e2b')
            ax3.hist(df_raw['Attendance'], bins=30, color='#4f8ef7', alpha=0.8, edgecolor='none')
            ax3.axvline(75, color='#f87171', linewidth=1.5, linestyle='--', label='75% line')
            ax3.set_title('Attendance Distribution', color='#e2e8f0', fontsize=11)
            ax3.tick_params(colors='#64748b', labelsize=8)
            for spine in ax3.spines.values(): spine.set_edgecolor('#252a3a')
            ax3.grid(alpha=0.07, color='white')
            ax3.legend(facecolor='#252a3a', edgecolor='#252a3a', labelcolor='#e2e8f0', fontsize=8)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

        with d2:
            fig4, ax4 = plt.subplots(figsize=(4.5, 3), facecolor='#1a1e2b')
            ax4.set_facecolor('#1a1e2b')
            ax4.hist(df_raw['Exam_Score'], bins=30, color='#a78bfa', alpha=0.8, edgecolor='none')
            ax4.set_title('Exam Score Distribution', color='#e2e8f0', fontsize=11)
            ax4.tick_params(colors='#64748b', labelsize=8)
            for spine in ax4.spines.values(): spine.set_edgecolor('#252a3a')
            ax4.grid(alpha=0.07, color='white')
            plt.tight_layout(); st.pyplot(fig4); plt.close()

        # with d3:
        #     fig5, ax5 = plt.subplots(figsize=(4.5, 3), facecolor='#1a1e2b')
        #     ax5.set_facecolor('#1a1e2b')
        #     mot_map_r = {0: 'Low', 1: 'Medium', 2: 'High'}
        #     from sklearn.preprocessing import LabelEncoder
        #     le = LabelEncoder()
        #     df_raw['Motivation_enc'] = le.fit_transform(df_raw['Motivation_Level'].astype(str))
        #     for m_enc, m_label, color in [(0, 'Low', '#f87171'), (1, 'Medium', '#fb923c'), (2, 'High', '#34d399')]:
        #         subset = df_raw[df_raw['Motivation_enc'] == m_enc]['Exam_Score']
        #         if len(subset) > 0:
        #             ax5.hist(subset, bins=20, alpha=0.6, label=m_label, color=color, edgecolor='none')
        #     ax5.set_title('Score by Motivation Level', color='#e2e8f0', fontsize=11)
        #     ax5.tick_params(colors='#64748b', labelsize=8)
        #     for spine in ax5.spines.values(): spine.set_edgecolor('#252a3a')
        #     ax5.grid(alpha=0.07, color='white')
        #     ax5.legend(facecolor='#252a3a', edgecolor='#252a3a', labelcolor='#e2e8f0', fontsize=8)
        #     plt.tight_layout(); st.pyplot(fig5); plt.close()


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — GRADE ESTIMATOR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        col_in, col_out = st.columns([1, 1.3], gap="large")

        with col_in:
            st.markdown("<div class='edu-card' style='padding:28px;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Student Parameters</div>", unsafe_allow_html=True)

            att  = st.slider("📅 Attendance %",         0, 100, 85,
                             help="Percentage of classes attended")
            hrs  = st.slider("📚 Hours Studied / Week", 0, 50,  20,
                             help="Average weekly study hours")
            prev = st.slider("📝 Previous Score",       0, 100, 70,
                             help="Last exam score")
            tut  = st.number_input("🧑‍🏫 Tutoring Sessions", 0, 10, 2,
                                   help="Number of tutoring sessions attended")

            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("🔍 Predict Grade", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_out:
            if 'prediction_result' not in st.session_state:
                st.session_state['prediction_result'] = None

            if run:
                input_data = np.array([[att, hrs, prev, tut]])
                prediction = model.predict(input_data)[0]
                st.session_state['prediction_result'] = {
                    'prediction': prediction,
                    'att': att, 'hrs': hrs, 'prev': prev, 'tut': tut
                }

            if st.session_state['prediction_result'] is None:
                st.markdown("""
                <div class='edu-card' style='padding:48px; text-align:center;'>
                    <div style='font-size:3rem; margin-bottom:12px;'>🎓</div>
                    <div style='color:#64748b; font-size:1rem; font-weight:600;'>
                        Set student parameters and click<br>
                        <span style='color:#4f8ef7;'>Predict Grade</span> to see results
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                res        = st.session_state['prediction_result']
                prediction = res['prediction']
                att_r      = res['att']
                hrs_r      = res['hrs']
                tut_r      = res['tut']

                # Risk level
                if att_r < 60:
                    risk = "Critical";        badge_cls = "badge-red";    risk_color = "#f87171"
                elif att_r < 75:
                    risk = "At Risk";         badge_cls = "badge-orange"; risk_color = "#fb923c"
                elif prediction < 55:
                    risk = "Needs Attention"; badge_cls = "badge-orange"; risk_color = "#fb923c"
                else:
                    risk = "On Track";        badge_cls = "badge-green";  risk_color = "#34d399"

                # Grade letter
                if   prediction >= 90: grade = "A+"
                elif prediction >= 80: grade = "A"
                elif prediction >= 70: grade = "B"
                elif prediction >= 60: grade = "C"
                elif prediction >= 50: grade = "D"
                else:                  grade = "F"

                st.markdown(f"""
                <div class='edu-card' style='padding:32px; text-align:center;'>
                    <div class='section-header'>Predicted Outcome</div>
                    <div class='score-ring' style='color:{risk_color}; font-size:4rem;'>{prediction:.1f}</div>
                    <div style='color:#64748b; font-size:0.85rem; margin:4px 0 12px;'>Estimated Exam Score</div>
                    <div style='display:flex; justify-content:center; gap:10px; flex-wrap:wrap;'>
                        <span class='badge badge-blue'>Grade: {grade}</span>
                        <span class='badge {badge_cls}'>{risk}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Progress bar visual
                st.markdown("<div class='section-header' style='margin-top:20px;'>Score Breakdown</div>", unsafe_allow_html=True)
                fig6, ax6 = plt.subplots(figsize=(6, 1.5), facecolor='#1a1e2b')
                ax6.set_facecolor('#1a1e2b')
                ax6.barh(['Predicted Score'], [prediction], color=risk_color, height=0.4)
                ax6.barh(['Predicted Score'], [100 - prediction], left=prediction, color='#252a3a', height=0.4)
                ax6.set_xlim(0, 100)
                ax6.axvline(75, color='#facc15', linewidth=1.5, linestyle='--', alpha=0.8)
                ax6.tick_params(colors='#64748b', labelsize=9)
                ax6.set_xlabel('Score', color='#64748b', fontsize=9)
                for spine in ax6.spines.values(): spine.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig6)
                plt.close()

                # Intervention panel
                st.markdown("<div class='section-header' style='margin-top:20px;'>🎯 Strategic Intervention</div>", unsafe_allow_html=True)

                if att_r < 60:
                    st.error(f"""
**🚨 CRITICAL — Attendance Crisis**

Attendance of **{att_r}%** is severely below the minimum threshold — immediate action required.

**Action Task 1:** Initiate parental/guardian contact within 24 hours. \\
**Action Task 2:** Issue a formal Attendance Warning Letter and log it in the student record.\\
**Action Task 3:** Refer to school counsellor for root-cause assessment (transport, health, family).\\
**Action Task 4:** Set up a weekly attendance check-in with the form tutor.\\

Without urgent attendance recovery, academic continuation is at serious risk.
                    """)
                elif att_r < 75:
                    st.warning(f"""
**⚠️ AT RISK — Attendance Recovery Required**

Attendance of **{att_r}%** is below the mandatory 75% threshold.

**Action Task 1:** Schedule a 1-on-1 support meeting this week — identify barriers (transport, health, motivation).
**Action Task 2:** Create a personalised Attendance Improvement Plan (AIP) with weekly targets.
**Action Task 3:** Assign a peer study buddy to encourage daily attendance.
**Action Task 4:** Monitor for the next 3 weeks and escalate if no improvement is seen.

📈 A 10% attendance boost is projected to improve the exam score by ~5 marks.
                    """)
                elif hrs_r < 10:
                    st.warning(f"""
**🟠 LOW STUDY HOURS — Discipline Intervention**

Only **{hrs_r} hrs/week** of study detected — well below the recommended 15–20 hrs.

**Action Task 1:** Recommend a structured weekly timetable; share a template.
**Action Task 2:** Introduce an accountability partner or group study session.
**Action Task 3:** Suggest productivity tools (Pomodoro, flashcards, past papers).

📈 Adding 5 more hrs/week could yield a **+8 mark** improvement on current trajectory.
                    """)
                elif prediction < 70:
                    st.warning(f"""
**🔵 BELOW TARGET — Academic Support Needed**

Predicted score of **{prediction:.1f}** is below the 70-mark target.

**Action Task 1:** Provide targeted revision packs for weak topic areas.
**Action Task 2:** Enrol in 1–2 additional tutoring sessions per week.
**Action Task 3:** Set a mock exam in 3 weeks to track improvement.
                    """)
                else:
                    st.success(f"""
**🌟 ON TRACK — Enrichment Mode**

Excellent indicators across attendance, study hours, and predicted grade (**{prediction:.1f}**).

**Action Task 1:** Introduce advanced challenge material or competitive exam preparation.
**Action Task 2:** Nominate for leadership, mentoring, or academic excellence programs.
**Action Task 3:** Maintain momentum — schedule a mid-term check-in to sustain performance.
                    """)

                # What-If Scenarios
                st.markdown("<div class='section-header' style='margin-top:20px;'>📐 What-If Scenarios</div>", unsafe_allow_html=True)
                scenarios = {
                    "Current":          model.predict([[att_r,              hrs_r,              res['prev'], tut_r]])[0],
                    "+10% Attendance":  model.predict([[min(att_r+10, 100), hrs_r,              res['prev'], tut_r]])[0],
                    "+5 Study Hours":   model.predict([[att_r,              min(hrs_r+5, 50),   res['prev'], tut_r]])[0],
                    "+1 Tutoring Ses":  model.predict([[att_r,              hrs_r,              res['prev'], min(tut_r+1, 10)]])[0],
                }
                fig7, ax7 = plt.subplots(figsize=(6, 2.5), facecolor='#1a1e2b')
                ax7.set_facecolor('#1a1e2b')
                s_labels = list(scenarios.keys())
                s_values = list(scenarios.values())
                bars = ax7.barh(s_labels, s_values, color=['#4f8ef7','#34d399','#a78bfa','#fb923c'], height=0.5, edgecolor='none')
                for bar, val in zip(bars, s_values):
                    ax7.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}', va='center', color='#e2e8f0', fontsize=9)
                ax7.set_xlim(0, 110)
                ax7.tick_params(colors='#64748b', labelsize=9)
                ax7.set_xlabel('Predicted Score', color='#64748b', fontsize=9)
                for spine in ax7.spines.values(): spine.set_visible(False)
                ax7.grid(axis='x', alpha=0.07, color='white')
                plt.tight_layout(); st.pyplot(fig7); plt.close()


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3 — DATA EXPLORER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        e1, e2 = st.columns([1, 1], gap="large")

        with e1:
            st.markdown("<div class='section-header'>Feature Correlation Heatmap</div>", unsafe_allow_html=True)
            num_cols = ['Attendance', 'Hours_Studied', 'Previous_Scores',
                        'Tutoring_Sessions', 'Sleep_Hours', 'Exam_Score']
            corr_matrix = df_raw[[c for c in num_cols if c in df_raw.columns]].corr()

            fig8, ax8 = plt.subplots(figsize=(6, 5), facecolor='#1a1e2b')
            ax8.set_facecolor('#1a1e2b')
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        ax=ax8, linewidths=0.5, linecolor='#13161e',
                        annot_kws={'size': 9, 'color': 'white'},
                        cbar_kws={'shrink': 0.8})
            ax8.tick_params(colors='#e2e8f0', labelsize=9)
            ax8.set_title('Numeric Feature Correlations', color='#e2e8f0', pad=12)
            plt.tight_layout(); st.pyplot(fig8); plt.close()

        with e2:
            st.markdown("<div class='section-header'>Top & Bottom Performers</div>", unsafe_allow_html=True)
            top_n = st.selectbox("Show Top/Bottom N students", [5, 10, 20], index=0)

            # Name first if available, then key performance columns
            perf_cols = ['Name'] if 'Name' in df_raw.columns else []
            perf_cols += ['Attendance', 'Hours_Studied', 'Previous_Scores',
                          'Tutoring_Sessions', 'Motivation_Level', 'Exam_Score']
            display_cols = [c for c in perf_cols if c in df_raw.columns]

            st.markdown("**🏆 Top Performers**")
            st.dataframe(
                df_raw.nlargest(top_n, 'Exam_Score')[display_cols].reset_index(drop=True),
                use_container_width=True, height=min(top_n * 38 + 38, 280)
            )
            st.markdown("**⚠️ Students Needing Support**")
            st.dataframe(
                df_raw.nsmallest(top_n, 'Exam_Score')[display_cols].reset_index(drop=True),
                use_container_width=True, height=min(top_n * 38 + 38, 280)
            )

        st.markdown("<div class='section-header' style='margin-top:20px;'>Full Dataset Preview</div>", unsafe_allow_html=True)
        ds1, ds2 = st.columns([1, 1], gap="medium")
        with ds1:
            search_att = st.slider("Filter by minimum attendance", 0, 100, 0)
        with ds2:
            name_search_explorer = st.text_input("Search by name", placeholder="e.g. Deepak",
                                                  help="Partial name match, case-insensitive")
        filtered_df = df_raw[df_raw['Attendance'] >= search_att].copy()
        if name_search_explorer.strip():
            if 'Name' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Name'].str.contains(name_search_explorer.strip(), case=False, na=False)]
            else:
                st.warning("Name column not found in dataset.")
        # Show Name as first column if it exists
        explorer_cols = (['Name'] if 'Name' in filtered_df.columns else []) +                         [c for c in filtered_df.columns if c != 'Name']
        st.info(f"Showing {len(filtered_df):,} students with attendance ≥ {search_att}%"
                + (f"  |  matching \"{name_search_explorer.strip()}\"" if name_search_explorer.strip() else ""))
        st.dataframe(filtered_df[explorer_cols].head(200), use_container_width=True, height=350)