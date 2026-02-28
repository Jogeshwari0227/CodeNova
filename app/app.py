import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False


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

    st.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='edu-card' style='padding:14px; margin-bottom:12px;'>
        <div style='font-size:0.85rem; color:#64748b;'>Problem Statement</div>
        <div style='font-size:0.95rem; font-weight:600; margin-top:4px;'>PS 05 — Attendance & Performance Correlation</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state['logged_in']:
        st.markdown("<div class='section-header' style='margin-top:20px;'>Account</div>", unsafe_allow_html=True)
        st.markdown("<div class='badge badge-green'>● Logged In as Admin</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='color:#64748b; font-size:0.75rem; line-height:1.6;'>
    Built with RandomForest ML<br>
    Dataset: 6,607 student records<br>
    Features: Attendance, Study Hours,<br>
    &nbsp;&nbsp;Previous Scores, Motivation
    </div>
    """, unsafe_allow_html=True)


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
                <div class='section-header'>Pearson Correlation</div>
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

        with d3:
            fig5, ax5 = plt.subplots(figsize=(4.5, 3), facecolor='#1a1e2b')
            ax5.set_facecolor('#1a1e2b')
            mot_map_r = {0: 'Low', 1: 'Medium', 2: 'High'}
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_raw['Motivation_enc'] = le.fit_transform(df_raw['Motivation_Level'].astype(str))
            for m_enc, m_label, color in [(0, 'Low', '#f87171'), (1, 'Medium', '#fb923c'), (2, 'High', '#34d399')]:
                subset = df_raw[df_raw['Motivation_enc'] == m_enc]['Exam_Score']
                if len(subset) > 0:
                    ax5.hist(subset, bins=20, alpha=0.6, label=m_label, color=color, edgecolor='none')
            ax5.set_title('Score by Motivation Level', color='#e2e8f0', fontsize=11)
            ax5.tick_params(colors='#64748b', labelsize=8)
            for spine in ax5.spines.values(): spine.set_edgecolor('#252a3a')
            ax5.grid(alpha=0.07, color='white')
            ax5.legend(facecolor='#252a3a', edgecolor='#252a3a', labelcolor='#e2e8f0', fontsize=8)
            plt.tight_layout(); st.pyplot(fig5); plt.close()


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
                    st.error("""
                    **🚨 CRITICAL — Attendance Crisis**

                    This student is severely below the minimum attendance threshold.
                    Immediate parental contact and counselling referral is strongly advised.
                    Without attendance recovery, academic continuation is at serious risk.
                    """)
                elif att_r < 75:
                    st.warning(f"""
                    **⚠️ AT RISK — Attendance Recovery Required**

                    Attendance of {att_r}% is below the 75% minimum.
                    **Action:** Schedule a 1-on-1 meeting. Identify root causes (transport, health, motivation).
                    Even a 10% attendance boost is projected to increase the exam score by ~5 marks.
                    """)
                elif hrs_r < 10:
                    st.warning(f"""
                    **🟠 LOW STUDY HOURS — Discipline Intervention**

                    Only {hrs_r} hours/week of study detected.
                    **Action:** Recommend a structured weekly timetable.
                    Adding just 5 more hours/week could yield a **+8 mark** improvement.
                    """)
                else:
                    st.success(f"""
                    **🌟 ON TRACK — Enrichment Mode**

                    This student shows strong behavioral indicators.
                    **Action:** Introduce advanced challenge material, competitive programs, or leadership opportunities to maximize potential.
                    """)

                # What-if scenario
                st.markdown("<div class='section-header' style='margin-top:20px;'>📐 What-If Scenarios</div>", unsafe_allow_html=True)
                scenarios = {
                    "Current":          model.predict([[att_r,              hrs_r,          res['prev'], tut_r]])[0],
                    "+10% Attendance":  model.predict([[min(att_r+10, 100), hrs_r,          res['prev'], tut_r]])[0],
                    "+5 Study Hours":   model.predict([[att_r,              min(hrs_r+5,50),res['prev'], tut_r]])[0],
                    "+1 Tutoring":      model.predict([[att_r,              hrs_r,          res['prev'], min(tut_r+1,10)]])[0],
                }
                fig7, ax7 = plt.subplots(figsize=(6, 2.5), facecolor='#1a1e2b')
                ax7.set_facecolor('#1a1e2b')
                s_labels = list(scenarios.keys())
                s_values = list(scenarios.values())
                bar_colors = ['#4f8ef7', '#34d399', '#a78bfa', '#fb923c']
                bars = ax7.barh(s_labels, s_values, color=bar_colors, height=0.5, edgecolor='none')
                for bar, val in zip(bars, s_values):
                    ax7.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                             f'{val:.1f}', va='center', color='#e2e8f0', fontsize=9)
                ax7.set_xlim(0, 110)
                ax7.tick_params(colors='#64748b', labelsize=9)
                ax7.set_xlabel('Predicted Score', color='#64748b', fontsize=9)
                for spine in ax7.spines.values(): spine.set_visible(False)
                ax7.grid(axis='x', alpha=0.07, color='white')
                plt.tight_layout()
                st.pyplot(fig7)
                plt.close()


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

            display_cols = ['Attendance', 'Hours_Studied', 'Previous_Scores',
                            'Tutoring_Sessions', 'Motivation_Level', 'Exam_Score']
            display_cols = [c for c in display_cols if c in df_raw.columns]

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
        search_att = st.slider("Filter by minimum attendance", 0, 100, 0)
        filtered_df = df_raw[df_raw['Attendance'] >= search_att]
        st.info(f"Showing {len(filtered_df):,} students with attendance ≥ {search_att}%")
        st.dataframe(filtered_df.head(100), use_container_width=True, height=350)