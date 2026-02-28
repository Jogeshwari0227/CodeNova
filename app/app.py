import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
:root {
    --bg:      #0d0f14;
    --panel:   #13161e;
    --card:    #1a1e2b;
    --border:  #252a3a;
    --accent:  #4f8ef7;
    --accent2: #a78bfa;
    --green:   #34d399;
    --orange:  #fb923c;
    --red:     #f87171;
    --yellow:  #facc15;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --mono:    'Space Mono', monospace;
    --sans:    'Syne', sans-serif;
}
.stApp { background: var(--bg) !important; }
html, body, [class*="css"] { font-family: var(--sans) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--panel) !important; border-radius: 12px !important; padding: 4px !important; gap: 4px !important; border: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; border-radius: 8px !important; font-family: var(--sans) !important; font-weight: 600 !important; padding: 8px 20px !important; transition: all .2s !important; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: white !important; }
.stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text) !important; }
.stButton > button { background: var(--accent) !important; color: white !important; border: none !important; border-radius: 10px !important; font-family: var(--sans) !important; font-weight: 700 !important; padding: 10px 28px !important; transition: opacity .2s !important; }
.stButton > button:hover { opacity: 0.85 !important; }
[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; padding: 20px !important; }
[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 12px !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--mono) !important; font-size: 2rem !important; }
.edu-card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 24px; margin-bottom: 16px; }
.edu-card-accent { border-left: 4px solid var(--accent); }
.edu-card-green  { border-left: 4px solid var(--green); }
.edu-card-red    { border-left: 4px solid var(--red); }
.badge { display: inline-block; padding: 4px 12px; border-radius: 100px; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
.badge-green  { background: rgba(52,211,153,.15); color: var(--green);  border: 1px solid rgba(52,211,153,.3); }
.badge-orange { background: rgba(251,146,60,.15);  color: var(--orange); border: 1px solid rgba(251,146,60,.3); }
.badge-red    { background: rgba(248,113,113,.15); color: var(--red);    border: 1px solid rgba(248,113,113,.3); }
.badge-blue   { background: rgba(79,142,247,.15);  color: var(--accent); border: 1px solid rgba(79,142,247,.3); }
.badge-purple { background: rgba(167,139,250,.15); color: var(--accent2);border: 1px solid rgba(167,139,250,.3); }
.score-ring { font-family: var(--mono); font-size: 3.5rem; font-weight: 700; text-align: center; line-height: 1; }
.section-header { font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }
.hero-title { font-size: 2.6rem; font-weight: 800; background: linear-gradient(135deg, #4f8ef7 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.1; }
.hero-sub { color: var(--muted); font-size: 0.95rem; margin-top: 6px; }
hr { border-color: var(--border) !important; }
figure { background: transparent !important; }
.stAlert { border-radius: 12px !important; }
/* Landing page role cards */
.role-card { background: var(--card); border: 1px solid var(--border); border-radius: 20px; padding: 36px 28px; text-align: center; transition: border-color .2s; cursor: pointer; }
.role-card:hover { border-color: var(--accent); }
.role-icon { font-size: 3rem; margin-bottom: 12px; }
.role-title { font-size: 1.3rem; font-weight: 800; margin-bottom: 6px; }
.role-desc { color: var(--muted); font-size: 0.85rem; line-height: 1.5; }

</style>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────────────────────────
for key, default in [('role', None), ('edu_logged_in', False), ('prediction_result', None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Data / Model Loaders ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/StudentPerformanceFactors.csv')

@st.cache_resource
def load_model():
    return joblib.load('model/model.pkl')



# ════════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ════════════════════════════════════════════════════════════════════════════════
def show_landing():
    # Sidebar branding only
    with st.sidebar:
        st.markdown("""
        <div style='padding: 8px 0 20px 0;'>
            <div style='font-size:1.6rem; font-weight:800;'>🎓 EduInsight <span style='color:#4f8ef7;'>AI</span></div>
            <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>Attendance & Performance Platform</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div style='color:#64748b; font-size:0.75rem; line-height:1.8;'>
        PS 05 — Attendance &<br>Performance Correlation<br><br>
        🗄️ 6,607 student records<br>
        🤖 Random Forest Regressor<br>
        📐 4 predictive features
        </div>
        """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; margin-bottom:40px;'>
            <div class='hero-title' style='font-size:3rem;'>EduInsight AI</div>
            <div class='hero-sub' style='font-size:1.05rem;'>
                AI-Powered Attendance & Performance Analytics Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='text-align:center; margin-bottom:20px;'>Select Your Portal</div>", unsafe_allow_html=True)

        rc1, rc2 = st.columns(2, gap="large")
        with rc1:
            st.markdown("""
            <div class='role-card edu-card-accent'>
                <div class='role-icon'>👨‍🏫</div>
                <div class='role-title'>Educator Portal</div>
                <div class='role-desc'>Access AI grade predictions, statistical correlations, trend analysis, and strategic intervention tools.</div>
                <br>
                <span class='badge badge-blue'>🔐 Secured Access</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Educator Login →", use_container_width=True, key="goto_edu"):
                st.session_state['role'] = 'educator'
                st.rerun()

        with rc2:
            st.markdown("""
            <div class='role-card edu-card-green'>
                <div class='role-icon'>🎓</div>
                <div class='role-title'>Student Portal</div>
                <div class='role-desc'>View the Merit Leaderboard, check your attendance standing, and explore the Defaulter List.</div>
                <br>
                <span class='badge badge-green'>✅ Public Access</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Student Login →", use_container_width=True, key="goto_stu"):
                st.session_state['role'] = 'student'
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; color:#64748b; font-size:0.78rem;'>
        Student portal requires no credentials · Educator portal: admin / samsung123
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# EDUCATOR LOGIN PAGE
# ════════════════════════════════════════════════════════════════════════════════
def show_educator_login():
    with st.sidebar:
        st.markdown("""
        <div style='padding: 8px 0 20px 0;'>
            <div style='font-size:1.6rem; font-weight:800;'>🎓 EduInsight <span style='color:#4f8ef7;'>AI</span></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("← Back to Landing", use_container_width=True):
            st.session_state['role'] = None
            st.rerun()

    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='edu-card' style='border-top: 3px solid #4f8ef7; padding:40px; text-align:center;'>
            <div style='font-size:2.5rem; margin-bottom:8px;'>👨‍🏫</div>
            <div class='hero-title' style='font-size:1.8rem;'>Educator Login</div>
            <div class='hero-sub'>Restricted — Authorized Personnel Only</div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='edu-card' style='padding:32px;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Credentials</div>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔐 Sign In", use_container_width=True):
                if username == "admin" and password == "samsung123":
                    st.session_state['edu_logged_in'] = True
                    st.success("✅ Login successful! Loading dashboard...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align:center; color:#64748b; font-size:0.78rem; margin-top:10px;'>
            Demo: admin / samsung123
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# EDUCATOR DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
def show_educator_dashboard():
    df_raw = load_data()
    model  = load_model()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='padding: 8px 0 20px 0;'>
            <div style='font-size:1.6rem; font-weight:800;'>🎓 EduInsight <span style='color:#4f8ef7;'>AI</span></div>
            <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>Educator Dashboard</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div class='badge badge-blue' style='margin-bottom:12px;'>● Educator Session Active</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div style='color:#64748b; font-size:0.75rem; line-height:1.8;'>
        PS 05 — Attendance &<br>Performance Correlation<br><br>
        🗄️ 6,607 student records<br>
        🤖 Random Forest Regressor<br>
        📐 4 predictive features
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['edu_logged_in'] = False
            st.session_state['role'] = None
            st.session_state['prediction_result'] = None
            st.rerun()
        if st.button("← Back to Landing", use_container_width=True):
            st.session_state['edu_logged_in'] = False
            st.session_state['role'] = None
            st.rerun()

    # Header
    st.markdown("""
    <div class='hero-title'>Educator Dashboard</div>
    <div class='hero-sub'>AI-Powered Analytics & Strategic Intervention Platform · Confidential</div>
    <br>
    """, unsafe_allow_html=True)

    # KPIs
    avg_att   = df_raw['Attendance'].mean()
    avg_score = df_raw['Exam_Score'].mean()
    corr_val  = df_raw[['Attendance', 'Exam_Score']].corr().iloc[0, 1]
    total_stu = len(df_raw)
    at_risk   = len(df_raw[df_raw['Attendance'] < 75])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📊 Total Students",  f"{total_stu:,}")
    k2.metric("📅 Avg Attendance",  f"{avg_att:.1f}%")
    k3.metric("🏆 Avg Exam Score",  f"{avg_score:.1f}")
    k4.metric("🔗 Att-Score Corr.", f"{corr_val:.3f}")
    k5.metric("🚨 At-Risk (<75%)",  f"{at_risk:,}", delta=f"-{at_risk/total_stu*100:.1f}%", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈 Correlation Analysis", "🤖 Grade Estimator", "📊 Data Explorer"])

    # ── TAB 1: CORRELATION ──
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.4, 1], gap="large")
        with c1:
            st.markdown("<div class='section-header'>Attendance vs Exam Score — Trend Line</div>", unsafe_allow_html=True)
            sample = df_raw.sample(800, random_state=42)
            fig, ax = plt.subplots(figsize=(9, 5), facecolor='#1a1e2b')
            ax.set_facecolor('#1a1e2b')
            ax.scatter(sample['Attendance'], sample['Exam_Score'], alpha=0.25, s=18, color='#4f8ef7', linewidths=0)
            z = np.polyfit(sample['Attendance'], sample['Exam_Score'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sample['Attendance'].min(), sample['Attendance'].max(), 200)
            ax.plot(x_line, p(x_line), color='#a78bfa', linewidth=2.5, label='Trend Line')
            ax.axvline(75, color='#f87171', linewidth=1.5, linestyle='--', alpha=0.7, label='75% Threshold')
            ax.set_xlabel('Attendance (%)', color='#64748b', fontsize=11)
            ax.set_ylabel('Exam Score', color='#64748b', fontsize=11)
            ax.tick_params(colors='#64748b')
            for spine in ax.spines.values(): spine.set_edgecolor('#252a3a')
            ax.legend(facecolor='#252a3a', edgecolor='#252a3a', labelcolor='#e2e8f0', fontsize=10)
            ax.grid(True, alpha=0.08, color='white')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            st.markdown("<div class='section-header'>Statistical Summary</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='edu-card edu-card-accent'>
                <div class='section-header'>Pearson Correlation</div>
                <div class='score-ring' style='color:#4f8ef7;'>{corr_val:.3f}</div>
                <div style='text-align:center; color:#64748b; font-size:0.8rem; margin-top:8px;'>Moderate–Strong positive correlation</div>
            </div>
            """, unsafe_allow_html=True)
            bins_cut = pd.cut(df_raw['Attendance'], bins=[0, 60, 75, 85, 100])
            group_means = df_raw.groupby(bins_cut, observed=True)['Exam_Score'].mean().values
            bins_label = ['<60%', '60–75%', '75–85%', '85–100%']
            st.markdown("<div class='section-header' style='margin-top:16px;'>Avg Score by Attendance Band</div>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(5, 2.8), facecolor='#1a1e2b')
            ax2.set_facecolor('#1a1e2b')
            bars = ax2.bar(bins_label, group_means, color=['#f87171','#fb923c','#facc15','#34d399'], width=0.55, edgecolor='none', zorder=3)
            for bar, val in zip(bars, group_means):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.1f}', ha='center', va='bottom', color='#e2e8f0', fontsize=9)
            ax2.set_ylabel('Avg Score', color='#64748b', fontsize=9)
            ax2.tick_params(colors='#64748b', labelsize=8)
            for spine in ax2.spines.values(): spine.set_edgecolor('#252a3a')
            ax2.grid(axis='y', alpha=0.08, color='white', zorder=0)
            ax2.set_ylim(0, max(group_means) + 8)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

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
            hrs_corr = df_raw.sample(800, random_state=1)
            ax5.scatter(hrs_corr['Hours_Studied'], hrs_corr['Exam_Score'], alpha=0.25, s=15, color='#34d399', linewidths=0)
            z2 = np.polyfit(hrs_corr['Hours_Studied'], hrs_corr['Exam_Score'], 1)
            p2 = np.poly1d(z2)
            xl2 = np.linspace(hrs_corr['Hours_Studied'].min(), hrs_corr['Hours_Studied'].max(), 100)
            ax5.plot(xl2, p2(xl2), color='#facc15', linewidth=2)
            ax5.set_title('Study Hours vs Score', color='#e2e8f0', fontsize=11)
            ax5.set_xlabel('Hours/Week', color='#64748b', fontsize=8)
            ax5.tick_params(colors='#64748b', labelsize=8)
            for spine in ax5.spines.values(): spine.set_edgecolor('#252a3a')
            ax5.grid(alpha=0.07, color='white')
            plt.tight_layout(); st.pyplot(fig5); plt.close()

    # ── TAB 2: GRADE ESTIMATOR ──
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        col_in, col_out = st.columns([1, 1.3], gap="large")
        with col_in:
            st.markdown("<div class='edu-card' style='padding:28px;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Student Parameters</div>", unsafe_allow_html=True)
            att  = st.slider("📅 Attendance %",         0, 100, 85)
            hrs  = st.slider("📚 Hours Studied / Week", 0, 50,  20)
            prev = st.slider("📝 Previous Score",       0, 100, 70)
            tut  = st.number_input("🧑‍🏫 Tutoring Sessions", 0, 10, 2)
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("🔍 Predict Grade", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_out:
            if run:
                pred = model.predict(np.array([[att, hrs, prev, tut]]))[0]
                st.session_state['prediction_result'] = {'prediction': pred, 'att': att, 'hrs': hrs, 'prev': prev, 'tut': tut}

            if st.session_state['prediction_result'] is None:
                st.markdown("""
                <div class='edu-card' style='padding:48px; text-align:center;'>
                    <div style='font-size:3rem; margin-bottom:12px;'>🎓</div>
                    <div style='color:#64748b; font-size:1rem; font-weight:600;'>
                        Set parameters and click<br><span style='color:#4f8ef7;'>Predict Grade</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                res        = st.session_state['prediction_result']
                prediction = res['prediction']
                att_r      = res['att']
                hrs_r      = res['hrs']
                tut_r      = res['tut']

                if att_r < 60:    risk, badge_cls, risk_color = "Critical",        "badge-red",    "#f87171"
                elif att_r < 75:  risk, badge_cls, risk_color = "At Risk",         "badge-orange", "#fb923c"
                elif prediction < 55: risk, badge_cls, risk_color = "Needs Attention", "badge-orange", "#fb923c"
                else:             risk, badge_cls, risk_color = "On Track",        "badge-green",  "#34d399"

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

                st.markdown("<div class='section-header' style='margin-top:20px;'>Score Breakdown</div>", unsafe_allow_html=True)
                fig6, ax6 = plt.subplots(figsize=(6, 1.5), facecolor='#1a1e2b')
                ax6.set_facecolor('#1a1e2b')
                ax6.barh(['Predicted Score'], [prediction], color=risk_color, height=0.4)
                ax6.barh(['Predicted Score'], [100-prediction], left=prediction, color='#252a3a', height=0.4)
                ax6.set_xlim(0, 100)
                ax6.axvline(75, color='#facc15', linewidth=1.5, linestyle='--', alpha=0.8)
                ax6.tick_params(colors='#64748b', labelsize=9)
                ax6.set_xlabel('Score', color='#64748b', fontsize=9)
                for spine in ax6.spines.values(): spine.set_visible(False)
                plt.tight_layout(); st.pyplot(fig6); plt.close()

                # ── Strategic Intervention (Action Tasks) ──
                st.markdown("<div class='section-header' style='margin-top:20px;'>🎯 Strategic Intervention — Action Tasks</div>", unsafe_allow_html=True)
                if att_r < 60:
                    st.error("""
**🚨 CRITICAL — Attendance Crisis**

**Action Task 1:** Initiate immediate parental/guardian contact within 24 hours.\n\n
**Action Task 2:** Issue a formal Attendance Warning Letter and log in student record.\n\n
**Action Task 3:** Refer to school counsellor for root-cause assessment (transport, health, family).\n\n
**Action Task 4:** Set up a weekly attendance check-in with the form tutor.\n\n

Without urgent attendance recovery, academic continuation is at serious risk.
                    """)
                elif att_r < 75:
                    st.warning(f"""
**⚠️ AT RISK — Attendance Recovery Required**

Attendance of **{att_r}%** is below the mandatory 75% threshold.

**Action Task 1:** Schedule a 1-on-1 support meeting this week — identify barriers (transport, health, motivation).\n\n
**Action Task 2:** Create a personalised Attendance Improvement Plan (AIP) with weekly targets.\n\n
**Action Task 3:** Assign a peer study buddy to encourage daily attendance.\n\n
**Action Task 4:** Monitor for the next 3 weeks and escalate if no improvement is seen.\n\n

📈 A 10% attendance boost is projected to improve the exam score by ~5 marks.
                    """)
                elif hrs_r < 10:
                    st.warning(f"""
**🟠 LOW STUDY HOURS — Discipline Intervention**

Only **{hrs_r} hrs/week** of study detected — well below the recommended 15–20 hrs.

**Action Task 1:** Recommend a structured weekly timetable; share a template.\n\n
**Action Task 2:** Introduce accountability partner or group study session.\n\n
**Action Task 3:** Suggest productivity tools (Pomodoro, flashcards, past papers).\n\n

📈 Adding 5 more hrs/week could yield a **+8 mark** improvement on current trajectory.
                    """)
                elif prediction < 70:
                    st.warning(f"""
**🔵 BELOW TARGET — Academic Support Needed**

Predicted score of **{prediction:.1f}** is below the 70-mark target.

**Action Task 1:** Provide targeted revision packs for weak topic areas.\n\n
**Action Task 2:** Enrol in 1–2 additional tutoring sessions per week.\n\n
**Action Task 3:** Set a mock exam in 3 weeks to track improvement.\n\n
                    """)
                else:
                    st.success(f"""
**🌟 ON TRACK — Enrichment Mode**

Excellent indicators across attendance, study hours, and predicted grade ({prediction:.1f}).

**Action Task 1:** Introduce advanced challenge material or competitive exam preparation.\n\n
**Action Task 2:** Nominate for leadership, mentoring, or academic excellence programs.\n\n
**Action Task 3:** Maintain momentum — schedule a mid-term check-in to sustain performance.\n\n
                    """)

                # What-If Scenarios
                st.markdown("<div class='section-header' style='margin-top:20px;'>📐 What-If Scenarios</div>", unsafe_allow_html=True)
                scenarios = {
                    "Current":         model.predict([[att_r,              hrs_r,              res['prev'], tut_r]])[0],
                    "+10% Attendance": model.predict([[min(att_r+10,100),  hrs_r,              res['prev'], tut_r]])[0],
                    "+5 Study Hours":  model.predict([[att_r,              min(hrs_r+5,50),    res['prev'], tut_r]])[0],
                    "+1 Tutoring Ses": model.predict([[att_r,              hrs_r,              res['prev'], min(tut_r+1,10)]])[0],
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

    # ── TAB 3: DATA EXPLORER ──
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        e1, e2 = st.columns([1, 1], gap="large")
        with e1:
            st.markdown("<div class='section-header'>Feature Correlation Heatmap</div>", unsafe_allow_html=True)
            num_cols = ['Attendance','Hours_Studied','Previous_Scores','Tutoring_Sessions','Sleep_Hours','Exam_Score']
            corr_matrix = df_raw[[c for c in num_cols if c in df_raw.columns]].corr()
            fig8, ax8 = plt.subplots(figsize=(6, 5), facecolor='#1a1e2b')
            ax8.set_facecolor('#1a1e2b')
            from matplotlib.colors import LinearSegmentedColormap
            edu_cmap = LinearSegmentedColormap.from_list(
                'edu_blue', ['#0d0f14', '#1a2744', '#2d4a8a', '#4f8ef7', '#a8c8ff'], N=256
            )
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=edu_cmap, ax=ax8, linewidths=0.5, linecolor='#13161e',
                        annot_kws={'size': 9, 'color': 'white'}, cbar_kws={'shrink': 0.8},
                        vmin=-1, vmax=1)
            ax8.tick_params(colors='#e2e8f0', labelsize=9)
            ax8.set_title('Numeric Feature Correlations', color='#e2e8f0', pad=12)
            plt.tight_layout(); st.pyplot(fig8); plt.close()
        with e2:
            st.markdown("<div class='section-header'>Top & Bottom Performers</div>", unsafe_allow_html=True)
            top_n = st.selectbox("Show Top/Bottom N", [5, 10, 20], index=0)
            display_cols = [c for c in ['Name','Attendance','Hours_Studied','Previous_Scores','Tutoring_Sessions','Motivation_Level','Exam_Score'] if c in df_raw.columns]
            st.markdown("**🏆 Top Performers**")
            st.dataframe(df_raw.nlargest(top_n, 'Exam_Score')[display_cols].reset_index(drop=True), use_container_width=True, height=min(top_n*38+38, 280))
            st.markdown("**⚠️ Students Needing Support**")
            st.dataframe(df_raw.nsmallest(top_n, 'Exam_Score')[display_cols].reset_index(drop=True), use_container_width=True, height=min(top_n*38+38, 280))

        st.markdown("<div class='section-header' style='margin-top:20px;'>Full Dataset Preview</div>", unsafe_allow_html=True)
        search_att = st.slider("Filter by minimum attendance", 0, 100, 0)
        filtered_df = df_raw[df_raw['Attendance'] >= search_att]
        st.info(f"Showing {len(filtered_df):,} students with attendance ≥ {search_att}%")
        # Show Name as first column in full preview
        preview_cols = ['Name'] + [c for c in filtered_df.columns if c != 'Name']
        st.dataframe(filtered_df[preview_cols].head(100), use_container_width=True, height=350)


# ════════════════════════════════════════════════════════════════════════════════
# STUDENT PORTAL  (Public)
# ════════════════════════════════════════════════════════════════════════════════
def show_student_portal():
    df_raw = load_data()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='padding: 8px 0 20px 0;'>
            <div style='font-size:1.6rem; font-weight:800;'>🎓 EduInsight <span style='color:#4f8ef7;'>AI</span></div>
            <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>Student Portal</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div class='badge badge-green' style='margin-bottom:12px;'>✅ Public Access</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div style='color:#64748b; font-size:0.75rem; line-height:1.8;'>
        🏆 Merit Leaderboard<br>
        🚨 Defaulter List<br>
        📊 Class Highlights
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        if st.button("← Back to Landing", use_container_width=True):
            st.session_state['role'] = None
            st.rerun()

    # Header
    st.markdown("""
    <div class='hero-title'>Student Achievement Portal</div>
    <div class='hero-sub'>Public View · Celebrating Excellence · Encouraging Attendance</div>
    <br>
    """, unsafe_allow_html=True)

    # Class snapshot KPIs (non-sensitive)
    total  = len(df_raw)
    top10  = df_raw.nlargest(10, 'Exam_Score')['Exam_Score'].mean()
    def_ct = len(df_raw[df_raw['Attendance'] < 75])
    pass_r = len(df_raw[df_raw['Exam_Score'] >= 60]) / total * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Class Size",       f"{total:,}")
    k2.metric("🥇 Top-10 Avg Score", f"{top10:.1f}")
    k3.metric("⚠️ Below 75% Att.",   f"{def_ct:,}")
    k4.metric("✅ Pass Rate (≥60)",   f"{pass_r:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🏆 Merit Leaderboard", "🚨 Defaulter List"])

    # ── Merit Leaderboard ──
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='edu-card edu-card-accent' style='padding:20px 24px 12px;'>
            <div class='section-header'>Top 10 Students — Ranked by Exam Score</div>
            <div style='color:#64748b; font-size:0.82rem;'>
            Outstanding achievement in academics. Keep striving for excellence! 🌟
            </div>
        </div>
        """, unsafe_allow_html=True)

        top10_df = df_raw.nlargest(10, 'Exam_Score')[['Name', 'Exam_Score', 'Attendance', 'Hours_Studied', 'Previous_Scores', 'Tutoring_Sessions']].reset_index(drop=True)

        # Leaderboard dataframe
        medals = {0: "🥇", 1: "🥈", 2: "🥉"}
        leaderboard_display = top10_df.copy()
        leaderboard_display.insert(0, "Rank", [f"{medals.get(i, '')} #{i+1}" for i in range(len(leaderboard_display))])
        leaderboard_display.columns = ["Rank", "Student Name", "Exam Score", "Attendance %", "Study Hrs/Wk", "Prev. Score", "Tutoring Sessions"]
        leaderboard_display = leaderboard_display.set_index("Rank")
        st.dataframe(leaderboard_display, use_container_width=True, height=420)

        # Score distribution chart for context
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Class Score Distribution</div>", unsafe_allow_html=True)
        fig_l, ax_l = plt.subplots(figsize=(9, 3.5), facecolor='#1a1e2b')
        ax_l.set_facecolor('#1a1e2b')
        counts, bins, patches = ax_l.hist(df_raw['Exam_Score'], bins=40, edgecolor='none', color='#4f8ef7', alpha=0.7)
        # Highlight top 10 zone
        top_threshold = top10_df['Exam_Score'].min()
        for patch, left in zip(patches, bins[:-1]):
            if left >= top_threshold:
                patch.set_facecolor('#facc15')
                patch.set_alpha(1.0)
        ax_l.axvline(top_threshold, color='#facc15', linewidth=2, linestyle='--', label=f'Top-10 threshold ({top_threshold:.0f})')
        ax_l.axvline(60, color='#f87171', linewidth=1.5, linestyle='--', alpha=0.7, label='Pass Mark (60)')
        ax_l.set_xlabel('Exam Score', color='#64748b', fontsize=10)
        ax_l.set_ylabel('Number of Students', color='#64748b', fontsize=10)
        ax_l.tick_params(colors='#64748b', labelsize=9)
        for spine in ax_l.spines.values(): spine.set_edgecolor('#252a3a')
        ax_l.legend(facecolor='#252a3a', edgecolor='#252a3a', labelcolor='#e2e8f0', fontsize=9)
        ax_l.grid(alpha=0.07, color='white')
        plt.tight_layout(); st.pyplot(fig_l); plt.close()

    # ── Defaulter List ──
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='edu-card edu-card-red' style='padding:20px 24px 12px;'>
            <div class='section-header'>⚠️ Attendance Defaulter List — Below 75% Threshold</div>
            <div style='color:#64748b; font-size:0.82rem;'>
            Students listed here are below the mandatory attendance requirement. Regular attendance is critical for academic success.
            </div>
        </div>
        """, unsafe_allow_html=True)

        defaulters = df_raw[df_raw['Attendance'] < 75].sort_values('Attendance')[
            ['Name', 'Attendance', 'Exam_Score', 'Hours_Studied', 'Previous_Scores', 'Tutoring_Sessions']
        ].reset_index(drop=True)

        total_def = len(defaulters)
        crit_def  = len(defaulters[defaulters['Attendance'] < 60])

        da1, da2, da3 = st.columns(3)
        da1.metric("🚨 Total Defaulters", f"{total_def:,}")
        da2.metric("🔴 Critical (<60%)",  f"{crit_def:,}")
        da3.metric("📊 % of Class",       f"{total_def/len(df_raw)*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Defaulter dataframe
        show_df = defaulters.head(20).copy()
        show_df.insert(1, "Status", ["🔴 Critical" if a < 60 else "🟠 At Risk" for a in show_df['Attendance']])
        show_df.index = [f"#{i+1}" for i in range(len(show_df))]
        show_df.columns = ["Student Name", "Status", "Attendance %", "Exam Score", "Study Hrs/Wk", "Prev. Score", "Tutoring Sessions"]
        st.dataframe(show_df, use_container_width=True, height=500)
        st.caption(f"Showing 20 of {total_def} defaulters · Sorted by lowest attendance")

        # Attendance distribution of defaulters
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Defaulter Attendance Distribution</div>", unsafe_allow_html=True)
        fig_d, ax_d = plt.subplots(figsize=(9, 3), facecolor='#1a1e2b')
        ax_d.set_facecolor('#1a1e2b')
        ax_d.hist(defaulters['Attendance'], bins=25, color='#f87171', alpha=0.75, edgecolor='none')
        ax_d.axvline(60, color='#fb923c', linewidth=2, linestyle='--', label='Critical threshold (60%)')
        ax_d.axvline(75, color='#facc15', linewidth=2, linestyle='--', label='Minimum threshold (75%)')
        ax_d.set_xlabel('Attendance (%)', color='#64748b', fontsize=10)
        ax_d.set_ylabel('Count', color='#64748b', fontsize=10)
        ax_d.tick_params(colors='#64748b', labelsize=9)
        for spine in ax_d.spines.values(): spine.set_edgecolor('#252a3a')
        ax_d.legend(facecolor='#252a3a', edgecolor='#252a3a', labelcolor='#e2e8f0', fontsize=9)
        ax_d.grid(alpha=0.07, color='white')
        plt.tight_layout(); st.pyplot(fig_d); plt.close()

        st.info("⚠️ If your name appears in this list, please speak to your class teacher or counsellor immediately to discuss an attendance improvement plan.")


# ════════════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════════════
role = st.session_state['role']

if role is None:
    show_landing()
elif role == 'student':
    show_student_portal()
elif role == 'educator':
    if st.session_state['edu_logged_in']:
        show_educator_dashboard()
    else:
        show_educator_login()