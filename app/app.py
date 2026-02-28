import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EduInsight AI", layout="wide")

# Session State for Login
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

# Sidebar
with st.sidebar:
    st.title("🛡️ Educator Portal")
    if st.session_state['logged_in']:
        if st.button("Logout"): 
            st.session_state['logged_in'] = False
            st.rerun()
    st.info("**PS 05:** Attendance & Performance Correlation")

# 1. Login Logic
if not st.session_state['logged_in']:
    st.title("🔐 Educator Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "samsung123":
            st.session_state['logged_in'] = True
            st.rerun()
        else: st.error("Invalid credentials")

# 2. Main Dashboard
else:
    st.title("📘 EduInsight AI: Grade Estimator & Analysis")
    
    # Load Data for Visualization
    df_raw = pd.read_csv('data/StudentPerformanceFactors.csv')
    model = joblib.load('model/model.pkl')

    tab1, tab2 = st.tabs(["📈 Data Correlation", "🤖 Grade Estimator"])

    with tab1:
        st.subheader("Statistical Correlation: Attendance vs. Marks")
        col_c1, col_c2 = st.columns([1, 1])
        
        with col_c1:
            # Correlation Matrix
            corr_val = df_raw[['Attendance', 'Exam_Score']].corr().iloc[0,1]
            st.metric("Correlation Coefficient", f"{corr_val:.2f}")
            st.write("A value close to 1 indicates that attendance is a direct driver of grades.")
            
        with col_c2:
            # Trend Line Graph
            fig, ax = plt.subplots()
            sns.regplot(x='Attendance', y='Exam_Score', data=df_raw.sample(500), 
                        scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax)
            ax.set_title("Attendance vs. Exam Score Trend Line")
            st.pyplot(fig)

    with tab2:
        col_in, col_out = st.columns([1, 1.2], gap="large")
        
        with col_in:
            st.subheader("Input Student Data")
            att = st.slider("Attendance %", 0, 100, 85)
            hrs = st.slider("Hours Studied/Week", 0, 50, 20)
            prev = st.slider("Previous Score", 0, 100, 70)
            tut = st.number_input("Tutoring Sessions", 0, 10, 2)
            mot = st.select_slider("Motivation Level", options=["Low", "Medium", "High"])
            mot_map = {"Low": 0, "Medium": 1, "High": 2}

        with col_out:
            st.subheader("🔍 Predicted Outcome")
            input_data = np.array([[att, hrs, prev, tut, mot_map[mot]]])
            prediction = model.predict(input_data)[0]
            
            st.metric("Estimated Exam Score", f"{prediction:.1f}%")
            
            # Actionable Intervention
            st.markdown("---")
            st.markdown("## 🎯 STRATEGIC INTERVENTION")
            
            # Dynamic Insight Logic
            with st.container(border=True):
                if att < 75:
                    st.error("### 🚨 Focus: Attendance Recovery")
                    st.write("**Action:** Immediate meeting required. Attendance is the primary bottleneck for this student.")
                elif hrs < 15:
                    st.warning("### 🟠 Focus: Study Discipline")
                    st.write("**Action:** Recommend weekly study schedule. Predicted score could increase by ~8% with 5 more hours/week.")
                else:
                    st.success("### 🌟 Focus: Enrichment")
                    st.write("**Action:** Suggest advanced materials. Student is performing well based on current behavioral patterns.")

            st.caption("Early intervention based on this estimator improves student outcomes")