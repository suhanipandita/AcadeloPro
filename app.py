import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from supabase import create_client, Client
from compare_models import get_model_comparison_results, create_comparison_charts

# Set Page Config
st.set_page_config(page_title="Acadelo-Pro (Cloud)", layout="wide", page_icon="🎓", initial_sidebar_state="expanded")

# ===================================================
# 🌟 UI UPGRADE: MediCore Structural Theme (Edu-Tech Colors)
# ===================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #8ecae6; 
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #293241;
    }
    
    header[data-testid="stHeader"] {
        background-color: #3d5a80;
        border-bottom: 1px solid #1e1b18;
    }
    [data-testid="stSidebar"] {
        background-color: #3d5a80;
        border-right: 2px solid #b56576;
    }
    
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="stMetricValue"] {
        color: #4338CA; 
        font-weight: 700;
    }
    
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        color: #FFFFFF;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.2s ease;
        box-shadow: 0px 4px 6px rgba(79, 70, 229, 0.25);
    }
    div[data-testid="stButton"] > button:hover {
        transform: scale(1.02);
        box-shadow: 0px 6px 12px rgba(79, 70, 229, 0.4);
        color: #FFFFFF;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-bottom: none;
        border-radius: 8px 8px 0px 0px;
        padding: 12px 20px;
        font-weight: 600;
        color: #6B7280;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #F8F9FA;
        color: #4F46E5; 
        border-top: 3px solid #4F46E5;
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #D1D5DB;
        padding: 10px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #4F46E5;
        box-shadow: 0 0 0 1px #4F46E5;
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E5E7EB;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.02);
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 Acadelo-Pro")
    st.caption("v2.0.0 - Cloud Edition")
    st.divider()
    st.markdown("""
    **Core Modules:**
    - 📊 Behavioral Prediction
    - 🤝 Smart Team Formation
    - 📈 Algorithm Benchmarking
    - 💬 Peer Review System
    """)
    st.divider()
    st.info("Powered by XGBoost & Supabase")

# ===================================================
# 1. CONFIGURATION & CREDENTIALS
# ===================================================
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()

MODEL_FILE = 'procrastination_model.pkl'

# ===================================================
# 2. HELPER FUNCTIONS
# ===================================================

@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        return pickle.load(open(MODEL_FILE, "rb"))
    except:
        st.error(f"Model file '{MODEL_FILE}' not found. Please run 'train_model.py' first.")
        return None

def load_db(supabase):
    try:
        response = supabase.table('students').select("*").execute()
        data = response.data
        if data:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(columns=[
                'student_id', 'clicks_total', 'days_active', 'gap_before_deadline',
                'material_diversity', 'cramming_ratio', 'clicks_last_7d'
            ])
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def save_single_student(supabase, row_data):
    try:
        supabase.table('students').insert(row_data).execute()
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

# --- FEEDBACK HELPERS ---
def save_feedback(supabase, feedback_data):
    try:
        supabase.table('peer_feedback').insert(feedback_data).execute()
        return True
    except Exception as e:
        st.error(f"Feedback save failed: {e}")
        return False

def load_feedback(supabase):
    try:
        response = supabase.table('peer_feedback').select("*").execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def bulk_insert_advanced(supabase, num_students=50):
    new_rows = []
    count_anchor = int(num_students * 0.33)
    count_risk = int(num_students * 0.17)
    count_member = num_students - (count_anchor + count_risk)
    
    for _ in range(count_anchor):
        clicks = int(np.random.randint(400, 800))
        cram_ratio = np.random.uniform(0.1, 0.3)
        new_rows.append({
            "student_id": f"22310{np.random.randint(100, 999)}",
            "clicks_total": clicks,
            "days_active": int(np.random.randint(15, 30)),
            "gap_before_deadline": int(np.random.randint(0, 2)),
            "material_diversity": int(np.random.randint(10, 25)),
            "cramming_ratio": round(cram_ratio, 2),
            "clicks_last_7d": int(clicks * cram_ratio)
        })

    for _ in range(count_member):
        clicks = int(np.random.randint(100, 300))
        cram_ratio = np.random.uniform(0.3, 0.6)
        new_rows.append({
            "student_id": f"22310{np.random.randint(100, 999)}",
            "clicks_total": clicks,
            "days_active": int(np.random.randint(5, 14)),
            "gap_before_deadline": int(np.random.randint(3, 8)),
            "material_diversity": int(np.random.randint(4, 12)),
            "cramming_ratio": round(cram_ratio, 2),
            "clicks_last_7d": int(clicks * cram_ratio)
        })

    for _ in range(count_risk):
        clicks = int(np.random.randint(0, 50))
        cram_ratio = np.random.uniform(0.8, 1.0)
        new_rows.append({
            "student_id": f"22310{np.random.randint(100, 999)}",
            "clicks_total": clicks,
            "days_active": int(np.random.randint(0, 3)),
            "gap_before_deadline": int(np.random.randint(7, 30)),
            "material_diversity": int(np.random.randint(0, 3)),
            "cramming_ratio": round(cram_ratio, 2),
            "clicks_last_7d": int(clicks * cram_ratio)
        })
    
    import random
    random.shuffle(new_rows)
    
    try:
        supabase.table('students').insert(new_rows).execute()
        return True, len(new_rows)
    except Exception as e:
        st.error(f"Bulk insert failed: {e}")
        return False, 0

# ===================================================
# 3. INITIALIZATION & UI
# ===================================================

supabase = init_supabase()
model = load_model()

st.title("🎓 Acadelo-Pro")
st.markdown("**Smart Team Formation System** | *Advanced Behavioral Analytics*")
st.write("---")

if not supabase or not model:
    st.stop()

# 4 TABS: Data Entry, Team Balancer, Model Analytics, Peer Feedback
tab1, tab2, tab3, tab4 = st.tabs(["📝 Data Entry", "⚖️ Team Balancer", "📊 Model Analytics", "💬 Peer Feedback"])

# --- TAB 1: DATA ENTRY ---
with tab1:
    st.subheader("Add Student Profile")
    col1, col2 = st.columns(2)
    with col1:
        s_id = st.text_input("Student ID", value="", placeholder="e.g. 22310884")
        clicks = st.number_input("Total Clicks (Semester)", 0, 2000, 150)
        active = st.number_input("Active Days", 0, 60, 5)
    with col2:
        gap = st.number_input("Gap (Days since login)", 0, 60, 2)
        clicks_recent = st.number_input("Clicks in Last 7 Days", 0, 500, 30)

    cram_ratio = clicks_recent / clicks if clicks > 0 else 0.0
    diversity_estimate = min(25, int(clicks / 15)) + 1
    
    st.info(f"📊 Auto-Calculated Metrics: Cramming Ratio: **{cram_ratio:.2f}** | Diversity Score: **{diversity_estimate}**")

    if st.button("☁️ Save & Predict"):
        if s_id == "":
            st.warning("Enter Student ID")
        else:
            row_data = {
                "student_id": s_id, "clicks_total": clicks, "days_active": active,
                "gap_before_deadline": gap, "material_diversity": diversity_estimate,
                "cramming_ratio": cram_ratio, "clicks_last_7d": clicks_recent
            }
            if save_single_student(supabase, row_data):
                st.success(f"Saved **{s_id}**!")
                input_df = pd.DataFrame([row_data]).drop(columns=['student_id'])
                input_df = input_df[['clicks_total', 'days_active', 'gap_before_deadline', 
                                   'material_diversity', 'cramming_ratio', 'clicks_last_7d']]
                pred = model.predict(input_df)[0]
                if pred < 0:
                    st.error(f"Predicted Result: {abs(pred):.1f} Days Late")
                else:
                    st.success(f"Predicted Result: {pred:.1f} Days Early")

# --- TAB 2: TEAM FORMATION ---
with tab2:
    st.subheader("Optimized Class Partitioning")
    classroom = load_db(supabase)
    feedback_db = load_feedback(supabase)
    
    if classroom.empty:
        st.warning("Database is empty.")
    else:
        st.write(f"Loaded **{len(classroom)}** students.")
        if st.button("🚀 Generate Teams"):
            if len(classroom) < 4:
                st.error("Need 4+ students.")
            else:
                features = ['clicks_total', 'days_active', 'gap_before_deadline', 
                           'material_diversity', 'cramming_ratio', 'clicks_last_7d']
                classroom[features] = classroom[features].fillna(0)
                classroom['predicted_early'] = model.predict(classroom[features])
                classroom['risk_score'] = classroom['predicted_early'] * -1 
                
                # Peer Feedback Adjustment
                if not feedback_db.empty:
                    avg_feedback = feedback_db.groupby('reviewee_id')['rating'].mean().reset_index()
                    avg_feedback.rename(columns={'reviewee_id': 'student_id', 'rating': 'peer_score'}, inplace=True)
                    classroom = pd.merge(classroom, avg_feedback, on='student_id', how='left')
                else:
                    classroom['peer_score'] = np.nan
                
                classroom['peer_score'] = classroom['peer_score'].fillna(3.0)
                classroom['adjusted_risk'] = classroom['risk_score'] + ((3.0 - classroom['peer_score']) * 0.5)

                num_teams = max(1, len(classroom) // 4)
                sorted_students = classroom.sort_values('adjusted_risk', ascending=False).reset_index(drop=True)
                smart_risks = [0] * num_teams
                final_roster = []
                
                for i, row in sorted_students.iterrows():
                    cycle = i // num_teams
                    idx = i % num_teams
                    team_idx = (num_teams - 1) - idx if cycle % 2 == 1 else idx
                    
                    if team_idx < num_teams:
                        smart_risks[team_idx] += row['adjusted_risk']
                        role = "👤 Member"
                        if row['adjusted_risk'] >= 1.5: role = "⚠️ Risk Factor"
                        elif row['adjusted_risk'] <= -2.0: role = "🛡️ Anchor"
                        final_roster.append({
                            "Team ID": team_idx + 1, "Student ID": row['student_id'],
                            "Role": role, "Risk Score": row['adjusted_risk'], "Peer Rating": f"⭐ {row['peer_score']:.1f}"
                        })
                
                st.subheader("📋 Final Teams Roster")
                st.dataframe(pd.DataFrame(final_roster).sort_values(['Team ID', 'Risk Score']), hide_index=True, use_container_width=True)

# --- TAB 3: MODEL ANALYTICS ---
with tab3:
    st.subheader("Live Database Algorithm Benchmark")
    if st.button("Run Live Benchmark", type="primary"):
        with st.spinner("Training models..."):
            db_data = load_db(supabase)
            results_df = get_model_comparison_results(df=db_data)
            if results_df is not None:
                st.success("Benchmarking Complete!")
                st.dataframe(results_df.style.highlight_max(subset=['R2 Score'], color='lightgreen'), use_container_width=True)
                fig = create_comparison_charts(results_df)
                st.pyplot(fig)
            else:
                st.error("Not enough historical data in Supabase.")

# --- TAB 4: PEER FEEDBACK & DB ---
with tab4:
    col_entry, col_db = st.columns([1, 1])
    
    with col_entry:
        st.subheader("💬 Submit Peer Review")
        classroom_data = load_db(supabase)
        student_list = classroom_data['student_id'].tolist() if not classroom_data.empty else []
        
        with st.form("feedback_form"):
            reviewer = st.selectbox("Your ID", options=["Select..."] + student_list)
            reviewee = st.selectbox("Teammate ID", options=["Select..."] + student_list)
            rating = st.slider("Collaboration Rating (1-5)", 1, 5, 3)
            submitted_fb = st.form_submit_button("Submit Review")
            
            if submitted_fb:
                if reviewer == "Select..." or reviewee == "Select...":
                    st.error("Select both IDs.")
                elif reviewer == reviewee:
                    st.error("You cannot review yourself.")
                else:
                    if save_feedback(supabase, {"reviewer_id": reviewer, "reviewee_id": reviewee, "rating": rating}):
                        st.success("Feedback submitted!")

    with col_db:
        st.subheader("💾 Database Management")
        current_db = load_db(supabase)
        st.dataframe(current_db, use_container_width=True)
        
        st.subheader("🎲 Bulk Tools")
        num = st.number_input("Count", 1, 200, 60)
        if st.button("Generate & Insert"):
            success, count = bulk_insert_advanced(supabase, num)
            if success:
                st.success(f"Added {count} profiles!")
                st.rerun()
        
        if st.button("🗑️ DELETE ALL DATA"):
            supabase.table('students').delete().gt('id', 0).execute()
            supabase.table('peer_feedback').delete().gt('id', 0).execute()
            st.warning("Database Cleared!")
            st.rerun()