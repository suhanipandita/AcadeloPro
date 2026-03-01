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
st.set_page_config(page_title="Acadelo-Pro (Cloud)", layout="wide")

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

def bulk_insert_advanced(supabase, num_students=50):
    """
    Generates students with Advanced Behavioral Patterns.
    """
    new_rows = []
    
    # Deterministic Counts
    count_anchor = int(num_students * 0.33)
    count_risk = int(num_students * 0.17)
    count_member = num_students - (count_anchor + count_risk)
    
    # 1. Anchors (Consistent, Low Cramming, High Diversity)
    for _ in range(count_anchor):
        clicks = int(np.random.randint(400, 800))
        cram_ratio = np.random.uniform(0.1, 0.3) # Only 10-30% work in last week
        
        new_rows.append({
            "student_id": f"22310{np.random.randint(100, 999)}",
            "clicks_total": clicks,
            "days_active": int(np.random.randint(15, 30)),
            "gap_before_deadline": int(np.random.randint(0, 2)),
            "material_diversity": int(np.random.randint(10, 25)), # High diversity
            "cramming_ratio": round(cram_ratio, 2),
            "clicks_last_7d": int(clicks * cram_ratio)
        })

    # 2. Members (Average)
    for _ in range(count_member):
        clicks = int(np.random.randint(100, 300))
        cram_ratio = np.random.uniform(0.3, 0.6) # 30-60% work in last week
        
        new_rows.append({
            "student_id": f"22310{np.random.randint(100, 999)}",
            "clicks_total": clicks,
            "days_active": int(np.random.randint(5, 14)),
            "gap_before_deadline": int(np.random.randint(3, 8)),
            "material_diversity": int(np.random.randint(4, 12)),
            "cramming_ratio": round(cram_ratio, 2),
            "clicks_last_7d": int(clicks * cram_ratio)
        })

    # 3. Risks (Panic Workers or Ghosts)
    for _ in range(count_risk):
        clicks = int(np.random.randint(0, 50))
        cram_ratio = np.random.uniform(0.8, 1.0) # 80-100% work in last week (Panic)
        
        new_rows.append({
            "student_id": f"22310{np.random.randint(100, 999)}",
            "clicks_total": clicks,
            "days_active": int(np.random.randint(0, 3)),
            "gap_before_deadline": int(np.random.randint(7, 30)),
            "material_diversity": int(np.random.randint(0, 3)), # Low diversity
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

# --- LIVE MODEL COMPARISON FUNCTION ---
@st.cache_data
def run_model_comparison():
    if not os.path.exists('processed_data.csv'):
        return None
    
    df = pd.read_csv('processed_data.csv')
    features = ['clicks_total', 'days_active', 'gap_before_deadline', 
                'material_diversity', 'cramming_ratio', 'clicks_last_7d']
    
    X = df[features]
    y = df['days_early']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "R2 Score": r2})
        
    return pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

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

# ADDED 4TH TAB FOR ANALYTICS
tab1, tab2, tab3, tab4 = st.tabs(["📝 Data Entry", "⚖️ Auto-Team Balancer", "📊 Model Analytics", "💾 Database"])

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
        clicks_recent = st.number_input("Clicks in Last 7 Days", 0, 500, 30, help="Used to calculate Cramming Ratio")

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
                
                num_teams = max(1, len(classroom) // 4)
                sorted_students = classroom.sort_values('risk_score', ascending=False).reset_index(drop=True)
                smart_risks = [0] * num_teams
                final_roster = []
                
                for i, row in sorted_students.iterrows():
                    cycle = i // num_teams
                    idx = i % num_teams
                    team_idx = (num_teams - 1) - idx if cycle % 2 == 1 else idx
                    
                    if team_idx < num_teams:
                        smart_risks[team_idx] += row['risk_score']
                        role = "👤 Member"
                        if row['risk_score'] >= 1.5: role = "⚠️ Risk Factor"
                        elif row['risk_score'] <= -2.0: role = "🛡️ Anchor"
                            
                        final_roster.append({
                            "Team ID": team_idx + 1, "Student ID": row['student_id'],
                            "Role": role, "Risk Score": row['risk_score']
                        })
                
                col1, col2 = st.columns(2)
                classroom_rnd = classroom.copy().sample(frac=1).reset_index(drop=True)
                t_ids = np.array_split(range(len(classroom_rnd)), num_teams)
                classroom_rnd['team_rnd'] = 0
                for t, idxs in enumerate(t_ids): classroom_rnd.loc[idxs, 'team_rnd'] = t+1
                random_risk = classroom_rnd.groupby('team_rnd')['risk_score'].sum()

                with col1:
                    st.markdown("### 🔴 Random")
                    fig1, ax1 = plt.subplots()
                    ax1.plot(random_risk.values, marker='o', color='red', linestyle='--')
                    ax1.set_title(f"Variance: {random_risk.var():.2f}")
                    st.pyplot(fig1)
                with col2:
                    st.markdown("### 🟢 Acadelo")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(smart_risks, marker='o', color='green')
                    ax2.set_title(f"Variance: {np.var(smart_risks):.2f}")
                    st.pyplot(fig2)

                st.write("---")
                st.subheader("📋 Final Teams Roster")
                st.dataframe(pd.DataFrame(final_roster).sort_values(['Team ID', 'Risk Score']), hide_index=True, use_container_width=True)

# --- TAB 3: MODEL SHOWDOWN ---
with tab3:
    st.subheader("Live Database Algorithm Benchmark")
    st.write("This tab pulls the **current historical data directly from your Supabase database** and trains 7 different machine learning models live to prove XGBoost's dominance.")
    
    if st.button("Run Live Benchmark", type="primary"):
        with st.spinner("Fetching DB records and training 6 different models..."):
            
            # 1. Load data directly from Supabase
            db_data = load_db(supabase)
            
            # 2. Pass the LIVE database data to our imported function
            results_df = get_model_comparison_results(df=db_data)
            
            if results_df is not None:
                st.success(f"Benchmarking Complete! Trained on {len(db_data.dropna(subset=['days_early']))} historical DB records.")
                
                st.dataframe(
                    results_df.style.highlight_max(subset=['R2 Score'], color='lightgreen')
                                  .highlight_min(subset=['RMSE (Days)'], color='lightgreen'),
                    use_container_width=True
                )
                
                fig = create_comparison_charts(results_df)
                st.pyplot(fig)
                
                st.info("""
                **Why do we see these results?**
                * **Linear Models:** Fail to capture the non-linear complexity of human behavior.
                * **AdaBoost/Random Forest:** Good baselines, but struggle with highly granular noise.
                * **XGBoost, LightGBM, CatBoost:** The "Big 3". By tuning XGBoost to our specific database architecture, we create a massive gap in accuracy over the untuned baselines.
                """)
            else:
                st.error("Not enough historical data in Supabase (missing 'days_early' outcomes) to run benchmark. Run seed_database.py first!")
                
# --- TAB 4: DB MANAGEMENT ---
with tab4:
    st.subheader("Manage Database")
    current_db = load_db(supabase)
    st.dataframe(current_db, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎲 Bulk Generate")
        num = st.number_input("Count", 1, 200, 60)
        if st.button("Generate & Insert"):
            success, count = bulk_insert_advanced(supabase, num)
            if success:
                st.success(f"Added {count} profiles!")
                st.rerun()
    
    with col2:
        st.subheader("⚠️ Cleanup")
        if st.button("🗑️ DELETE ALL"):
            supabase.table('students').delete().gt('id', 0).execute()
            st.warning("Cleared!")
            st.rerun()