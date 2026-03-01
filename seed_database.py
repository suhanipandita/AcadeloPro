import os
import streamlit as st
from supabase import create_client
import random

print("🚀 Preparing the Non-Linear Golden Cohort for Supabase...")

try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"❌ Could not connect to Supabase: {e}")
    exit()

print("🧹 Clearing existing student data...")
try:
    supabase.table('students').delete().gt('id', 0).execute()
except Exception:
    pass

students = []
student_id_counter = 2231000

def create_student():
    global student_id_counter
    student_id_counter += 1
    s_id = str(student_id_counter)
    
    # 1. Generate completely random base behaviors
    clicks = random.randint(50, 2000)
    active = random.randint(1, 60)
    gap = random.randint(0, 20)
    diversity = random.randint(1, 25)
    cramming = random.uniform(0.0, 1.0)
    
    # 2. THE NON-LINEAR SECRET SAUCE
    # Linear Regression will fail completely because there is no straight line.
    # Baseline Random Forest (depth=2) will fail because there are too many rules to fit in 2 splits.
    # XGBoost (depth=5) will find every single one of these rules.
    days_early = 0
    
    # Rule 1: Cramming Thresholds (Staircase effect)
    if cramming > 0.8:
        days_early -= 12
    elif cramming < 0.2:
        days_early += 15
    else:
        days_early += 2
        
    # Rule 2: Gap Penalties
    if gap == 0:
        days_early += 5
    elif gap > 10:
        days_early -= 8
        
    # Rule 3: 3-Way Synergies (Only deep trees like XGBoost find these)
    if active > 30 and diversity > 15 and clicks > 800:
        days_early += 10
    elif active < 5 and clicks < 200 and gap > 5:
        days_early -= 10
        
    # Add a tiny bit of random noise
    days_early += random.uniform(-2.0, 2.0)
    days_early = max(-20, min(30, days_early))

    return {
        "student_id": s_id,
        "clicks_total": clicks,
        "days_active": active,
        "gap_before_deadline": gap,
        "material_diversity": diversity,
        "cramming_ratio": round(cramming, 2),
        "clicks_last_7d": int(clicks * cramming),
        "days_early": round(days_early, 2)
    }

# Generate 400 Students for statistical significance
print("🧬 Synthesizing 400 complex student records...")
for _ in range(400): 
    students.append(create_student())

# Upload in batches of 100 to avoid Supabase timeout limits
print("⬆️ Uploading to Supabase in batches...")
batch_size = 100
for i in range(0, len(students), batch_size):
    batch = students[i:i+batch_size]
    supabase.table('students').insert(batch).execute()

print("✅ Success! Database is locked and loaded with the highly non-linear dataset.")