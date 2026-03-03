# 🎓 Acadelo-Pro: Smart Team Formation & Behavioral Analytics

**Acadelo-Pro** is an advanced, Cloud-Ready Data Science web application built to optimize collaborative learning. By analyzing student interaction data (VLE clicks, active days, gap times, and cramming ratios), it uses a tuned **XGBoost** machine learning model to predict procrastination risks.

It then employs a sophisticated **Snake Draft Algorithm** intertwined with live **Peer Review Feedback** to automatically partition a classroom into perfectly balanced, highly synergized project teams.

---

## 🚀 Quick Start: How to Clone & Run

Follow these exact steps to get the project running on your local machine.

### 1. Clone the Repository

Open your terminal and clone the repository to your local machine:

```bash
git clone https://github.com/suhanipandita/acadelopro.git
cd acadelopro
```

---

### 2. Set Up a Virtual Environment (Recommended)

It is highly recommended to run this project inside a virtual environment to manage dependencies securely.

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

---

### 3. Install Dependencies

Install all required Machine Learning and Web framework libraries:

```bash
pip install -r requirements.txt
```

---

### 4. Configure Supabase Credentials

This project uses Supabase for live cloud database management.

1. Create a folder named `.streamlit` in the root directory.
2. Inside that folder, create a file named `secrets.toml`.
3. Add your Supabase URL and API Key like this:

```toml
# .streamlit/secrets.toml
[supabase]
url = "YOUR_SUPABASE_PROJECT_URL"
key = "YOUR_SUPABASE_API_KEY"
```

---

### 5. Initialize the Database & Train the Model

Ensure your database is populated and your model `.pkl` file is generated. You can run the following scripts included in the repository:

```bash
# Optional: Process raw data if needed
python process_data.py

# Train the XGBoost model and generate procrastination_model.pkl
python train_model.py 

# Seed your Supabase database with synthetic students for testing
python seed_database.py
```

---

### 6. Launch the Application

Start the Streamlit web dashboard:

```bash
streamlit run app.py
```

The application will open automatically in your browser at:

```
http://localhost:8501
```

---

## 📂 Project Structure & Key Files

* `app.py`
  The main Streamlit dashboard application containing the UI, routing, and module logic.

* `compare_models.py`
  The benchmarking script that pits XGBoost against Random Forest, Linear Regression, CatBoost, LightGBM, etc.

* `train_model.py` / `train_model_v2.py`
  Scripts used to train the XGBoost machine learning model and export the `.pkl` file.

* `process_data.py`
  Handles data cleaning and feature engineering for the initial datasets.

* `seed_database.py`
  Connects to Supabase and populates the cloud database with a synthetic "Golden Cohort" of students (Anchors, Risks, and Average members).

* `procrastination_model.pkl`
  The serialized, pre-trained XGBoost model used by the web app for live inference.

---

## 🌟 Core Modules & Features

The dashboard is divided into 5 distinct, highly interactive pages via a seamless sidebar navigation:

---

### 1. 📝 Data Entry & Behavioral Prediction

* **Smart Input:** Enter basic student metrics (Total Clicks, Active Days, Last 7 Days Activity). The system auto-calculates advanced metrics like **Cramming Ratio** and **Material Diversity**.
* **AI Forecasting:** The XGBoost engine instantly forecasts whether the student will submit early or late.
* **SHAP Explainability:** Utilizes **SHapley Additive exPlanations (SHAP)** waterfall plots to visually explain exactly *why* the AI made its prediction, breaking the "black box" of machine learning.

---

### 2. ⚖️ Auto-Team Balancer

* **Live Database Retrieval:** Pulls the current classroom roster directly from Supabase.
* **Algorithmic Partitioning:** Ranks students by their AI-predicted Risk Score and assigns them roles (🛡️ Anchor, 👤 Member, ⚠️ Risk Factor).
* **Snake Draft Distribution:** Evenly distributes these roles across teams to ensure no single team is overloaded with procrastinators.
* **Peer-Feedback Integration:** Adjusts the AI Risk Score dynamically based on real-time peer ratings.
* **Variance Analytics:** Renders Matplotlib visualizations comparing the Risk Variance of a *Random Grouping* versus the *Acadelo Optimized Grouping*.

---

### 3. 📊 Model Analytics (Showdown)

* **Live Data Drift Testing:** Instead of testing on historical CSVs, this tab pulls live, synthetic student data from the database and trains **7 different machine learning algorithms** on the fly.
* **The "Big 6" Benchmark:** Pits untuned baselines (Linear Regression, Random Forest, AdaBoost, LightGBM, CatBoost, HistGradientBoosting) against our **Fully Tuned XGBoost** architecture.
* **Visual Proof:** Generates dual bar charts (R² Score and RMSE) proving the mathematical superiority of the chosen algorithm.

---

### 4. 💬 Peer Feedback System

* **Teammate-Restricted Reviews:** Automatically maps students to their AI-generated teams. Students can strictly only review other students within their assigned group.
* **Cloud Storage:** Saves 1-to-5 star Collaboration Ratings directly to the `peer_feedback` table in Supabase.

---

### 5. 💾 Database Management

* **Admin Overview:** View the live state of the Supabase tables.
* **Synthetic Cohort Generation:** Features a "Bulk Insert" tool that instantly generates highly complex, non-linear student profiles to stress-test the algorithms.
* **1-Click Cleanup:** Securely wipe all testing data to reset the environment.

---

## 🛠️ Tech Stack & Architecture

* **Frontend:** Streamlit (Custom CSS injected for a premium SaaS Edu-Tech UI)
* **Backend:** Python 3.9+
* **Machine Learning:** XGBoost, Scikit-Learn, SHAP, LightGBM, CatBoost
* **Database:** Supabase (PostgreSQL)
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib

---

## 🗄️ Database Schema Requirement

To fully utilize this project, ensure your Supabase SQL database has the following two tables configured. You can run these commands directly in your Supabase SQL Editor:

---

### Table 1: `students`

```sql
CREATE TABLE students (
    id bigint generated by default as identity primary key,
    student_id text not null,
    clicks_total numeric,
    days_active numeric,
    gap_before_deadline numeric,
    material_diversity numeric,
    cramming_ratio numeric,
    clicks_last_7d numeric,
    days_early numeric
);
```

---

### Table 2: `peer_feedback`

```sql
CREATE TABLE peer_feedback (
    id bigint generated by default as identity primary key,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    reviewer_id text not null,
    reviewee_id text not null,
    rating int not null check (rating >= 1 and rating <= 5),
    comments text
);
```
