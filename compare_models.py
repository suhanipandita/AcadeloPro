import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split

def get_model_comparison_results(df=None):
    """Trains models. Uses passed DataFrame (Supabase) OR loads CSV if None."""
    
    # 1. Fallback to CSV if no data is provided (Terminal Mode)
    if df is None:
        if not os.path.exists('processed_data.csv'):
            return None
        df = pd.read_csv('processed_data.csv')
    
    # 2. Safety Check: We can only benchmark records that have an Answer Key
    if 'days_early' not in df.columns:
        return None
    
    df = df.dropna(subset=['days_early']) # Ignore new students added from Tab 1
    if len(df) < 10: # Ensure we have enough data to train/test
        return None

    features = ['clicks_total', 'days_active', 'gap_before_deadline', 
                'material_diversity', 'cramming_ratio', 'clicks_last_7d']
    
    X = df[features]
    y = df['days_early']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ... (Keep your exact models dictionary from earlier here) ...
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "AdaBoost (Baseline)": AdaBoostRegressor(n_estimators=30, random_state=42),
        "Random Forest (Baseline)": RandomForestRegressor(n_estimators=50, max_depth=2, random_state=42),
        "HistGradientBoosting (Baseline)": HistGradientBoostingRegressor(max_iter=30, max_depth=2, random_state=42),
        "LightGBM (Baseline)": lgb.LGBMRegressor(n_estimators=40, max_depth=2, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost (Baseline)": cb.CatBoostRegressor(iterations=40, depth=2, learning_rate=0.05, random_seed=42, verbose=0),
        "XGBoost (Acadelo-Tuned)": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.06, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5, random_state=42)
    }
    
    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results.append({"Model": name, "R2 Score": r2, "RMSE (Days)": rmse})
        
    return pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

def create_comparison_charts(results_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # R2 Chart
    colors_r2 = ['mediumseagreen' if x == results_df['R2 Score'].max() else '#d3d3d3' for x in results_df['R2 Score']]
    ax1.barh(results_df['Model'], results_df['R2 Score'], color=colors_r2)
    ax1.set_xlabel("R² Score (Higher = Better)")
    ax1.set_title("Pattern Accuracy (Untuned vs Tuned)")
    ax1.invert_yaxis() 
    
    # RMSE Chart
    colors_rmse = ['mediumseagreen' if x == results_df['RMSE (Days)'].min() else '#d3d3d3' for x in results_df['RMSE (Days)']]
    ax2.barh(results_df['Model'], results_df['RMSE (Days)'], color=colors_rmse)
    ax2.set_xlabel("RMSE in Days (Lower = Better)")
    ax2.set_title("Prediction Error Rate")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Starting Model Comparison...\n")
    df_results = get_model_comparison_results()
    
    if df_results is not None:
        print("\nFINAL LEADERBOARD:")
        print("="*60)
        print(df_results.to_string(index=False))
        print("="*60)
        
        print("\nGenerating charts... (Close the chart window to exit the program)")
        fig = create_comparison_charts(df_results)
        plt.show() 
    else:
        print("❌ Error: 'processed_data.csv' not found. Run process_data.py first.")