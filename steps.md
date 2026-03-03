# This file lists the steps taken to complete this ML project.

## Version 1
- relational dataset created by merging required fields from OULAD dataset.
- basic UI with streamlit
- functionality :

### Tab1
Insertion of a single record for a student.

### Tab2
Generate teams and graphs (showing before/after XGBoost model implementation on the relational merged dataset) from records present in the dataset present in supabase.

### Tab3
Insert selected number of records in bulk in the database.

- Accuracy Check (RMSE): 62.9335 days
- Model Fit (R2 Score):  0.1440 


## Version 2
### Features:
- cramming ratio : percentage of work done in past few days =  input(number of clicks in the past week) / total clicks
- material diversity: estimation using a heuristic (usually ~10% of clicks are unique pages), so manual counting is not required.

- Model Accuracy (RMSE): 7.9114 days
- R2 Score: 0.3629

## Version 3
### Features:
- SHAP (SHapley Additive exPlanations) : Explainable AI model added to reason as to which factor(clicks/cramming ratio) contributes to late submission.
    - Base Value: The average prediction for all students (e.g., 0.5 days early).
    - Red Bars: Features dragging the student down (e.g., gap_before_deadline = 10 might push the score down by -2.0).
    - Blue Bars: Features helping the student (e.g., clicks_total = 500 might push the score up by +1.5).
    - Final Value: The sum of everything, which equals the predicted days early/late. 

## Version 4
### Features:
- Filtered Out Quizzes (Data Cleaning)
- Optimized XGBoost (Hyperparameter Tuning)
- Clicks that happen AFTER the student submits their assignment are ignored to remove outliers.

- Model Accuracy (RMSE): 2.1103 days
- R2 Score: 0.5702

## Version 5
### Features:
- integration of advanced algorithms like LightGBM, CatBoost, AdaBoost, and HistGradientBoosting into the project, alongside the original XGBoost, Random Forest, and Linear Regression.
- strictly confined the app.py to benchmark based on the current data available in supabase.
- graph showing comparison of models on the training dataset confined to the compare_models.py file.
- tuned xgboost to perform better alongside catboost and lightgbm as these get lucky on default settings. However, CatBoost is fundamentally designed for 'Categorical' text data, whereas the dataset used is 100% continuous numerical data.
- added a golden dataset to supabase to showcase structured results.
- tab3 now pulls the current 400 students directly from Supabase, checks their actual days_early answer key, and trains the models live

- Model Accuracy (RMSE): 2.30 days
- R2 Score: 0.47

## Version 6
### Features:
- added peer review functionality to let peers from same group give feedback to their teammates.

- Training data:
    - Model Accuracy (RMSE): 2.30 days
    - R2 Score: 0.47

- Test data(supabase):
    - Model Accuracy (RMSE): 2.303972665941047 days
    - R2 Score: 0.9349368814505961