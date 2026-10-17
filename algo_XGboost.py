import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import precision_score, recall_score
import pickle

print("Loading the 10-Parameter Dataset...")
df = pd.read_csv('CLEAN_physiological_data.csv')
df = df.select_dtypes(include=['number'])
df.columns = df.columns.str.strip()

# Dropping the 3 parameters plus Tcore to lock in your exact 10
X = df.drop(columns=['Tcore', 'CVC', 'Mean Tskin', 'change'], errors='ignore')
y = df['Tcore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the medical threshold for diagnosis
THRESHOLD = 38.0
y_test_binary = np.where(y_test >= THRESHOLD, 1, 0)

print("\n ALGORITHM 3: XGBOOST")
print("-" * 50)

# 1. Setup just the XGBoost Brain
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('regressor', XGBRegressor(random_state=42))
])

# 2. Train it
pipeline.fit(X_train, y_train)

# 3. Predict and convert the exact temperatures into a binary diagnosis
y_pred = pipeline.predict(X_test)
y_pred_binary = np.where(y_pred >= THRESHOLD, 1, 0)

# 4. Calculate exactly what you need for your slides
r2_accuracy = pipeline.score(X_test, y_test) * 100
precision = precision_score(y_test_binary, y_pred_binary, zero_division=0) * 100
sensitivity = recall_score(y_test_binary, y_pred_binary, zero_division=0) * 100

print(f"ALGORITHM: XGBoost")
print(f"   Accuracy (R²):  {r2_accuracy:.2f}%")
print(f"   Precision:      {precision:.2f}%")
print(f"   Sensitivity:    {sensitivity:.2f}%")
print("-" * 50)

# 5. SAVE THE SPECIFIC MODEL
print("Saving the XGBoost Brain...")
with open(r'C:\Users\candi\OneDrive\Desktop\Heat_Stress_App\model_xgb.pkl', 'wb') as file:
    pickle.dump(pipeline, file)
print("DONE! model_xgb.pkl is locked in your folder.")