import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix
import os

# 1. Setup Path and Load Data
base_path = r'C:\Users\candi\OneDrive\Desktop\Heat_Stress_App'
df = pd.read_csv(os.path.join(base_path, 'CLEAN_physiological_data.csv'))
df.columns = df.columns.str.strip()

# Exact columns from your CSV
cols = [
    'Mean Tbody', 'LSR', 'LDF', 'HR initial', 'SBP', 'DBP', 'MAP', 
    'SSNA', 'change (pre-post)', 'Heat exposure duration (s)', 'Sweat rate (L/h)'
]

X_values = df[cols].values 
y_true_binary = np.where(df['Tcore'] >= 38.0, 1, 0)

# 2. Load the Decision Tree Model specifically
model_file = 'model_dt.pkl'
full_path = os.path.join(base_path, model_file)

if os.path.exists(full_path):
    with open(full_path, 'rb') as f:
        model = pickle.load(f)
    
    # 3. Predict and Generate Matrix
    y_pred = model.predict(X_values)
    y_pred_binary = np.where(y_pred >= 38.0, 1, 0)
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    # 4. Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Normal (<38°C)', 'Heat Stress (≥38°C)'], 
                yticklabels=['Actual Normal', 'Actual Heat Stress'],
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title('Confusion Matrix: Decision Tree Regressor', fontsize=14, pad=20)
    plt.ylabel('Actual Clinical State', fontsize=12)
    plt.xlabel('AI Predicted State', fontsize=12)
    
    # Save and Show
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'DT_Confusion_Matrix_Only.png'), dpi=300)
    plt.show()
    print("Decision Tree matrix generated and saved to your desktop.")
else:
    print(f"Error: {model_file} not found. Ensure you saved your Decision Tree model correctly.")