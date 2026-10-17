import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensembasele import RandomForestRegressor
import pickle

print("1. Loading the Datt...")
df = pd.read_csv('CLEAN_physiological_data.csv')
df = df.select_dtypes(include=['number'])

print("2. Scrubbing column names for safety...")
df.columns = df.columns.str.strip()

print(" 3. Locking in your exact 10 parameters...")
y = df['Tcore']

# THE FIX: We tell Python to drop anything containing these specific words, regardless of brackets or spaces!
cols_to_drop = [col for col in df.columns if any(x in col for x in ['Tcore', 'CVC', 'Tskin', 'Change', 'change'])]
X = df.drop(columns=cols_to_drop, errors='ignore')

print(f"\n VERIFICATION: Your AI brain has exactly {X.shape[1]} parameters.")
if X.shape[1] != 10:
    print("STOP! It does not have exactly 10 columns! Check your CSV headers.")
    print(f"Current columns in X: {X.columns.tolist()}")
else:
    print("Perfect! 10 parameters confirmed. Training the Random Forest brain...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test) * 100
    print(f"▶️ Random Forest Accuracy: {score:.2f}%")

    print("\n 4. Saving the Version 2 Brain directly to your folder...")
    with open(r'C:\Users\candi\OneDrive\Desktop\Heat_Stress_App\heat_stress_v2.pkl', 'wb') as file:
        pickle.dump(pipeline, file)
    print("DONE! The v2 model is locked in.")