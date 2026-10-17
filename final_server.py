import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import pickle

print("1. Loading the Dataset...")
df = pd.read_csv('CLEAN_physiological_data.csv')
df = df.select_dtypes(include=['number'])

print("2. Scrubbing hidden spaces from column names...")
df.columns = df.columns.str.strip()

print("3. Forcing the deletion of redundant parameters...")
# Using a smarter drop method to catch the columns even if they have weird formatting
cols_to_drop = [col for col in df.columns if 'Tcore' in col or 'CVC' in col or 'Mean Tskin' in col]
X = df.drop(columns=cols_to_drop) 
y = df['Tcore']

print(f"\n VERIFICATION: Your AI brain has exactly {X.shape[1]} parameters.")
if X.shape[1] != 11:
    print("STOP! It still has the wrong number of columns!")
else:
    print("Perfect! 11 parameters confirmed. Training the brain...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', GradientBoostingRegressor(random_state=42))])
    pipeline.fit(X_train, y_train)

    print("\n 4. Saving the Brain directly to your folder...")
    with open(r'C:\Users\candi\OneDrive\Desktop\Heat_Stress_App\heat_stress_model.pkl', 'wb') as file:
        pickle.dump(pipeline, file)
    print("DONE! The true 11-parameter model is locked in.")