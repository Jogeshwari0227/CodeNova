import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import sys

sys.path.append(os.getcwd())
from utils.preprocess import preprocess_data

def train():
    csv_path = 'data/StudentPerformanceFactors.csv'
    if not os.path.exists(csv_path):
        print("Error: StudentPerformanceFactors.csv not found in data/ folder!")
        return

    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.pkl')

    print(f"✅ Model saved to model/model.pkl")
    print(f"   R² Score : {r2:.4f}")
    print(f"   MAE      : {mae:.2f} marks")

if __name__ == "__main__":
    train()