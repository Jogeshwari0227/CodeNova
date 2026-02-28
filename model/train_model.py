import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

# Ensure local imports work
sys.path.append(os.getcwd())
from utils.preprocess import preprocess_data

def train():
    # Load dataset
    if not os.path.exists('data/StudentPerformanceFactors.csv'):
        print("Error: StudentPerformanceFactors.csv not found!")
        return

    df = pd.read_csv('data/StudentPerformanceFactors.csv')
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Regressor (Predicting continuous marks)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save Model
    if not os.path.exists('model'): os.makedirs('model')
    joblib.dump(model, 'model/model.pkl')
    
    score = model.score(X_test, y_test)
    print(f"Model trained. Accuracy (R2 Score): {score:.2f}")

if __name__ == "__main__":
    train()