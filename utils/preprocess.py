import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # 1. Handle Missing Values
    df['Teacher_Quality'] = df['Teacher_Quality'].fillna('Medium')
    df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna('High School')
    df['Distance_from_Home'] = df['Distance_from_Home'].fillna('Moderate')

    # 2. Encode Categorical Variables
    le = LabelEncoder()
    categorical_cols = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Gender']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # 3. Define Features for the problem statement (Focus on Attendance and Behavior)
    features = [
        'Attendance', 
        'Hours_Studied', 
        'Previous_Scores', 
        'Tutoring_Sessions', 
        'Motivation_Level'
    ]
    
    if 'Exam_Score' in df.columns:
        return df[features], df['Exam_Score']
    return df[features]