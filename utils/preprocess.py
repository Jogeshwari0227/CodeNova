import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()

    # 1. Handle Missing Values
    df['Teacher_Quality'] = df['Teacher_Quality'].fillna('Medium')
    df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna('High School')
    df['Distance_from_Home'] = df['Distance_from_Home'].fillna('Moderate')

    # 2. Encode Categorical Variables
    le = LabelEncoder()
    categorical_cols = [
        'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
        'Gender', 'Extracurricular_Activities', 'Internet_Access',
        'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence',
        'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # 3. Core features for attendance & performance correlation
    features = [
        'Attendance',
        'Hours_Studied',
        'Previous_Scores',
        'Tutoring_Sessions',
    ]

    if 'Exam_Score' in df.columns:
        return df[features], df['Exam_Score']
    return df[features]