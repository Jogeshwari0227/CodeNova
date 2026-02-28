# 🎓 EduInsight AI — Attendance & Performance Correlation

> **PS 05** | AI-Based Attendance & Performance Correlation System

## 📁 Project Structure
```
CODENOVA/
├── app/
│   └── app.py                  ← Main Streamlit application
├── data/
│   └── StudentPerformanceFactors.csv
├── model/
│   ├── train_model.py          ← Train & save the ML model
│   └── model.pkl               ← Saved model (generated after training)
├── utils/
│   └── preprocess.py           ← Data cleaning & feature engineering
└── requirements.txt
```

## 🚀 Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run this from the project root (`CODENOVA/`):
```bash
python model/train_model.py
```
Expected output:
```
Dataset loaded: 6607 rows, 20 columns
✅ Model saved to model/model.pkl
   R² Score : 0.XX
   MAE      : X.XX marks
```

### 3. Launch the App
```bash
streamlit run app/app.py
```

Then open `http://localhost:8501` in your browser.

### 🔐 Login Credentials
| Field    | Value        |
|----------|--------------|
| Username | `admin`      |
| Password | `samsung123` |

---

## ✨ Features

| Feature | Description |
|---|---|
| 📈 **Correlation Analysis** | Scatter trend line, Pearson coefficient, score-by-attendance-band bar chart |
| 📊 **Distribution Charts** | Attendance & score histograms, motivation-level breakdown |
| 🤖 **Grade Estimator** | RandomForest ML prediction from 5 behavioral inputs |
| 🎯 **Strategic Intervention** | Dynamic, rule-based educator guidance per student |
| 📐 **What-If Scenarios** | Compare predicted score across +attendance, +study hours, +tutoring |
| 🗃️ **Data Explorer** | Heatmap, top/bottom performers, filterable full dataset |

---

## 🧠 Model Details
- **Algorithm:** RandomForestRegressor (200 trees, max_depth=10)
- **Features:** Attendance, Hours_Studied, Previous_Scores, Tutoring_Sessions, Motivation_Level
- **Target:** Exam_Score (continuous regression)
- **Evaluation:** R² Score + Mean Absolute Error