# 🩺 Stroke Prediction ML Project (Databricks)

## 📘 Overview
This project predicts the likelihood of stroke in patients using demographic and clinical data.  
It was built entirely on **Databricks**, leveraging **PySpark, Scikit-Learn, SMOTE**, and **Plotly** for model training, class balancing, and inline visualization — without external dashboards.

---

## ⚙️ Tech Stack
- **Environment:** Databricks Community Edition  
- **Languages:** Python (Pandas, PySpark, Plotly, Scikit-Learn)  
- **Modeling:** RandomForest, Logistic Regression  
- **Feature Engineering:** One-hot encoding, SMOTE balancing, risk bucketing  
- **Visualization:** Plotly interactive charts, inline Databricks dashboards  
- **MLOps Tools:** MLflow model logging, exportable `.pkl` models  

---

## 🧠 Key Steps

### 1️⃣ Data Preprocessing
- Cleaned and encoded categorical variables (`gender`, `smoking_status`, `work_type`, etc.)
- Applied **SMOTE** to balance the dataset between stroke and non-stroke cases.

### 2️⃣ Model Training
- Built **Logistic Regression** and **RandomForest** models.
- Achieved **ROC-AUC = 0.775** with balanced recall for positive class.

### 3️⃣ Feature Importance
- Top predictors: `age`, `avg_glucose_level`, `bmi`.

### 4️⃣ Inline Databricks Dashboard
- Accuracy & ROC metrics table  
- Probability distribution by class  
- Confusion matrix heatmap  
- Feature importance ranking  
- Risk segmentation and demographic breakdowns  

### 5️⃣ Model Deployment (Optional)
- Logged via **MLflow** with preprocessing encoders.
- Ready for downstream deployment or API integration.

---

## 🔍 Results
| Metric | Value |
|---------|-------|
| Accuracy | 91.1% |
| ROC-AUC | 0.775 |
| Balanced Recall (after SMOTE) | 0.26 for positive class |

---

## 🧩 Project Value
- End-to-end ML lifecycle on Databricks  
- Real-world healthcare data and imbalance handling  
- Inline visual analytics and SHAP-based explainability  
- Deployable model logged via MLflow  

---

### 🏷️ Badges
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Databricks](https://img.shields.io/badge/Platform-Databricks-orange)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-green)
![ScikitLearn](https://img.shields.io/badge/ML-ScikitLearn-yellow)
![Plotly](https://img.shields.io/badge/Viz-Plotly-purple)
