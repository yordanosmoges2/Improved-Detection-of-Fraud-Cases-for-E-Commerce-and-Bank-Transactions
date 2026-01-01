 Improved Detection of Fraud Cases for E-Commerce and Bank Transactions

**Author:** Yordanos Moges  
**Company Context:** Adey Innovations Inc.  
**Project Type:** Machine Learning – Fraud Detection  

---

## 📌 Project Overview

This project focuses on detecting fraudulent transactions in **e-commerce** and **banking** datasets using machine learning techniques. Fraud detection is a challenging problem due to **extreme class imbalance** and **complex user behavior patterns**.

The project is structured into **three main tasks**:

- **Task 1:** Data understanding, exploratory data analysis (EDA), feature engineering, and preprocessing  
- **Task 2:** Model training, ensemble methods, cross-validation, hyperparameter tuning, and model selection  
- **Task 3:** Model explainability using SHAP and actionable business insights  

The overall goal is to build **robust, well-evaluated, and explainable fraud detection models** while following best practices in machine learning and repository organization.

---

## 📂 Project Structure

Improved-Detection-of-Fraud-Cases/
│
├── data/
│ └── raw/
│ ├── Fraud_Data.csv
│ ├── creditcard.csv
│ └── IpAddress_to_Country.csv
│
├── notebooks/
│ ├── task1.ipynb # EDA, feature engineering, preprocessing
│ ├── task2.ipynb # Modeling, CV, tuning, model selection
│ ├── task3.ipynb # SHAP explainability and business insights
│ └── creditdata.ipynb
│
├── src/
│ ├── eda.py # Reusable EDA functions
│ └── preprocessing.py # Reusable preprocessing utilities
│
├── models/ # Saved models (placeholder)
│ └── .gitkeep
│
├── reports/ # Reports and figures (placeholder)
│ └── .gitkeep
│
├── tests/ # Future tests (placeholder)
│ └── .gitkeep
│
├── .gitignore
├── requirements.txt
└── README.md



## 🧠 Task 1: Data Understanding & Preprocessing

### Objectives
- Understand dataset structure and data quality  
- Perform exploratory data analysis (EDA)  
- Engineer meaningful behavioral and temporal features  
- Prepare a clean, model-ready dataset  

### Key Steps
- Missing value and duplicate analysis  
- Datetime conversion (`signup_time`, `purchase_time`)  
- Feature engineering:
  - Hour of day
  - Day of week
  - Time since signup
  - Transaction count per user
  - IP address to country mapping  
- One-hot encoding of categorical variables  
- Feature scaling  
- Stratified train–test split  

Reusable preprocessing and EDA logic is implemented in:
- `src/eda.py`
- `src/preprocessing.py`

---

## ⚖️ Handling Class Imbalance

Fraud detection datasets are **highly imbalanced**, with fraudulent transactions forming a very small minority.

To address this issue:

- **SMOTE (Synthetic Minority Oversampling Technique)** is applied **only on the training set**
- This avoids data leakage and ensures fair evaluation

After applying SMOTE:
- **Class 0 (Non-Fraud):** 93,502  
- **Class 1 (Fraud):** 93,502  

---

## 🤖 Task 2: Modeling, Cross-Validation & Model Selection

### Models Trained
- Logistic Regression (baseline and interpretable)
- Random Forest (ensemble)
- Gradient Boosting (ensemble)

### Cross-Validation
- Stratified 5-fold cross-validation
- Evaluation metrics:
  - ROC-AUC
  - F1-score
  - Recall  

Mean and standard deviation of metrics are reported for each model.

### Hyperparameter Tuning
- `RandomizedSearchCV` applied to:
  - Random Forest
  - Gradient Boosting  
- Optimization target: **ROC-AUC**

### Model Selection
- Logistic Regression offers interpretability but weaker fraud detection performance
- Ensemble models significantly improve fraud recall and overall performance

Final model selection is based on:
- Cross-validated ROC-AUC
- Fraud-class recall and F1-score
- Stability across folds

Final evaluation is performed on a **held-out test set**.

---

## 🔍 Task 3: Model Explainability with SHAP

To interpret model predictions and understand fraud drivers:

- Built-in Random Forest feature importance is extracted
- SHAP global feature importance is visualized
- SHAP force plots are generated for:
  - True Positive (correctly detected fraud)
  - False Positive (legitimate transaction flagged as fraud)
  - False Negative (missed fraud case)

This analysis provides transparency and supports actionable business decisions.

---

## 📊 Evaluation Metrics

Models are evaluated using:
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC

Special emphasis is placed on **fraud recall**, due to the high cost of missed fraudulent transactions.

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.2
matplotlib==3.9.1
seaborn==0.12.3
imblearn==0.11.1
jupyter==2.4.0


Install dependencies with:

```bash
pip install -r requirements.txt
▶️ How to Run
Clone the repository

Install dependencies

Open Jupyter Notebook

Run notebooks in order:

notebooks/task1.ipynb

notebooks/task2.ipynb

notebooks/task3.ipynb

✅ Project Status
✔ Task 1 completed (EDA & preprocessing)

✔ Task 2 completed (model training, CV & tuning)

✔ Task 3 completed (SHAP explainability)

✔ Repository follows best practices

✔ Ready for further optimization and deployment

🚀 Future Improvements
Persist trained models in models/

Add automated tests

Deploy as an API or batch scoring pipeline



