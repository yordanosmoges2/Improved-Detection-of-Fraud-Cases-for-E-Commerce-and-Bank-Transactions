Improved Detection of Fraud Cases for E-Commerce and Bank Transactions

Author: Yordanos Moges
Project: Fraud Detection for E-Commerce and Bank Transactions
Company Context: Adey Innovations Inc.

📌 Project Overview

This project focuses on detecting fraudulent transactions in e-commerce and banking datasets using machine learning. Fraud detection is challenging due to extreme class imbalance and complex user behavior patterns.

The project is organized into two main tasks:

Task 1: Data understanding, exploratory data analysis (EDA), feature engineering, and preprocessing

Task 2: Model training, ensemble methods, cross-validation, hyperparameter tuning, and model selection

The final goal is to build reliable, well-evaluated fraud detection models while following best practices in machine learning and repository structure.

📂 Project Structure
Improved-Detection-of-Fraud-Cases/
│
├── data/
│   └── raw/
│       ├── Fraud_Data.csv
│       ├── creditcard.csv
│       └── IpAddress_to_Country.csv
│
├── notebooks/
│   ├── task1.ipynb          # EDA, feature engineering, preprocessing
│   ├── task2.ipynb          # Modeling, CV, tuning, model selection
│   └── creditdata.ipynb
│
├── src/
│   ├── eda.py               # Reusable EDA functions
│   └── preprocessing.py    # Reusable preprocessing utilities
│
├── models/                  # Saved models (placeholder)
│   └── .gitkeep
│
├── reports/                 # Reports, figures, and results (placeholder)
│   └── .gitkeep
│
├── tests/                   # Future tests (placeholder)
│   └── .gitkeep
│
├── .gitignore
├── requirements.txt
└── README.md

🧠 Task 1: Data Understanding & Preprocessing
Objectives

Understand dataset structure and quality

Perform exploratory data analysis (EDA)

Engineer meaningful behavioral and temporal features

Prepare a clean, model-ready dataset

Key Steps

Missing value and duplicate analysis

Datetime conversion (signup_time, purchase_time)

Feature engineering:

Hour of day

Day of week

Time since signup

Transaction count per user

IP address to country mapping

One-hot encoding of categorical features

Feature scaling

Stratified train–test split

Reusable code is implemented in:

src/eda.py

src/preprocessing.py

⚖️ Handling Class Imbalance

Fraud data is highly imbalanced, with fraudulent transactions forming a small minority.

To address this:

SMOTE (Synthetic Minority Oversampling Technique) is applied only to the training set

This avoids data leakage and ensures fair evaluation

After SMOTE:

Class 0: 93,502

Class 1: 93,502

🤖 Task 2: Modeling, Cross-Validation & Model Selection
Models Trained

Logistic Regression (baseline, interpretable)

Random Forest (ensemble)

Gradient Boosting (ensemble)

Cross-Validation

Stratified 5-fold cross-validation

Metrics evaluated:

ROC-AUC

F1-score

Recall

Mean and standard deviation of metrics are reported for each model

Hyperparameter Tuning

RandomizedSearchCV applied to:

Random Forest

Gradient Boosting

Optimization target: ROC-AUC

Best estimators selected based on cross-validated performance

Model Selection

Logistic Regression provides interpretability but weaker fraud-class performance

Ensemble models significantly improve fraud detection metrics

The final model is selected based on:

Cross-validated ROC-AUC

Fraud recall and F1-score

Stability across folds

Final evaluation is performed on a held-out test set.

📊 Evaluation Metrics

Models are evaluated using:

Confusion Matrix

Precision, Recall, F1-score

ROC-AUC

Special emphasis is placed on fraud recall due to the high cost of false negatives.

📦 Dependencies

All dependencies are listed in requirements.txt:

pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.2
matplotlib==3.9.1
seaborn==0.12.3
imblearn==0.11.1
jupyter==2.4.0


Install with:

pip install -r requirements.txt

▶️ How to Run

Clone the repository

Install dependencies

Open Jupyter Notebook

Run notebooks in order:

notebooks/task1.ipynb

notebooks/task2.ipynb

✅ Project Status

✔ Task 1 completed (EDA & preprocessing)

✔ Task 2a completed (model training)

✔ Task 2b completed (cross-validation & tuning)

✔ Repository follows best practices

✔ Ready for further optimization and deployment

🚀 Future Improvements

Add SHAP-based model explainability

Persist trained models in models/

Add automated tests

Deploy as an API or batch scoring pipeline

