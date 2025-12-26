Improved Detection of Fraud Cases for E-Commerce and Bank Transactions
📌 Project Overview

This project focuses on detecting fraudulent transactions in e-commerce and banking data.
The goal of Task 1 is to perform data cleaning, exploratory data analysis (EDA), feature engineering, and preprocessing to prepare a high-quality dataset for machine learning models.

📂 Project Structure
kifiya-week5/
│
├── data/
│   └── raw/
│       ├── Fraud_Data.csv
│       ├── creditcard.csv
│       └── IpAddress_to_Country.csv
│
├── notebooks/
│   ├── task1.ipynb
│   └── creditdata.ipynb
│
├── src/
│   ├── eda.py
│   └── preprocessing.py
│
├── .gitignore
├── requirements.txt
└── README.md

🧠 Task 1 Objectives

Understand the structure of fraud datasets

Perform exploratory data analysis (EDA)

Engineer meaningful features

Handle missing values and duplicates

Encode categorical variables

Scale numerical features

Handle class imbalance using SMOTE

🔍 Exploratory Data Analysis (EDA)

EDA includes:

Class distribution analysis

Transaction amount distribution

Feature inspection and summary statistics

Visualization of fraud vs non-fraud patterns

Reusable plotting functions are implemented in:

src/eda.py

🛠️ Data Preprocessing

Key preprocessing steps:

Removal of duplicates

Conversion of time columns to datetime

Feature engineering:

Hour of day

Day of week

Time since signup

Transaction count per user

IP address to country mapping

One-hot encoding of categorical features

Feature scaling using StandardScaler

Train-test split with stratification

Reusable preprocessing utilities are implemented in:

src/preprocessing.py

⚖️ Handling Class Imbalance

Fraud datasets are highly imbalanced.
To address this:

SMOTE (Synthetic Minority Oversampling Technique) is applied only to the training set

This prevents data leakage and ensures fair evaluation

After SMOTE:

Class 0: 93,502

Class 1: 93,502

📦 Dependencies

All required packages are listed in requirements.txt:

pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.2
matplotlib==3.9.1
seaborn==0.12.3
imblearn==0.11.1
jupyter==2.4.0


Install dependencies using:

pip install -r requirements.txt

▶️ How to Run

Clone the repository

Install dependencies

Open Jupyter Notebook

Run:

notebooks/task1.ipynb

✅ Task 1 Status

✔ Completed
✔ Matches project requirements
✔ Data ready for modeling (Task 2)

🚀 Next Steps

Train machine learning models

Evaluate performance using precision, recall, F1-score, and ROC-AUC

Compare baseline and advanced models

