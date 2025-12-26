# src/eda.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(y, title="Class Distribution"):
    y.value_counts().plot(kind="bar")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


def plot_histogram(df, column, bins=50, title=None):
    sns.histplot(df[column], bins=bins)
    plt.title(title or f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def plot_box_by_class(df, target_col, feature_col, title=None):
    sns.boxplot(x=target_col, y=feature_col, data=df)
    plt.title(title or f"{feature_col} by {target_col}")
    plt.show()


def plot_fraud_rate_by_category(df, category_col, target_col="class", title=None, rotate=False):
    sns.barplot(
        x=category_col,
        y=target_col,
        data=df,
        estimator=lambda x: sum(x) / len(x)
    )
    plt.title(title or f"Fraud Rate by {category_col}")
    if rotate:
        plt.xticks(rotation=45)
    plt.show()
