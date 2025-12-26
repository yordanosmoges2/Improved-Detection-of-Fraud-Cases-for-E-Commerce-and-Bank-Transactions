# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, keep dataframe safe."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")
    df = df.drop_duplicates()
    return df


def add_time_features(df: pd.DataFrame,
                      signup_col="signup_time",
                      purchase_col="purchase_time") -> pd.DataFrame:
    """Add hour_of_day, day_of_week, and time_since_signup (seconds)."""
    df = df.copy()
    df[signup_col] = pd.to_datetime(df[signup_col])
    df[purchase_col] = pd.to_datetime(df[purchase_col])

    df["hour_of_day"] = df[purchase_col].dt.hour
    df["day_of_week"] = df[purchase_col].dt.dayofweek
    df["time_since_signup"] = (df[purchase_col] - df[signup_col]).dt.total_seconds()
    return df


def add_behavior_features(df: pd.DataFrame, user_col="user_id") -> pd.DataFrame:
    """Add txn_count_user (transaction frequency per user)."""
    df = df.copy()
    df["txn_count_user"] = df.groupby(user_col)[user_col].transform("count")
    return df


def ip_to_country_merge(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fraud data with ip range data using merge_asof.
    Assumes fraud_df has 'ip_address' and ip_df has:
    'lower_bound_ip_address', 'upper_bound_ip_address', 'country'
    """
    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    # strip possible hidden spaces in columns
    fraud_df.columns = fraud_df.columns.str.strip()
    ip_df.columns = ip_df.columns.str.strip()

    fraud_df["ip_address"] = fraud_df["ip_address"].astype("int64")
    ip_df["lower_bound_ip_address"] = ip_df["lower_bound_ip_address"].astype("int64")
    ip_df["upper_bound_ip_address"] = ip_df["upper_bound_ip_address"].astype("int64")

    fraud_df = fraud_df.sort_values("ip_address")
    ip_df = ip_df.sort_values("lower_bound_ip_address")

    merged = pd.merge_asof(
        fraud_df,
        ip_df,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    merged = merged[merged["ip_address"] <= merged["upper_bound_ip_address"]]
    return merged


def prepare_X_y(df: pd.DataFrame, target_col="class", drop_cols=None):
    """Split into X and y. Optionally drop columns from X."""
    if drop_cols is None:
        drop_cols = []
    y = df[target_col]
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    return X, y


def one_hot_encode(X: pd.DataFrame, cat_cols):
    """One-hot encode the given categorical columns."""
    return pd.get_dummies(X, columns=cat_cols, drop_first=True)


def scale_numeric(X: pd.DataFrame, numeric_cols):
    """Standard scale numeric columns. Returns (X_scaled, scaler)."""
    X = X.copy()
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return X, scaler


def drop_non_numeric_and_ids(X: pd.DataFrame) -> pd.DataFrame:
    """Drop object columns + common ID columns that break SMOTE."""
    X = X.copy()
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = X.drop(columns=obj_cols, errors="ignore")

    # Also drop typical ID/range columns if they exist
    drop_cols = [
        "device_id",
        "ip_address",
        "lower_bound_ip_address",
        "upper_bound_ip_address",
        "lower_bound_ip_address_x",
        "upper_bound_ip_address_x",
        "lower_bound_ip_address_y",
        "upper_bound_ip_address_y",
        "country_x",
        "country_y",
    ]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
    return X
