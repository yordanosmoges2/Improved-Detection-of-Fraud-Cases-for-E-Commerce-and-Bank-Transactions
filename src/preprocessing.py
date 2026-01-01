import pandas as pd



def ip_to_country_merge(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fraud transactions with IP-to-country ranges using merge_asof.

    Parameters
    ----------
    fraud_df : pd.DataFrame
        Must contain 'ip_address'.
    ip_df : pd.DataFrame
        Must contain lower/upper IP bound columns and 'country'. Column-name variants
        are handled (e.g., spaces vs underscores).

    Returns
    -------
    pd.DataFrame
        fraud_df with country + IP range columns merged in (where ip_address falls
        within [lower_bound_ip_address, upper_bound_ip_address]).
    """
    if fraud_df is None or fraud_df.empty:
        raise ValueError("fraud_df is empty.")
    if ip_df is None or ip_df.empty:
        raise ValueError("ip_df is empty.")

    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    # normalize columns (strip, lowercase, spaces->underscores)
    fraud_df.columns = fraud_df.columns.str.strip().str.lower().str.replace(" ", "_")
    ip_df.columns = ip_df.columns.str.strip().str.lower().str.replace(" ", "_")

    # validate fraud required column
    if "ip_address" not in fraud_df.columns:
        raise KeyError(f"fraud_df missing 'ip_address'. Found: {list(fraud_df.columns)}")

    # rename common variants in ip_df
    rename_map = {}
    if "lower_bound_ip_address" not in ip_df.columns:
        for c in ip_df.columns:
            if "lower" in c and "ip" in c:
                rename_map[c] = "lower_bound_ip_address"
                break
    if "upper_bound_ip_address" not in ip_df.columns:
        for c in ip_df.columns:
            if "upper" in c and "ip" in c:
                rename_map[c] = "upper_bound_ip_address"
                break
    if "country" not in ip_df.columns:
        for c in ip_df.columns:
            if "country" in c:
                rename_map[c] = "country"
                break
    if rename_map:
        ip_df = ip_df.rename(columns=rename_map)

    # validate ip_df required columns
    required_ip = {"lower_bound_ip_address", "upper_bound_ip_address", "country"}
    missing = required_ip - set(ip_df.columns)
    if missing:
        raise KeyError(f"ip_df missing columns {missing}. Found: {list(ip_df.columns)}")

    # drop old merge columns from fraud_df to avoid _x/_y confusion
    fraud_df = fraud_df.drop(
        columns=["lower_bound_ip_address", "upper_bound_ip_address", "country"],
        errors="ignore",
    )

    # robust numeric conversion (handles '123.0' and '1,234')
    def _clean_to_numeric(s: pd.Series) -> pd.Series:
        return pd.to_numeric(
            s.astype(str)
             .str.replace(r"\.0$", "", regex=True)
             .str.replace(",", "", regex=False)
             .str.strip(),
            errors="coerce"
        )

    try:
        fraud_df["ip_address"] = _clean_to_numeric(fraud_df["ip_address"])
        ip_df["lower_bound_ip_address"] = _clean_to_numeric(ip_df["lower_bound_ip_address"])
        ip_df["upper_bound_ip_address"] = _clean_to_numeric(ip_df["upper_bound_ip_address"])
    except Exception as e:
        raise ValueError(f"IP conversion failed: {e}")

    # drop rows where conversion failed
    fraud_df = fraud_df.dropna(subset=["ip_address"]).copy()
    ip_df = ip_df.dropna(subset=["lower_bound_ip_address", "upper_bound_ip_address"]).copy()

    # convert to int64 for merge_asof
    fraud_df["ip_address"] = fraud_df["ip_address"].astype("int64")
    ip_df["lower_bound_ip_address"] = ip_df["lower_bound_ip_address"].astype("int64")
    ip_df["upper_bound_ip_address"] = ip_df["upper_bound_ip_address"].astype("int64")

    # sort required by merge_asof
    fraud_df = fraud_df.sort_values("ip_address")
    ip_df = ip_df.sort_values("lower_bound_ip_address")

    # merge
    try:
        merged = pd.merge_asof(
            fraud_df,
            ip_df,
            left_on="ip_address",
            right_on="lower_bound_ip_address",
            direction="backward",
            suffixes=("", "_ip"),
        )
    except Exception as e:
        raise RuntimeError(f"merge_asof failed: {e}")

    # find correct upper bound column (handles suffix cases)
    upper_col = None
    for cand in [
        "upper_bound_ip_address",
        "upper_bound_ip_address_ip",
        "upper_bound_ip_address_y",
        "upper_bound_ip_address_x",
    ]:
        if cand in merged.columns:
            upper_col = cand
            break

    if upper_col is None:
        raise KeyError(f"Upper bound column not found. Columns: {list(merged.columns)}")

    # keep only valid in-range rows
    merged = merged[merged["ip_address"] <= merged[upper_col]].copy()
    return merged


