def ip_to_country_merge(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    # normalize columns
    fraud_df.columns = fraud_df.columns.str.strip().str.lower().str.replace(" ", "_")
    ip_df.columns = ip_df.columns.str.strip().str.lower().str.replace(" ", "_")

    # rename common variants in ip_df
    rename_map = {}
    if "lower_bound_ip_address" not in ip_df.columns:
        for c in ip_df.columns:
            if "lower" in c and "ip" in c:
                rename_map[c] = "lower_bound_ip_address"; break
    if "upper_bound_ip_address" not in ip_df.columns:
        for c in ip_df.columns:
            if "upper" in c and "ip" in c:
                rename_map[c] = "upper_bound_ip_address"; break
    if "country" not in ip_df.columns:
        for c in ip_df.columns:
            if "country" in c:
                rename_map[c] = "country"; break
    if rename_map:
        ip_df = ip_df.rename(columns=rename_map)

    # drop old merge columns from fraud_df (prevents _x/_y confusion)
    fraud_df = fraud_df.drop(
        columns=["lower_bound_ip_address", "upper_bound_ip_address", "country"],
        errors="ignore"
    )

    # convert to numeric cleanly
    fraud_df["ip_address"] = pd.to_numeric(fraud_df["ip_address"], errors="coerce")
    ip_df["lower_bound_ip_address"] = pd.to_numeric(ip_df["lower_bound_ip_address"], errors="coerce")
    ip_df["upper_bound_ip_address"] = pd.to_numeric(ip_df["upper_bound_ip_address"], errors="coerce")

    fraud_df = fraud_df.dropna(subset=["ip_address"]).copy()
    ip_df = ip_df.dropna(subset=["lower_bound_ip_address", "upper_bound_ip_address"]).copy()

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
        direction="backward",
        suffixes=("", "_ip")
    )

    # find correct upper bound column (handles suffix cases)
    upper_col = None
    for cand in ["upper_bound_ip_address", "upper_bound_ip_address_ip",
                 "upper_bound_ip_address_y", "upper_bound_ip_address_x"]:
        if cand in merged.columns:
            upper_col = cand
            break
    if upper_col is None:
        raise KeyError(f"Upper bound column not found. Columns: {list(merged.columns)}")

    merged = merged[merged["ip_address"] <= merged[upper_col]].copy()
    return merged

