import sys
from pathlib import Path
import pandas as pd

# add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import ip_to_country_merge


def test_ip_to_country_merge_basic():
    fraud = pd.DataFrame({
        "ip_address": [10, 20, 30]
    })

    ip = pd.DataFrame({
        "lower_bound_ip_address": [0, 15],
        "upper_bound_ip_address": [14, 40],
        "country": ["A", "B"]
    })

    merged = ip_to_country_merge(fraud, ip)

    # assertions required by rubric
    assert "country" in merged.columns
    assert (merged["ip_address"] <= merged["upper_bound_ip_address"]).all()



