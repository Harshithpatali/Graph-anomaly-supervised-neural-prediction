"""
Data loading, cleaning, IP-to-country enrichment, and chronological splitting utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRAUD_PATH = ROOT / "data" / "raw" / "Fraud_Data.csv"
IP_PATH = ROOT / "data" / "raw" / "IpAddress_to_Country.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two raw source tables."""
    fraud = pd.read_csv(FRAUD_PATH)
    ipmap = pd.read_csv(IP_PATH)
    return fraud, ipmap


def validate_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse signup_time and purchase_time to UTC datetime."""
    x = df.copy()
    for col in ["signup_time", "purchase_time"]:
        if col in x.columns:
            x[col] = pd.to_datetime(x[col], errors="coerce", utc=True)
    return x


def enrich_with_country(fraud: pd.DataFrame, ipmap: pd.DataFrame) -> pd.DataFrame:
    """
    Map numeric IP addresses to countries using range lookup.
    Efficient vectorized approach using sorted intervals + searchsorted.
    """
    x = fraud.copy()
    ipmap = ipmap.sort_values("lower_bound_ip_address").reset_index(drop=True)

    lowers = ipmap["lower_bound_ip_address"].values.astype(np.float64)
    uppers = ipmap["upper_bound_ip_address"].values.astype(np.float64)
    countries = ipmap["country"].values

    ips = x["ip_address"].astype(np.float64).values
    idx = np.searchsorted(lowers, ips, side="right") - 1
    idx = np.clip(idx, 0, len(lowers) - 1)

    # Validate that IP falls inside the chosen interval
    valid = (ips >= lowers[idx]) & (ips <= uppers[idx])
    country = np.where(valid, countries[idx], "Unknown")
    x["country"] = country
    return x


def chronological_split(
    df: pd.DataFrame,
    time_col: str = "purchase_time",
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split on purchase_time.
    Train → Validation → Test (locked until final evaluation).
    """
    x = df.sort_values(time_col).reset_index(drop=True)
    n = len(x)
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + val_frac))
    train = x.iloc[:t_end].copy()
    val = x.iloc[t_end:v_end].copy()
    test = x.iloc[v_end:].copy()
    return train, val, test


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning: drop exact duplicates, ensure positive purchase_value."""
    x = df.drop_duplicates().copy()
    if "purchase_value" in x.columns:
        x = x[x["purchase_value"] > 0]
    return x.reset_index(drop=True)


def save_processed(df: pd.DataFrame, name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_processed(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_parquet(path)
