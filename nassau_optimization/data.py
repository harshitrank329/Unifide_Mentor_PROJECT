from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

# This file contains functions for loading and preparing the dataset, as well as some constants about the factories and products.
FACTORY_COORDINATES: Dict[str, tuple[float, float]] = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}

# This lookup table tells the app which factory currently produces each product.
CURRENT_FACTORY_BY_PRODUCT: Dict[str, str] = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

# These are the main input columns the model uses after feature engineering.
BASE_FEATURE_COLUMNS = [
    "Division",
    "Region",
    "Ship Mode",
    "Product Name",
    "Product ID",
    "Factory",
    "Sales",
    "Units",
    "Gross Profit",
    "Cost",
    "profit_margin",
    "order_year",
    "order_month",
    "order_day_of_week",
]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    # Load the Nassau Candy CSV with safe date parsing.
    path = Path(csv_path)
    df = pd.read_csv(path)
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Convert date columns to datetime, coercing errors to NaT so we can drop them later.
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:

    # Create model-ready features and target columns.
    frame = _parse_dates(df)
    frame = frame.dropna(subset=["Order Date", "Ship Date"])

    # Lead time is the target we want to predict.
    frame["lead_time_days"] = (frame["Ship Date"] - frame["Order Date"]).dt.days

    # Profit margin helps the dashboard compare speed and profitability.
    frame["profit_margin"] = np.where(
        frame["Sales"].astype(float) != 0,
        frame["Gross Profit"].astype(float) / frame["Sales"].astype(float),
        0.0,
    )
    frame["order_year"] = frame["Order Date"].dt.year
    frame["order_month"] = frame["Order Date"].dt.month
    frame["order_day_of_week"] = frame["Order Date"].dt.dayofweek

    # Use the provided product-to-factory mapping as the current assignment.
    frame["Factory"] = frame["Product Name"].map(CURRENT_FACTORY_BY_PRODUCT).fillna("Unknown")

    # Convert numeric-looking columns to numbers so the model can use them.
    numeric_cols = ["Sales", "Units", "Gross Profit", "Cost", "lead_time_days"]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Remove incomplete rows and trim extreme values so the model is more stable.
    frame = frame.dropna(subset=["lead_time_days", "Sales", "Units", "Gross Profit", "Cost"])
    frame = _clip_outliers(frame, ["lead_time_days", "Sales", "Units", "Gross Profit", "Cost"])
    frame = frame.loc[frame["lead_time_days"] >= 0].reset_index(drop=True)
    return frame


def _clip_outliers(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    # Clip extreme values to the 1st and 99th percentiles to reduce noise in the model training.
    clipped = df.copy()
    for column in columns:
        lower = clipped[column].quantile(0.01)
        upper = clipped[column].quantile(0.99)
        clipped[column] = clipped[column].clip(lower, upper)
    return clipped


def train_test_time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Split the data into train and test sets based on order date.
    # Sort by date so the model is tested on later orders, which is realistic for business data.
    ordered = df.sort_values("Order Date").reset_index(drop=True)
    split_index = max(1, int(len(ordered) * train_ratio))
    train_df = ordered.iloc[:split_index].copy()
    test_df = ordered.iloc[split_index:].copy()
    return train_df, test_df
