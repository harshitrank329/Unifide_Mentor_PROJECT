from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BASE_FEATURE_COLUMNS, prepare_dataset, train_test_time_split

TARGET_COLUMN = "lead_time_days"

@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]


@dataclass
class ModelBundle:
    # Container for the trained model family used by the app.

    champion_name: str
    champion_pipeline: Pipeline
    results: List[ModelResult]
    feature_columns: List[str]


def _make_preprocessor() -> ColumnTransformer:
    # Categorical columns are one-hot encoded, while numeric columns are scaled.
    categorical = [
        "Division",
        "Region",
        "Ship Mode",
        "Product Name",
        "Product ID",
        "Factory",
    ]
    numeric = [
        "Sales",
        "Units",
        "Gross Profit",
        "Cost",
        "profit_margin",
        "order_year",
        "order_month",
        "order_day_of_week",
    ]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def _make_model_pipelines() -> Dict[str, Pipeline]:
    # Create the small model set used in the student-friendly version.
    # This keeps the comparison simple: one linear model and two tree-based models.
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", _make_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", _make_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=10,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", _make_preprocessor()),
                (
                    "model",
                    GradientBoostingRegressor(
                        learning_rate=0.05,
                        n_estimators=120,
                        max_depth=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:

    # RMSE, MAE, and R2 are the standard metrics used in this project.
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_model_suite(csv_path: str) -> ModelBundle:

    # Train the small model comparison set and return the best performer.
    raw_df = pd.read_csv(csv_path)
    data = prepare_dataset(raw_df)
    train_df, test_df = train_test_time_split(data)

    # Build, train, and score each model on the same train/test split.
    model_pipelines = _make_model_pipelines()
    results: List[ModelResult] = []

    x_train = train_df[BASE_FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[BASE_FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    for name, pipeline in model_pipelines.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = _evaluate(y_test, predictions)
        results.append(ModelResult(name=name, pipeline=pipeline, metrics=metrics))

    champion = min(results, key=lambda item: item.metrics["rmse"])

    # The champion model is the one with the lowest RMSE.
    return ModelBundle(
        champion_name=champion.name,
        champion_pipeline=champion.pipeline,
        results=results,
        feature_columns=BASE_FEATURE_COLUMNS,
    )


def evaluate_model_suite(bundle: ModelBundle, csv_path: str) -> pd.DataFrame:
    # Return a simple table of model metrics for the dashboard.
    rows = []
    for result in bundle.results:
        rows.append({"model": result.name, **result.metrics})
    return pd.DataFrame(rows).sort_values("rmse")
