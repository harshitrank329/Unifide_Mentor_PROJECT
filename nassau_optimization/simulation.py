from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import BASE_FEATURE_COLUMNS, FACTORY_COORDINATES, prepare_dataset
from .modeling import ModelBundle

@dataclass
class ScenarioRecommendation:
    # One ranked factory reassignment option for a product.

    product_name: str
    region: str
    ship_mode: str
    current_factory: str
    recommended_factory: str
    current_predicted_lead_time: float
    recommended_predicted_lead_time: float
    lead_time_reduction_pct: float
    profit_impact_proxy: float
    risk_score: float
    composite_score: float
    scenario_confidence: float


def _route_subset(frame: pd.DataFrame, product_name: str, region: str | None, ship_mode: str | None) -> pd.DataFrame:
    # Filter the historical orders for one product and optional user filters.
    
    # This narrows the data to the exact scenario the user wants to study.
    subset = frame.loc[frame["Product Name"] == product_name].copy()
    if region:
        subset = subset.loc[subset["Region"] == region]
    if ship_mode:
        subset = subset.loc[subset["Ship Mode"] == ship_mode]
    return subset


def _scenario_frame(subset: pd.DataFrame, candidate_factory: str) -> pd.DataFrame:
    # Create a copy of the data as if the product came from another factory.
    # Only the factory changes here; everything else stays the same.
    scenario = subset.copy()
    scenario["Factory"] = candidate_factory
    return scenario


def _mean_prediction(bundle: ModelBundle, frame: pd.DataFrame) -> float:
    # Predict average lead time for a group of orders.
    
    # We average predictions so each candidate factory is scored at the route level.
    if frame.empty:
        return float("nan")
    values = bundle.champion_pipeline.predict(frame[bundle.feature_columns])
    return float(np.mean(values))


def _scenario_confidence(subset: pd.DataFrame) -> float:
    """Convert lead-time variability into a simple 0-1 confidence score."""
   
    # More variation in historical lead time means less confidence in the recommendation.
    if subset.empty:
        return 0.0
    variability = float(subset["lead_time_days"].std(ddof=0) or 0.0)
    base = 1.0 / (1.0 + variability / 100.0)
    return float(np.clip(base, 0.0, 1.0))


def recommend_factory_reassignments(
    csv_path: str,
    bundle: ModelBundle,
    product_name: str,
    region: str | None = None,
    ship_mode: str | None = None,
    speed_weight: float = 0.7,
    top_n: int = 5,
) -> pd.DataFrame:
    # Rank alternative factories from best to worst for one product.
    raw_df = pd.read_csv(csv_path)
    prepared = prepare_dataset(raw_df)
    subset = _route_subset(prepared, product_name=product_name, region=region, ship_mode=ship_mode)

    # If there is no history for this selection, return an empty table for the dashboard.
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "product_name",
                "region",
                "ship_mode",
                "current_factory",
                "recommended_factory",
                "current_predicted_lead_time",
                "recommended_predicted_lead_time",
                "lead_time_reduction_pct",
                "profit_impact_proxy",
                "risk_score",
                "composite_score",
                "scenario_confidence",
            ]
        )

    current_factory = subset["Factory"].mode().iat[0]
    current_prediction = _mean_prediction(bundle, subset)
    confidence = _scenario_confidence(subset)
    historical_profit_margin = float(subset["profit_margin"].mean())

    # Score each alternate factory with a simple weighted formula.
    rows: List[ScenarioRecommendation] = []
    for candidate_factory in FACTORY_COORDINATES:
        if candidate_factory == current_factory:
            continue
        scenario = _scenario_frame(subset, candidate_factory)
        scenario_prediction = _mean_prediction(bundle, scenario)
        if np.isnan(scenario_prediction):
            continue
        lead_time_reduction_pct = 0.0
        if current_prediction and current_prediction > 0:
            lead_time_reduction_pct = ((current_prediction - scenario_prediction) / current_prediction) * 100.0
        profit_impact_proxy = float(historical_profit_margin * lead_time_reduction_pct)
        risk_score = float(np.clip(1.0 - confidence + max(0.0, -profit_impact_proxy) / 100.0, 0.0, 1.0))
        composite_score = float(
            speed_weight * lead_time_reduction_pct + (1.0 - speed_weight) * profit_impact_proxy - risk_score * 5.0
        )
        rows.append(
            ScenarioRecommendation(
                product_name=product_name,
                region=region or subset["Region"].mode().iat[0],
                ship_mode=ship_mode or subset["Ship Mode"].mode().iat[0],
                current_factory=current_factory,
                recommended_factory=candidate_factory,
                current_predicted_lead_time=current_prediction,
                recommended_predicted_lead_time=scenario_prediction,
                lead_time_reduction_pct=lead_time_reduction_pct,
                profit_impact_proxy=profit_impact_proxy,
                risk_score=risk_score,
                composite_score=composite_score,
                scenario_confidence=confidence,
            )
        )

    recommendations = pd.DataFrame([item.__dict__ for item in rows])
    
    # Show only the top recommendations to keep the output easy to read.
    recommendations = recommendations.sort_values("composite_score", ascending=False).head(top_n).reset_index(drop=True)
    return recommendations


def build_route_clusters(csv_path: str, n_clusters: int = 5) -> pd.DataFrame:
    
    # Group similar routes together so students can spot slow or busy patterns.
    raw_df = pd.read_csv(csv_path)
    prepared = prepare_dataset(raw_df)
    route_summary = (
        prepared.groupby(["Product Name", "Region", "Ship Mode", "Factory"], dropna=False)
        .agg(
            lead_time_days_mean=("lead_time_days", "mean"),
            lead_time_days_std=("lead_time_days", "std"),
            sales_mean=("Sales", "mean"),
            units_mean=("Units", "mean"),
            gross_profit_mean=("Gross Profit", "mean"),
            profit_margin_mean=("profit_margin", "mean"),
            order_count=("Row ID", "count"),
        )
        .reset_index()
    )

    features = route_summary[
        ["lead_time_days_mean", "lead_time_days_std", "sales_mean", "units_mean", "gross_profit_mean", "profit_margin_mean", "order_count"]
    ].fillna(0.0)
    
    # Standard scaling helps the clustering algorithm treat each feature fairly.
    scaled = StandardScaler().fit_transform(features)
    route_summary["route_cluster"] = KMeans(n_clusters=min(n_clusters, len(route_summary)), random_state=42, n_init=10).fit_predict(scaled)
    return route_summary.sort_values(["route_cluster", "lead_time_days_mean"], ascending=[True, False]).reset_index(drop=True)
