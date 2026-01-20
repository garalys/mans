"""
OP2 Monthly Bridge Builder

Creates monthly benchmark bridges comparing actual performance to OP2 targets.
Includes all four levels: Country Total, Country x Business, EU Total, EU x Business.
"""

import pandas as pd
import numpy as np

from ..config.logging_config import logger
from ..calculators.carrier_calculator import calculate_active_carriers
from .op2_data_extractor import extract_op2_monthly_base_cpkm, extract_op2_monthly_base_by_business
from .op2_normalizer import compute_op2_normalized_cpkm_monthly
from .op2_impacts import (
    calculate_op2_carrier_demand_impacts,
    calculate_op2_tech_impact_monthly,
    calculate_op2_market_rate_impact_monthly,
)
from .op2_helpers import get_set_impact_for_op2_monthly


def create_op2_monthly_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 monthly benchmark bridge at country-total level.

    Compares actual monthly performance against OP2 targets, including
    all impact calculations (carrier, demand, tech, market, SET).

    Input Table - df (actual data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_month     | string | Month in MXX format            |
        | orig_country     | string | Origin country code            |
        | total_cost_usd   | float  | Cost in USD                    |
        | distance_for_cpkm| float  | Distance in kilometers         |
        | executed_loads   | int    | Number of loads                |

    Input Table - df_op2:
        OP2 benchmark data with monthly_agg and monthly types

    Input Table - df_carrier:
        Carrier scaling percentages

    Input Table - final_bridge_df:
        Completed MTD bridge for SET and market rate impact lookup

    Output Table:
        Standard OP2 monthly bridge with all impact columns
    """
    logger.info("Creating OP2 country-total monthly bridge...")

    # Aggregate actual monthly data
    actual = (
        df[df["report_month"].notna()]
        .groupby(["report_year", "report_month", "orig_country"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance_km=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    # Calculate carriers
    actual["actual_carriers"] = actual.apply(
        lambda r: calculate_active_carriers(
            df[
                (df["report_year"] == r["report_year"])
                & (df["report_month"] == r["report_month"])
                & (df["orig_country"] == r["orig_country"])
            ],
            r["report_month"],
            r["orig_country"],
        ),
        axis=1,
    )

    actual["compare_cpkm"] = actual["actual_cost"] / actual["actual_distance_km"]

    # Get OP2 metrics
    op2_base = extract_op2_monthly_base_cpkm(df_op2)
    op2_norm = compute_op2_normalized_cpkm_monthly(df, df_op2, by_business=False)
    tech_impact = calculate_op2_tech_impact_monthly(df, df_op2, by_business=False)
    market_impact = calculate_op2_market_rate_impact_monthly(df, df_op2, final_bridge_df, by_business=False)

    # Merge components
    bridge = (
        actual
        .merge(op2_base, on=["report_year", "report_month", "orig_country"], how="left")
        .merge(op2_norm, on=["report_year", "report_month", "orig_country"], how="left")
        .merge(tech_impact, on=["report_year", "report_month", "orig_country"], how="left")
        .merge(market_impact, on=["report_year", "report_month", "orig_country"], how="left")
    )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_monthly"
    bridge["business"] = "Total"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge["op2_normalized_cpkm"]

    # Get SET impact from MTD bridge
    mtd_set_impact = get_set_impact_for_op2_monthly(final_bridge_df)
    mtd_set_impact = mtd_set_impact[mtd_set_impact["business"] == "Total"]
    bridge = bridge.merge(
        mtd_set_impact[["report_year", "report_month", "orig_country", "set_impact"]],
        on=["report_year", "report_month", "orig_country"],
        how="left",
    )

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = np.where(
        bridge["op2_base_loads"] > 0,
        (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100,
        0,
    )

    bridge["distance_variance_km"] = bridge["actual_distance_km"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = np.where(
        bridge["op2_base_distance"] > 0,
        (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100,
        0,
    )

    bridge["cpkm_variance"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]
    bridge["cpkm_variance_pct"] = np.where(
        bridge["op2_base_cpkm"] > 0,
        (bridge["cpkm_variance"] / bridge["op2_base_cpkm"]) * 100,
        0,
    )

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = np.where(
        bridge["op2_base_cost"] > 0,
        ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100,
        0,
    )

    # Normalized metrics
    bridge["normalized_variance"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
    bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate per-km impacts
    bridge["tech_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_tech_impact_value"] / bridge["actual_distance_km"],
        0,
    )

    bridge["market_rate_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_market_impact"] / bridge["actual_distance_km"],
        0,
    )

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_month"] + "_OP2"
    bridge["m2_distance_km"] = bridge["actual_distance_km"]
    bridge["benchmark_gap"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate carrier and demand impacts
    bridge[["carrier_impact", "demand_impact"]] = bridge.apply(
        lambda r: calculate_op2_carrier_demand_impacts(
            actual_loads=r["actual_loads"],
            actual_carriers=r["actual_carriers"],
            op2_loads=r["op2_base_loads"],
            op2_carriers=r["op2_carriers"],
            base_cpkm=r["op2_base_cpkm"],
            country=r["orig_country"],
            df_carrier=df_carrier,
            compare_year=int(r["report_year"].replace("R", "")),
            report_month=r["report_month"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    # Null out non-applicable fields
    for col in ["premium_impact", "supply_rates", "report_week"]:
        bridge[col] = None

    logger.info(f"Created {len(bridge)} OP2 country-total monthly bridge rows")
    return bridge


def create_op2_monthly_country_business_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 monthly benchmark bridge at country x business level.

    Same as create_op2_monthly_bridge but with business dimension.
    """
    logger.info("Creating OP2 country x business monthly bridge...")

    # Aggregate actual monthly data by business
    actual = (
        df[df["report_month"].notna()]
        .groupby(["report_year", "report_month", "orig_country", "business"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance_km=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    # Normalize business
    actual["business"] = actual["business"].str.upper()

    # Calculate carriers
    actual["actual_carriers"] = actual.apply(
        lambda r: calculate_active_carriers(
            df[
                (df["report_year"] == r["report_year"])
                & (df["report_month"] == r["report_month"])
                & (df["orig_country"] == r["orig_country"])
            ],
            r["report_month"],
            r["orig_country"],
        ),
        axis=1,
    )

    actual["compare_cpkm"] = actual["actual_cost"] / actual["actual_distance_km"]

    # Get OP2 metrics at business level
    op2_base = extract_op2_monthly_base_by_business(df_op2)
    op2_base["business"] = op2_base["business"].str.upper()

    op2_norm = compute_op2_normalized_cpkm_monthly(df, df_op2, by_business=True)
    op2_norm["business"] = op2_norm["business"].str.upper()

    tech_impact = calculate_op2_tech_impact_monthly(df, df_op2, by_business=True)
    tech_impact["business"] = tech_impact["business"].str.upper()

    market_impact = calculate_op2_market_rate_impact_monthly(df, df_op2, final_bridge_df, by_business=True)
    market_impact["business"] = market_impact["business"].str.upper()

    # Merge components
    bridge = (
        actual
        .merge(op2_base, on=["report_year", "report_month", "orig_country", "business"], how="left")
        .merge(op2_norm, on=["report_year", "report_month", "orig_country", "business"], how="left")
        .merge(tech_impact, on=["report_year", "report_month", "orig_country", "business"], how="left")
        .merge(market_impact, on=["report_year", "report_month", "orig_country", "business"], how="left")
    )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_monthly"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge["op2_normalized_cpkm"]

    # Get SET impact from MTD bridge
    mtd_set_impact = get_set_impact_for_op2_monthly(final_bridge_df)
    mtd_set_impact["business"] = mtd_set_impact["business"].str.upper()
    bridge = bridge.merge(
        mtd_set_impact[["report_year", "report_month", "orig_country", "business", "set_impact"]],
        on=["report_year", "report_month", "orig_country", "business"],
        how="left",
    )

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = np.where(
        bridge["op2_base_loads"] > 0,
        (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100,
        0,
    )

    bridge["distance_variance_km"] = bridge["actual_distance_km"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = np.where(
        bridge["op2_base_distance"] > 0,
        (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100,
        0,
    )

    bridge["cpkm_variance"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]
    bridge["cpkm_variance_pct"] = np.where(
        bridge["op2_base_cpkm"] > 0,
        (bridge["cpkm_variance"] / bridge["op2_base_cpkm"]) * 100,
        0,
    )

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = np.where(
        bridge["op2_base_cost"] > 0,
        ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100,
        0,
    )

    # Normalized metrics
    bridge["normalized_variance"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
    bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate per-km impacts
    bridge["tech_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_tech_impact_value"].fillna(0) / bridge["actual_distance_km"],
        0,
    )

    bridge["market_rate_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_market_impact"].fillna(0) / bridge["actual_distance_km"],
        0,
    )

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_month"] + "_OP2"
    bridge["m2_distance_km"] = bridge["actual_distance_km"]
    bridge["benchmark_gap"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate carrier and demand impacts
    bridge[["carrier_impact", "demand_impact"]] = bridge.apply(
        lambda r: calculate_op2_carrier_demand_impacts(
            actual_loads=r["actual_loads"],
            actual_carriers=r["actual_carriers"],
            op2_loads=r["op2_base_loads"] if pd.notna(r["op2_base_loads"]) else r["actual_loads"],
            op2_carriers=r["op2_carriers"] if pd.notna(r["op2_carriers"]) else r["actual_carriers"],
            base_cpkm=r["op2_normalized_cpkm"] if pd.notna(r["op2_normalized_cpkm"]) else r["compare_cpkm"],
            country=r["orig_country"],
            df_carrier=df_carrier,
            compare_year=int(r["report_year"].replace("R", "")),
            report_month=r["report_month"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    # Null out non-applicable fields
    for col in ["premium_impact", "supply_rates", "report_week"]:
        bridge[col] = None

    logger.info(f"Created {len(bridge)} OP2 country x business monthly bridge rows")
    return bridge
