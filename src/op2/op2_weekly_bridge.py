"""
OP2 Weekly Bridge Builder

Creates weekly benchmark bridges comparing actual performance to OP2 targets.
"""

import pandas as pd
import numpy as np

from ..config.logging_config import logger
from ..calculators.carrier_calculator import calculate_active_carriers
from .op2_data_extractor import extract_op2_weekly_base_cpkm, extract_op2_weekly_base_by_business
from .op2_normalizer import compute_op2_normalized_cpkm_weekly
from .op2_impacts import (
    calculate_op2_carrier_demand_impacts,
    calculate_op2_tech_impact,
    calculate_op2_market_rate_impact,
)
from .op2_helpers import get_set_impact_for_op2


def create_op2_weekly_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 weekly benchmark bridge at country-total level.

    Compares actual weekly performance against OP2 targets, calculating
    variance metrics and bridge components.

    Input Table - df (actual data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_week      | string | Week in WXX format             |
        | orig_country     | string | Origin country code            |
        | total_cost_usd   | float  | Cost in USD                    |
        | distance_for_cpkm| float  | Distance in kilometers         |
        | executed_loads   | int    | Number of loads                |
        | vehicle_carrier  | string | Carrier identifier             |
        | + other columns for impact calculations                      |

    Input Table - df_op2:
        OP2 benchmark data (see op2_data_extractor for structure)

    Input Table - df_carrier:
        Carrier scaling percentages (see carrier_calculator for structure)

    Input Table - final_bridge_df:
        Completed YoY/WoW/MTD bridge for SET impact lookup

    Output Table:
        | Column                    | Type   | Description                    |
        |---------------------------|--------|--------------------------------|
        | report_year               | string | Year in R20XX format           |
        | report_week               | string | Week in WXX format             |
        | orig_country              | string | Origin country code            |
        | business                  | string | Always 'Total'                 |
        | bridge_type               | string | Always 'op2_weekly'            |
        | bridging_value            | string | Format: 'R2025_W01_OP2'        |
        | actual_cost               | float  | Actual total cost              |
        | actual_distance           | float  | Actual total distance          |
        | actual_loads              | int    | Actual load count              |
        | actual_carriers           | int    | Actual carrier count           |
        | compare_cpkm              | float  | Actual CPKM                    |
        | base_cpkm                 | float  | OP2 baseline CPKM              |
        | normalised_cpkm           | float  | OP2 normalized CPKM            |
        | mix_impact                | float  | Mix impact ($/km)              |
        | carrier_impact            | float  | Carrier impact ($/km)          |
        | demand_impact             | float  | Demand impact ($/km)           |
        | carrier_and_demand_impact | float  | Combined carrier/demand        |
        | tech_impact               | float  | Tech impact ($/km)             |
        | market_rate_impact        | float  | Market rate impact ($/km)      |
        | set_impact                | float  | SET impact (from YoY)          |
        | *_variance                | float  | Variance metrics               |
        | benchmark_gap             | float  | Actual - OP2 CPKM              |
    """
    # Aggregate actual weekly data
    actual = (
        df[df["report_week"].notna()]
        .groupby(["report_year", "report_week", "orig_country"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    # Calculate carriers using same logic as YoY
    actual["actual_carriers"] = actual.apply(
        lambda r: calculate_active_carriers(
            df[
                (df["report_year"] == r["report_year"])
                & (df["report_week"] == r["report_week"])
                & (df["orig_country"] == r["orig_country"])
            ],
            r["report_week"],
            r["orig_country"],
        ),
        axis=1,
    )

    actual["compare_cpkm"] = actual["actual_cost"] / actual["actual_distance"]

    # Get OP2 metrics
    op2_base = extract_op2_weekly_base_cpkm(df_op2)
    op2_norm = compute_op2_normalized_cpkm_weekly(df, df_op2)
    tech_impact = calculate_op2_tech_impact(df, df_op2, by_business=False)
    market_impact_op2 = calculate_op2_market_rate_impact(df, df_op2, final_bridge_df, by_business=False)

    logger.info(f"op2_norm shape: {op2_norm.shape}")
    logger.info(f"op2_norm columns: {op2_norm.columns.tolist()}")

    # Log columns for debugging
    logger.info(f"actual columns: {actual.columns.tolist()}")
    logger.info(f"actual shape: {actual.shape}")
    logger.info(f"op2_base columns: {op2_base.columns.tolist()}")
    logger.info(f"tech_impact columns: {tech_impact.columns.tolist()}")
    logger.info(f"market_impact_op2 columns: {market_impact_op2.columns.tolist()}")

    # Merge all components step by step for debugging
    bridge = actual.merge(op2_base, on=["report_year", "report_week", "orig_country"], how="left")
    logger.info(f"After op2_base merge - columns: {bridge.columns.tolist()}")

    bridge = bridge.merge(op2_norm, on=["report_year", "report_week", "orig_country"], how="left")
    logger.info(f"After op2_norm merge - columns: {bridge.columns.tolist()}")

    bridge = bridge.merge(tech_impact, on=["report_year", "report_week", "orig_country"], how="left")
    logger.info(f"After tech_impact merge - columns: {bridge.columns.tolist()}")

    bridge = bridge.merge(market_impact_op2, on=["report_year", "report_week", "orig_country"], how="left")
    logger.info(f"After market_impact merge - columns: {bridge.columns.tolist()}")

    # Fill bridge schema
    bridge["bridge_type"] = "op2_weekly"
    bridge["business"] = "Total"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge["op2_normalized_cpkm"]

    # Get SET impact from YoY bridge
    yoy_set_impact = get_set_impact_for_op2(final_bridge_df).copy()
    logger.info(f"yoy_set_impact columns: {yoy_set_impact.columns.tolist()}")
    logger.info(f"yoy_set_impact shape: {yoy_set_impact.shape}")

    # Ensure dtypes match for merge
    for col in ["report_year", "report_week", "orig_country", "business"]:
        bridge[col] = bridge[col].astype(str)
        if col in yoy_set_impact.columns:
            yoy_set_impact[col] = yoy_set_impact[col].astype(str)

    bridge = bridge.merge(
        yoy_set_impact,
        on=["report_year", "report_week", "orig_country", "business"],
        how="left",
    )
    logger.info(f"After yoy_set_impact merge - columns: {bridge.columns.tolist()}")
    logger.info(f"After yoy_set_impact merge - shape: {bridge.shape}")

    # Sanity check - ensure required columns exist
    required_cols = ["actual_distance", "actual_cost", "actual_loads", "op2_base_distance", "op2_base_cost", "op2_base_loads"]
    missing_cols = [col for col in required_cols if col not in bridge.columns]
    if missing_cols:
        logger.error(f"Missing required columns after merges: {missing_cols}")
        logger.error(f"Available columns: {bridge.columns.tolist()}")
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100

    bridge["distance_variance_km"] = bridge["actual_distance"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100

    # Normalized metrics
    bridge["compare_cpkm_vs_normalised_op2_cpkm"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
    bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]

    bridge["tech_impact"] = np.where(
        bridge["actual_distance"] > 0,
        bridge["op2_tech_impact_value"] / bridge["actual_distance"],
        0,
    )

    bridge["market_rate_impact"] = np.where(
        bridge["actual_distance"] > 0,
        bridge["op2_market_impact"] / bridge["actual_distance"],
        0,
    )

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_week"] + "_OP2"
    bridge["w2_distance_km"] = bridge["actual_distance"]
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
            report_week=r["report_week"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    # Null out non-applicable fields
    for col in ["premium_impact", "supply_rates", "op2_active_carriers", "op2_lcr"]:
        bridge[col] = None

    return bridge


def create_op2_weekly_country_business_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 weekly bridge at country x business level.

    Same as create_op2_weekly_bridge but with business dimension.

    Input Tables:
        Same as create_op2_weekly_bridge

    Output Table:
        Same structure as create_op2_weekly_bridge but:
        - business column contains actual business values (not 'Total')
        - Granularity is country x week x business
    """
    # Aggregate actual weekly data by business
    actual = (
        df[df["report_week"].notna()]
        .groupby(["report_year", "report_week", "orig_country", "business"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    actual["actual_carriers"] = actual.apply(
        lambda r: calculate_active_carriers(
            df[
                (df["report_year"] == r["report_year"])
                & (df["report_week"] == r["report_week"])
                & (df["orig_country"] == r["orig_country"])
            ],
            r["report_week"],
            r["orig_country"],
        ),
        axis=1,
    )

    actual["compare_cpkm"] = actual["actual_cost"] / actual["actual_distance"]

    # Get OP2 metrics at business level
    op2_base = extract_op2_weekly_base_by_business(df_op2)
    op2_norm = compute_op2_normalized_cpkm_weekly(df, df_op2, by_business=True)
    tech_impact = calculate_op2_tech_impact(df, df_op2, by_business=True)
    market_impact_op2 = calculate_op2_market_rate_impact(df, df_op2, final_bridge_df, by_business=True)

    logger.info(f"op2_norm shape: {op2_norm.shape}")

    # Merge all components
    bridge = (
        actual
        .merge(op2_base, on=["report_year", "report_week", "orig_country", "business"], how="left")
        .merge(op2_norm, on=["report_year", "report_week", "orig_country", "business"], how="left")
        .merge(tech_impact, on=["report_year", "report_week", "orig_country", "business"], how="left")
        .merge(market_impact_op2, on=["report_year", "report_week", "orig_country", "business"], how="left")
    )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_weekly"
    bridge["business"] = bridge["business"].fillna("UNKNOWN")
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge["op2_normalized_cpkm"]

    # Get SET impact from YoY bridge
    yoy_set_impact = get_set_impact_for_op2(final_bridge_df).copy()

    # Ensure dtypes match for merge
    for col in ["report_year", "report_week", "orig_country", "business"]:
        bridge[col] = bridge[col].astype(str)
        if col in yoy_set_impact.columns:
            yoy_set_impact[col] = yoy_set_impact[col].astype(str)

    bridge = bridge.merge(
        yoy_set_impact,
        on=["report_year", "report_week", "orig_country", "business"],
        how="left",
    )

    # Calculate variance metrics using np.where to handle missing values
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = np.where(
        bridge["op2_base_loads"] > 0,
        (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100,
        0,
    )

    bridge["distance_variance_km"] = bridge["actual_distance"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = np.where(
        bridge["op2_base_distance"] > 0,
        (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100,
        0,
    )

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = np.where(
        bridge["op2_base_cost"] > 0,
        ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100,
        0,
    )

    bridge["compare_cpkm_vs_normalised_op2_cpkm"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
    bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]

    bridge["tech_impact"] = np.where(
        bridge["actual_distance"] > 0,
        bridge["op2_tech_impact_value"] / bridge["actual_distance"],
        0,
    )

    bridge["market_rate_impact"] = np.where(
        bridge["actual_distance"] > 0,
        bridge["op2_market_impact"] / bridge["actual_distance"],
        0,
    )

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_week"] + "_OP2"

    # Calculate carrier and demand impacts
    bridge[["carrier_impact", "demand_impact"]] = bridge.apply(
        lambda r: calculate_op2_carrier_demand_impacts(
            actual_loads=r["actual_loads"],
            actual_carriers=r["actual_carriers"],
            op2_loads=r["actual_loads"],  # OP2 expected = normalized
            op2_carriers=r["op2_carriers"],
            base_cpkm=r["op2_normalized_cpkm"],
            country=r["orig_country"],
            df_carrier=df_carrier,
            compare_year=int(r["report_year"].replace("R", "")),
            report_week=r["report_week"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    return bridge
