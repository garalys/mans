"""
OP2 Quarterly Bridge Builder

Creates quarterly benchmark bridges comparing actual performance to OP2 targets.
Quarterly data available at Country and EU level only (not at business level).
"""

import pandas as pd
import numpy as np

from ..config.logging_config import logger
from ..calculators.carrier_calculator import calculate_active_carriers
from .op2_data_extractor import extract_op2_quarterly_base_cpkm
from .op2_impacts import calculate_op2_carrier_demand_impacts


# Quarter to months mapping
QUARTER_TO_MONTHS = {
    "Q1": ["M01", "M02", "M03"],
    "Q2": ["M04", "M05", "M06"],
    "Q3": ["M07", "M08", "M09"],
    "Q4": ["M10", "M11", "M12"],
}


def _get_quarter_from_month(month: str) -> str:
    """Map month (M01-M12) to quarter (Q1-Q4)."""
    month_num = int(month.replace("M", ""))
    if month_num <= 3:
        return "Q1"
    elif month_num <= 6:
        return "Q2"
    elif month_num <= 9:
        return "Q3"
    else:
        return "Q4"


def create_op2_quarterly_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 quarterly benchmark bridge at country-total level.

    Compares actual quarterly performance against OP2 targets.
    OP2 base metrics from quarterly data (Period = Q1-Q4).
    Other metrics aggregated from monthly calculations.

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
        OP2 benchmark data with monthly_agg containing Q1-Q4 periods

    Input Table - df_carrier:
        Carrier scaling percentages

    Input Table - final_bridge_df:
        Completed bridges for SET and market rate impact lookup

    Output Table:
        Standard OP2 quarterly bridge with all impact columns
    """
    logger.info("Creating OP2 country-total quarterly bridge...")

    # Add quarter column to actual data
    df_quarterly = df[df["report_month"].notna()].copy()
    df_quarterly["report_quarter"] = df_quarterly["report_month"].apply(_get_quarter_from_month)

    # Aggregate actual quarterly data
    actual = (
        df_quarterly
        .groupby(["report_year", "report_quarter", "orig_country"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance_km=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    # Calculate carriers per quarter (unique carriers across all months in quarter)
    actual["actual_carriers"] = actual.apply(
        lambda r: df_quarterly[
            (df_quarterly["report_year"] == r["report_year"])
            & (df_quarterly["report_quarter"] == r["report_quarter"])
            & (df_quarterly["orig_country"] == r["orig_country"])
            & (df_quarterly["executed_loads"] > 0)
        ]["vehicle_carrier"].nunique(),
        axis=1,
    )

    actual["compare_cpkm"] = actual["actual_cost"] / actual["actual_distance_km"]

    # Get OP2 quarterly base metrics
    op2_base = extract_op2_quarterly_base_cpkm(df_op2)

    # Aggregate normalized CPKM from monthly level
    # First, get monthly normalized data from monthly bridges in final_bridge_df
    monthly_op2 = final_bridge_df[
        (final_bridge_df["bridge_type"] == "op2_monthly")
        & (final_bridge_df["business"] == "Total")
        & (final_bridge_df["orig_country"] != "EU")
    ].copy()

    if not monthly_op2.empty:
        monthly_op2["report_quarter"] = monthly_op2["report_month"].apply(_get_quarter_from_month)

        # Aggregate monthly metrics to quarterly
        quarterly_norm = monthly_op2.groupby(
            ["report_year", "report_quarter", "orig_country"],
            as_index=False,
        ).agg(
            op2_normalized_cost=("normalised_cpkm", lambda x: (x * monthly_op2.loc[x.index, "actual_distance_km"]).sum()),
            norm_distance=("actual_distance_km", "sum"),
            op2_tech_impact_value=("tech_impact", lambda x: (x * monthly_op2.loc[x.index, "actual_distance_km"]).sum()),
            op2_market_impact=("market_rate_impact", lambda x: (x * monthly_op2.loc[x.index, "actual_distance_km"]).sum()),
            set_impact_sum=("set_impact", lambda x: (x * monthly_op2.loc[x.index, "actual_distance_km"]).sum() if x.notna().any() else 0),
        )
        quarterly_norm["op2_normalized_cpkm"] = quarterly_norm["op2_normalized_cost"] / quarterly_norm["norm_distance"]
    else:
        quarterly_norm = pd.DataFrame()

    # Merge components
    bridge = actual.merge(
        op2_base,
        on=["report_year", "report_quarter", "orig_country"],
        how="left",
    )

    if not quarterly_norm.empty:
        bridge = bridge.merge(
            quarterly_norm[["report_year", "report_quarter", "orig_country", "op2_normalized_cpkm",
                           "op2_tech_impact_value", "op2_market_impact", "set_impact_sum", "norm_distance"]],
            on=["report_year", "report_quarter", "orig_country"],
            how="left",
        )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_quarterly"
    bridge["business"] = "Total"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge.get("op2_normalized_cpkm", bridge["op2_base_cpkm"])

    # Get SET impact (aggregated from monthly)
    if "set_impact_sum" in bridge.columns and "norm_distance" in bridge.columns:
        bridge["set_impact"] = np.where(
            bridge["norm_distance"] > 0,
            bridge["set_impact_sum"] / bridge["norm_distance"],
            0,
        )
    else:
        bridge["set_impact"] = None

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
    if "op2_normalized_cpkm" in bridge.columns:
        bridge["normalized_variance"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
        bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]
    else:
        bridge["normalized_variance"] = 0
        bridge["mix_impact"] = 0

    # Calculate per-km impacts
    if "op2_tech_impact_value" in bridge.columns:
        bridge["tech_impact"] = np.where(
            bridge["actual_distance_km"] > 0,
            bridge["op2_tech_impact_value"].fillna(0) / bridge["actual_distance_km"],
            0,
        )
    else:
        bridge["tech_impact"] = 0

    if "op2_market_impact" in bridge.columns:
        bridge["market_rate_impact"] = np.where(
            bridge["actual_distance_km"] > 0,
            bridge["op2_market_impact"].fillna(0) / bridge["actual_distance_km"],
            0,
        )
    else:
        bridge["market_rate_impact"] = 0

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_quarter"] + "_OP2"
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
            report_month=r["report_quarter"],  # Use quarter as period for lookup
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    # Null out non-applicable fields
    for col in ["premium_impact", "supply_rates", "report_week", "report_month"]:
        bridge[col] = None

    logger.info(f"Created {len(bridge)} OP2 country-total quarterly bridge rows")
    return bridge


def create_op2_quarterly_country_business_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 quarterly benchmark bridge at country x business level.

    OP2 doesn't have quarterly data at business level, so OP2 base metrics
    are aggregated from monthly_bridge data (M01+M02+M03 = Q1, etc.).
    Other metrics aggregated from monthly business bridges.

    Input Tables:
        Same as create_op2_quarterly_bridge

    Output Table:
        Same structure but with business dimension
    """
    from .op2_data_extractor import extract_op2_monthly_base_by_business

    logger.info("Creating OP2 country x business quarterly bridge...")

    # Add quarter column to actual data
    df_quarterly = df[df["report_month"].notna()].copy()
    df_quarterly["report_quarter"] = df_quarterly["report_month"].apply(_get_quarter_from_month)
    df_quarterly["business"] = df_quarterly["business"].str.upper()

    # Aggregate actual quarterly data by business
    actual = (
        df_quarterly
        .groupby(["report_year", "report_quarter", "orig_country", "business"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance_km=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    # Calculate carriers per quarter by business
    actual["actual_carriers"] = actual.apply(
        lambda r: df_quarterly[
            (df_quarterly["report_year"] == r["report_year"])
            & (df_quarterly["report_quarter"] == r["report_quarter"])
            & (df_quarterly["orig_country"] == r["orig_country"])
            & (df_quarterly["business"] == r["business"])
            & (df_quarterly["executed_loads"] > 0)
        ]["vehicle_carrier"].nunique(),
        axis=1,
    )

    actual["compare_cpkm"] = actual["actual_cost"] / actual["actual_distance_km"]

    # Get OP2 base from monthly business data and aggregate to quarterly
    monthly_base = extract_op2_monthly_base_by_business(df_op2)
    monthly_base["report_quarter"] = monthly_base["report_month"].apply(_get_quarter_from_month)
    monthly_base["business"] = monthly_base["business"].str.upper()

    # Aggregate monthly OP2 base to quarterly
    op2_base = monthly_base.groupby(
        ["report_year", "report_quarter", "orig_country", "business"],
        as_index=False,
    ).agg(
        op2_base_distance=("op2_base_distance", "sum"),
        op2_base_cost=("op2_base_cost", "sum"),
        op2_base_loads=("op2_base_loads", "sum"),
    )
    op2_base["op2_base_cpkm"] = op2_base["op2_base_cost"] / op2_base["op2_base_distance"]

    # Get quarterly carriers from country-level quarterly data
    quarterly_base = extract_op2_quarterly_base_cpkm(df_op2)
    op2_base = op2_base.merge(
        quarterly_base[["report_year", "report_quarter", "orig_country", "op2_carriers"]],
        on=["report_year", "report_quarter", "orig_country"],
        how="left",
    )

    # Aggregate from monthly business bridges for impacts
    monthly_op2 = final_bridge_df[
        (final_bridge_df["bridge_type"] == "op2_monthly")
        & (final_bridge_df["business"] != "Total")
        & (final_bridge_df["orig_country"] != "EU")
    ].copy()

    if not monthly_op2.empty:
        monthly_op2["report_quarter"] = monthly_op2["report_month"].apply(_get_quarter_from_month)
        monthly_op2["business"] = monthly_op2["business"].str.upper()

        quarterly_metrics = monthly_op2.groupby(
            ["report_year", "report_quarter", "orig_country", "business"],
            as_index=False,
        ).agg(
            op2_normalized_cost=("normalised_cpkm", lambda x: (x * monthly_op2.loc[x.index, "actual_distance_km"]).sum()),
            norm_distance=("actual_distance_km", "sum"),
            op2_tech_impact_value=("tech_impact", lambda x: (x * monthly_op2.loc[x.index, "actual_distance_km"]).sum()),
            op2_market_impact=("market_rate_impact", lambda x: (x * monthly_op2.loc[x.index, "actual_distance_km"]).sum()),
            set_impact_sum=("set_impact", lambda x: (x * monthly_op2.loc[x.index, "actual_distance_km"]).sum() if x.notna().any() else 0),
        )
        quarterly_metrics["op2_normalized_cpkm"] = quarterly_metrics["op2_normalized_cost"] / quarterly_metrics["norm_distance"]
    else:
        quarterly_metrics = pd.DataFrame()

    # Merge components
    bridge = actual.merge(
        op2_base,
        on=["report_year", "report_quarter", "orig_country", "business"],
        how="left",
    )

    if not quarterly_metrics.empty:
        bridge = bridge.merge(
            quarterly_metrics[["report_year", "report_quarter", "orig_country", "business",
                              "op2_normalized_cpkm", "op2_tech_impact_value", "op2_market_impact",
                              "set_impact_sum", "norm_distance"]],
            on=["report_year", "report_quarter", "orig_country", "business"],
            how="left",
        )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_quarterly"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge.get("op2_normalized_cpkm", bridge["op2_base_cpkm"])

    # SET impact
    if "set_impact_sum" in bridge.columns and "norm_distance" in bridge.columns:
        bridge["set_impact"] = np.where(
            bridge["norm_distance"] > 0,
            bridge["set_impact_sum"] / bridge["norm_distance"],
            0,
        )
    else:
        bridge["set_impact"] = None

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
    if "op2_normalized_cpkm" in bridge.columns:
        bridge["normalized_variance"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
        bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]
    else:
        bridge["normalized_variance"] = 0
        bridge["mix_impact"] = 0

    # Per-km impacts
    if "op2_tech_impact_value" in bridge.columns:
        bridge["tech_impact"] = np.where(
            bridge["actual_distance_km"] > 0,
            bridge["op2_tech_impact_value"].fillna(0) / bridge["actual_distance_km"],
            0,
        )
    else:
        bridge["tech_impact"] = 0

    if "op2_market_impact" in bridge.columns:
        bridge["market_rate_impact"] = np.where(
            bridge["actual_distance_km"] > 0,
            bridge["op2_market_impact"].fillna(0) / bridge["actual_distance_km"],
            0,
        )
    else:
        bridge["market_rate_impact"] = 0

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_quarter"] + "_OP2"
    bridge["m2_distance_km"] = bridge["actual_distance_km"]
    bridge["benchmark_gap"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate carrier and demand impacts
    bridge[["carrier_impact", "demand_impact"]] = bridge.apply(
        lambda r: calculate_op2_carrier_demand_impacts(
            actual_loads=r["actual_loads"],
            actual_carriers=r["actual_carriers"],
            op2_loads=r["op2_base_loads"] if pd.notna(r["op2_base_loads"]) else r["actual_loads"],
            op2_carriers=r["op2_carriers"] if pd.notna(r["op2_carriers"]) else r["actual_carriers"],
            base_cpkm=r["op2_normalized_cpkm"] if pd.notna(r.get("op2_normalized_cpkm")) else r["compare_cpkm"],
            country=r["orig_country"],
            df_carrier=df_carrier,
            compare_year=int(r["report_year"].replace("R", "")),
            report_month=r["report_quarter"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    # Null out non-applicable fields
    for col in ["premium_impact", "supply_rates", "report_week", "report_month"]:
        bridge[col] = None

    logger.info(f"Created {len(bridge)} OP2 country x business quarterly bridge rows")
    return bridge
