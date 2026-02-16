"""
OP2 Weekly Bridge Builder

Creates weekly benchmark bridges comparing actual performance to OP2 targets.
"""

import pandas as pd
import numpy as np

from ..config.logging_config import logger
from ..calculators.carrier_calculator import calculate_active_carriers
from ..calculators.mix_calculator import (
    compute_hierarchical_mix,
    compute_normalised_distance,
    compute_seven_metrics,
    compute_mix_impacts,
    compute_cell_cpkm,
    compute_op2_cell_cpkm,
)
from .op2_data_extractor import extract_op2_weekly_base_cpkm, extract_op2_weekly_base_by_business
from .op2_normalizer import compute_op2_normalized_cpkm_weekly
from .op2_impacts import (
    calculate_op2_carrier_demand_impacts,
    calculate_op2_tech_impact,
    calculate_op2_market_rate_impact,
)
# from .op2_helpers import get_set_impact_for_op2  # Commented out - replaced by equipment_type_mix


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
            actual_distance_km=("distance_for_cpkm", "sum"),
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

    actual["compare_cpkm"] = actual["actual_cost"] / actual["actual_distance_km"]

    # Get OP2 metrics
    op2_base = extract_op2_weekly_base_cpkm(df_op2)
    op2_norm = compute_op2_normalized_cpkm_weekly(df, df_op2)
    tech_impact = calculate_op2_tech_impact(df, df_op2, by_business=False)
    market_impact_op2 = calculate_op2_market_rate_impact(df, df_op2, final_bridge_df, by_business=False)

    # Merge all components
    bridge = (
        actual
        .merge(op2_base, on=["report_year", "report_week", "orig_country"], how="left")
        .merge(op2_norm, on=["report_year", "report_week", "orig_country"], how="left")
        .merge(tech_impact, on=["report_year", "report_week", "orig_country"], how="left")
        .merge(market_impact_op2, on=["report_year", "report_week", "orig_country"], how="left")
    )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_weekly"
    bridge["business"] = "Total"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]

    # Ensure dtypes are strings for consistency
    for col in ["report_year", "report_week", "orig_country", "business"]:
        bridge[col] = bridge[col].astype(str)

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100

    bridge["distance_variance_km"] = bridge["actual_distance_km"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100

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

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_week"] + "_OP2"
    bridge["w2_distance_km"] = bridge["actual_distance_km"]
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

    # Initialize hierarchical mix columns
    # OP2 mix decomposition: country_mix through business_flow_mix from OP2 granular;
    # equipment_type_mix from YoY bridge (per plan)
    for col in ["country_mix", "corridor_mix", "distance_band_mix", "business_flow_mix", "equipment_type_mix"]:
        bridge[col] = None

    # Compute hierarchical mix decomposition per (year, week, country)
    _compute_op2_mix_decomposition(df, df_op2, bridge, time_col="report_week", bridge_type_filter="weekly")

    return bridge


def _compute_op2_mix_decomposition(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    bridge: pd.DataFrame,
    time_col: str = "report_week",
    bridge_type_filter: str = "weekly",
) -> None:
    """
    Compute hierarchical mix decomposition for OP2 bridges (in-place).

    Base mix comes from OP2 granular data (with is_op2_base=True, hardcoding
    RET=100%, SET=0%). Compare mix comes from actual data.

    Updates bridge DataFrame in-place with:
        country_mix, corridor_mix, distance_band_mix, business_flow_mix, equipment_type_mix
    """
    # Determine the OP2 bridge type filter and column mapping
    if bridge_type_filter == "weekly":
        op2_type = "weekly"
        op2_time_col_raw = "Week"
        op2_time_col = "report_week"
    else:  # monthly
        op2_type = "monthly_bridge"
        op2_time_col_raw = "Report Month"
        op2_time_col = "report_month"

    # Extract OP2 granular data
    op2 = df_op2[df_op2["Bridge type"] == op2_type].copy()
    if op2.empty:
        return

    op2 = op2.rename(columns={
        "Report Year": "report_year",
        op2_time_col_raw: op2_time_col,
        "Orig_EU5": "orig_country",
        "Dest_EU5": "dest_country",
        "Business Flow": "business",
        "Distance Band": "distance_band",
        "CpKM": "op2_cpkm",
        "Distance": "distance_for_cpkm",
    })

    for col in ["report_year", op2_time_col, "orig_country", "dest_country", "business", "distance_band"]:
        op2[col] = op2[col].astype(str)
    op2["business"] = op2["business"].str.upper()

    # Prepare actual data
    actual_df = df.copy()
    actual_df["business"] = actual_df["business"].str.upper()
    actual_df["report_year"] = actual_df["report_year"].astype(str)
    actual_df["distance_band"] = (
        actual_df["distance_band"]
        .astype(str)
        .str.strip()
        .str.replace(r"^\d+\.", "", regex=True)
    )

    # Determine if bridge is at total or business level
    is_total = "Total" in bridge["business"].values

    # Iterate over each unique (year, time_period) — compute mixes from ALL countries
    for _, grp in bridge.groupby(["report_year", time_col]):
        year = grp["report_year"].iloc[0]
        period = grp[time_col].iloc[0]

        # Get OP2 data for this period — ALL countries (full network)
        op2_slice = op2[
            (op2["report_year"] == year)
            & (op2[op2_time_col] == period)
        ]

        # Get actual data for this period — ALL countries (full network)
        actual_slice = actual_df[
            (actual_df["report_year"] == year)
            & (actual_df[time_col] == period)
        ]

        if op2_slice.empty or actual_slice.empty:
            continue

        # Compute hierarchical mixes from full network data
        base_mix = compute_hierarchical_mix(op2_slice, distance_col="distance_for_cpkm", is_op2_base=True)
        compare_mix = compute_hierarchical_mix(actual_slice)

        # Compute normalised distance
        norm_dist = compute_normalised_distance(base_mix, compare_mix)

        # Compute cell-level CPKMs
        base_cpkm_cells = compute_op2_cell_cpkm(op2_slice)
        compare_cpkm_cells = compute_cell_cpkm(actual_slice)

        # Compute seven metrics
        seven = compute_seven_metrics(
            base_mix, compare_mix, norm_dist, base_cpkm_cells, compare_cpkm_cells
        )

        # Total compare distance (full network)
        compare_dist = actual_slice["distance_for_cpkm"].sum()

        # Use EU-level aggregation since we have all countries
        bridge_level = "total" if is_total else "business"

        mix_results = compute_mix_impacts(
            seven, compare_dist,
            bridge_level=bridge_level,
            aggregation_level="eu",
        )

        # Write mix columns to ALL rows matching this time period (all countries)
        mask = (
            (bridge["report_year"] == year)
            & (bridge[time_col] == period)
        )

        for mix_col in ["country_mix", "corridor_mix", "distance_band_mix", "business_flow_mix", "equipment_type_mix"]:
            if mix_results.get(mix_col) is not None:
                bridge.loc[mask, mix_col] = mix_results[mix_col]

        # mix_impact = sum of individual mix columns; normalised_cpkm = base_cpkm + mix_impact
        if mix_results.get("mix_impact") is not None:
            bridge.loc[mask, "mix_impact"] = mix_results["mix_impact"]
            bridge.loc[mask, "normalised_cpkm"] = bridge.loc[mask, "base_cpkm"] + mix_results["mix_impact"]


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
            actual_distance_km=("distance_for_cpkm", "sum"),
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

    actual["compare_cpkm"] = actual["actual_cost"] / actual["actual_distance_km"]

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

    # Calculate variance metrics using np.where to handle missing values
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

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = np.where(
        bridge["op2_base_cost"] > 0,
        ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100,
        0,
    )

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

    # Initialize hierarchical mix columns
    for col in ["country_mix", "corridor_mix", "distance_band_mix", "business_flow_mix", "equipment_type_mix"]:
        bridge[col] = None

    # Compute hierarchical mix decomposition per (year, week, country)
    _compute_op2_mix_decomposition(df, df_op2, bridge, time_col="report_week", bridge_type_filter="weekly")

    return bridge
