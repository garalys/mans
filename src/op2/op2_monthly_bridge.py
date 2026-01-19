"""
OP2 Monthly Bridge Builder

Creates monthly benchmark bridges comparing actual performance to OP2 targets.
"""

import pandas as pd

from ..calculators.carrier_calculator import calculate_active_carriers
from .op2_data_extractor import extract_op2_monthly_base_cpkm
from .op2_normalizer import compute_op2_normalized_cpkm_monthly
from .op2_impacts import calculate_op2_carrier_demand_impacts


def create_op2_monthly_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 monthly benchmark bridge.

    Compares actual monthly performance against OP2 targets.

    Input Table - df (actual data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_month     | string | Month in MXX format            |
        | orig_country     | string | Origin country code            |
        | total_cost_usd   | float  | Cost in USD                    |
        | distance_for_cpkm| float  | Distance in kilometers         |
        | executed_loads   | int    | Number of loads                |
        | + other columns                                              |

    Input Table - df_op2:
        OP2 benchmark data with monthly_agg and monthly_bridge types

    Input Table - df_carrier:
        Carrier scaling percentages

    Output Table:
        | Column                    | Type   | Description                    |
        |---------------------------|--------|--------------------------------|
        | report_year               | string | Year in R20XX format           |
        | report_month              | string | Month in MXX format            |
        | orig_country              | string | Origin country code            |
        | business                  | string | Always 'Total'                 |
        | bridge_type               | string | Always 'op2_monthly'           |
        | bridging_value            | string | Format: 'R2025_M01_OP2'        |
        | actual_cost               | float  | Actual total cost              |
        | actual_distance           | float  | Actual total distance          |
        | actual_loads              | int    | Actual load count              |
        | actual_carriers           | int    | Actual carrier count           |
        | compare_cpkm              | float  | Actual CPKM                    |
        | op2_*                     | float  | OP2 baseline metrics           |
        | *_variance                | float  | Variance metrics               |
        | mix_impact                | float  | Mix impact ($/km)              |
        | carrier_impact            | float  | Carrier impact ($/km)          |
        | demand_impact             | float  | Demand impact ($/km)           |
        | benchmark_gap             | float  | Actual - OP2 CPKM              |
    """
    # Aggregate actual monthly data
    actual = (
        df[df["report_month"].notna()]
        .groupby(["report_year", "report_month", "orig_country"], as_index=False)
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
                & (df["report_month"] == r["report_month"])
                & (df["orig_country"] == r["orig_country"])
            ],
            r["report_month"],
            r["orig_country"],
        ),
        axis=1,
    )

    actual["compare_cpkm"] = actual["actual_cost"] / actual["actual_distance"]

    # Get OP2 metrics
    op2_base = extract_op2_monthly_base_cpkm(df_op2)
    op2_norm = compute_op2_normalized_cpkm_monthly(df, df_op2)

    # Merge components
    bridge = (
        actual
        .merge(op2_base, on=["report_year", "report_month", "orig_country"], how="left")
        .merge(op2_norm, on=["report_year", "report_month", "orig_country"], how="left")
    )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_monthly"
    bridge["business"] = "Total"

    # Populate OP2 columns
    bridge["op2_loads"] = bridge["op2_base_loads"]
    bridge["op2_distance_km"] = bridge["op2_base_distance"]
    bridge["op2_cost_usd"] = bridge["op2_base_cost"]
    bridge["op2_cpkm"] = bridge["op2_base_cpkm"]

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100

    bridge["distance_variance_km"] = bridge["actual_distance"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100

    bridge["cpkm_variance"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]
    bridge["cpkm_variance_pct"] = (bridge["cpkm_variance"] / bridge["op2_base_cpkm"]) * 100

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100

    # Normalized metrics
    bridge["normalized_variance"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
    bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_month"] + "_OP2"
    bridge["m2_distance_km"] = bridge["actual_distance"]
    bridge["normalised_cpkm"] = bridge["op2_normalized_cpkm"]
    bridge["benchmark_gap"] = bridge["compare_cpkm"] - bridge["op2_cpkm"]

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
    for col in ["base_cpkm", "market_rate_impact", "tech_impact", "set_impact", "premium_impact", "supply_rates", "report_week"]:
        bridge[col] = None

    return bridge
