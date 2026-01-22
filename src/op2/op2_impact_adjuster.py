"""
OP2 Impact Adjuster

Adjusts carrier and demand impacts for OP2 bridges to reconcile with actual CPKM.
"""

import pandas as pd

from ..config.logging_config import logger


def adjust_op2_carrier_demand_impacts(bridge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust carrier and demand impacts for OP2 weekly bridges.

    Uses the same reconciliation logic as other bridge types to ensure
    the sum of impacts equals the actual CPKM change.

    Input Table - bridge_df:
        | Column                    | Type   | Description                     |
        |---------------------------|--------|---------------------------------|
        | bridge_type               | string | Filter for 'op2_weekly'         |
        | normalised_cpkm           | float  | OP2 normalized CPKM             |
        | carrier_impact            | float  | Carrier impact ($/km)           |
        | demand_impact             | float  | Demand impact ($/km)            |
        | premium_impact            | float  | Premium impact ($/km)           |
        | market_rate_impact        | float  | Market rate impact ($/km)       |
        | set_impact                | float  | SET impact ($/km)               |
        | tech_impact               | float  | Tech impact ($/km)              |
        | compare_cpkm              | float  | Actual CPKM                     |
        | actual_distance           | float  | Actual distance (km)            |

    Output Table - bridge_df (modified):
        Same as input with:
        - carrier_impact, demand_impact adjusted to close discrepancy
        - carrier_and_demand_impact recalculated
        - *_impact_mm columns added (impacts in millions USD)

    Formula:
        1. expected_cpkm = normalised_cpkm + sum(all impacts)
        2. discrepancy = compare_cpkm - expected_cpkm
        3. Proportionally adjust carrier and demand impacts
        4. impact_mm = impact * actual_distance / 1,000,000
    """
    logger.info("Adjusting OP2 carrier & demand impacts (reconciling)...")

    mask_op2 = bridge_df["bridge_type"].isin(["op2_weekly", "op2_monthly", "op2_quarterly"])

    # Calculate expected CPKM
    bridge_df.loc[mask_op2, "expected_cpkm"] = bridge_df.loc[mask_op2].apply(
        lambda row: (
            row["normalised_cpkm"]
            + (row["carrier_impact"] or 0)
            + (row["demand_impact"] or 0)
            + (row["premium_impact"] or 0)
            + (row["market_rate_impact"] or 0)
            + (row["set_impact"] or 0)
            + (row["tech_impact"] or 0)
        ),
        axis=1,
    )

    # Calculate discrepancy
    bridge_df.loc[mask_op2, "discrepancy"] = (
        bridge_df.loc[mask_op2, "compare_cpkm"]
        - bridge_df.loc[mask_op2, "expected_cpkm"]
    )

    # Rebalance carrier and demand impacts
    for idx, row in bridge_df.loc[mask_op2].iterrows():
        if (
            pd.notnull(row["discrepancy"])
            and pd.notnull(row["carrier_impact"])
            and pd.notnull(row["demand_impact"])
        ):
            total = abs(row["carrier_impact"]) + abs(row["demand_impact"])

            if total > 0:
                carrier_weight = abs(row["carrier_impact"]) / total
                demand_weight = abs(row["demand_impact"]) / total

                new_carrier = row["carrier_impact"] + row["discrepancy"] * carrier_weight
                new_demand = row["demand_impact"] + row["discrepancy"] * demand_weight

                bridge_df.at[idx, "carrier_impact"] = new_carrier
                bridge_df.at[idx, "demand_impact"] = new_demand
                bridge_df.at[idx, "carrier_and_demand_impact"] = new_carrier + new_demand

    # Calculate impacts in millions
    distance = bridge_df.loc[mask_op2, "actual_distance_km"]

    impacts = [
        "mix_impact",
        "carrier_impact",
        "demand_impact",
        "carrier_and_demand_impact",
        "premium_impact",
        "market_rate_impact",
        "set_impact",
        "tech_impact",
    ]

    for impact in impacts:
        bridge_df.loc[mask_op2, f"{impact}_mm"] = (
            bridge_df.loc[mask_op2, impact].fillna(0)
            * distance
            / 1_000_000
        )

    # Cleanup temporary columns
    bridge_df.drop(
        columns=["expected_cpkm", "discrepancy"],
        inplace=True,
        errors="ignore",
    )

    logger.info(f"OP2 rows rebalanced: {mask_op2.sum()}")

    return bridge_df
