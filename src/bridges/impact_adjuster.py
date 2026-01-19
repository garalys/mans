"""
Impact Adjuster

Adjusts carrier and demand impacts to reconcile with actual CPKM changes
and calculates impacts in millions USD.
"""

import pandas as pd
import numpy as np

from ..utils.date_utils import extract_year_from_report_year


def adjust_carrier_demand_impacts(bridge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust carrier and demand impacts and calculate impacts in millions.

    This function reconciles the sum of all impact components with the actual
    CPKM change by proportionally adjusting carrier and demand impacts.

    Input Table - bridge_df:
        | Column                    | Type   | Description                           |
        |---------------------------|--------|---------------------------------------|
        | normalised_cpkm           | float  | Normalized cost per km                |
        | carrier_impact            | float  | Carrier availability impact ($/km)    |
        | demand_impact             | float  | Demand change impact ($/km)           |
        | premium_impact            | float  | Premium pricing impact ($/km)         |
        | market_rate_impact        | float  | Market rate impact ($/km)             |
        | set_impact                | float  | SET pricing impact ($/km)             |
        | tech_impact               | float  | Technology impact ($/km)              |
        | compare_cpkm              | float  | Comparison period CPKM                |
        | bridge_type               | string | 'YoY', 'WoW', or 'MTD'                |
        | bridging_value            | string | Period comparison identifier          |
        | w2_distance_km            | float  | WoW week 2 distance (optional)        |
        | m2_distance_km            | float  | MTD month 2 distance (optional)       |
        | compare_distance_km       | float  | Compare period distance (optional)    |

    Output Table - bridge_df (modified):
        Same as input with additional/modified columns:
            | Column                    | Type   | Description                     |
            |---------------------------|--------|---------------------------------|
            | carrier_impact (adjusted) | float  | Rebalanced carrier impact       |
            | demand_impact (adjusted)  | float  | Rebalanced demand impact        |
            | carrier_and_demand_impact | float  | Combined adjusted impacts       |
            | *_impact_mm               | float  | Each impact in millions USD     |

    Args:
        bridge_df: Bridge DataFrame with calculated metrics

    Returns:
        pd.DataFrame: Modified bridge_df with adjusted impacts

    Formula:
        1. expected_cpkm = normalised_cpkm + sum(all impacts)
        2. discrepancy = compare_cpkm - expected_cpkm
        3. Adjust carrier/demand proportionally:
           - carrier_weight = |carrier_impact| / (|carrier_impact| + |demand_impact|)
           - new_carrier = carrier_impact + discrepancy * carrier_weight
           - Similar for demand
        4. impact_mm = impact * distance / 1,000,000
    """
    # Calculate expected CPKM (sum of all components)
    bridge_df["expected_cpkm"] = bridge_df.apply(
        lambda row: (
            row["normalised_cpkm"]
            + (row["carrier_impact"] if pd.notnull(row["carrier_impact"]) else 0)
            + (row["demand_impact"] if pd.notnull(row["demand_impact"]) else 0)
            + (row["premium_impact"] if pd.notnull(row["premium_impact"]) else 0)
            + (row["market_rate_impact"] if pd.notnull(row["market_rate_impact"]) else 0)
            + (row["set_impact"] if pd.notnull(row["set_impact"]) else 0)
            + (row["tech_impact"] if pd.notnull(row["tech_impact"]) else 0)
        )
        if pd.notnull(row["normalised_cpkm"])
        else None,
        axis=1,
    )

    # Calculate discrepancy between expected and actual
    bridge_df["discrepancy"] = bridge_df.apply(
        lambda row: (row["compare_cpkm"] - row["expected_cpkm"])
        if pd.notnull(row["expected_cpkm"]) and pd.notnull(row["compare_cpkm"])
        else None,
        axis=1,
    )

    # Adjust impacts and calculate combined carrier_and_demand_impact
    for idx, row in bridge_df.iterrows():
        if (
            pd.notnull(row["discrepancy"])
            and pd.notnull(row["carrier_impact"])
            and pd.notnull(row["demand_impact"])
        ):
            total_impact = abs(row["carrier_impact"]) + abs(row["demand_impact"])
            if total_impact > 0:
                carrier_weight = abs(row["carrier_impact"]) / total_impact
                demand_weight = abs(row["demand_impact"]) / total_impact

                # Proportionally distribute discrepancy
                new_carrier = row["carrier_impact"] + (row["discrepancy"] * carrier_weight)
                new_demand = row["demand_impact"] + (row["discrepancy"] * demand_weight)

                bridge_df.at[idx, "carrier_impact"] = new_carrier
                bridge_df.at[idx, "demand_impact"] = new_demand
                bridge_df.at[idx, "carrier_and_demand_impact"] = new_carrier + new_demand

        # Get appropriate distance column for _mm calculations
        distance = _get_distance_for_row(row)

        # Calculate impacts in millions
        if pd.notnull(distance):
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
                if pd.notnull(bridge_df.at[idx, impact]):
                    bridge_df.at[idx, f"{impact}_mm"] = (
                        bridge_df.at[idx, impact] * distance
                    ) / 1_000_000

    # Drop temporary columns
    bridge_df = bridge_df.drop(["expected_cpkm", "discrepancy"], axis=1)

    return bridge_df


def _get_distance_for_row(row: pd.Series) -> float:
    """
    Get the appropriate distance value for calculating _mm impacts.

    Logic:
        - WoW: Use w2_distance_km
        - MTD: Use m2_distance_km
        - YoY: Extract compare year and use y{year}_distance_km
        - Fallback: compare_distance_km
    """
    distance = None

    # For WoW, use w2_distance_km
    if pd.notnull(row.get("w2_distance_km")):
        distance = row["w2_distance_km"]
    # For MTD, use m2_distance_km
    elif pd.notnull(row.get("m2_distance_km")):
        distance = row["m2_distance_km"]
    # For YoY, extract compare year from bridging_value
    elif row.get("bridge_type") == "YoY" and pd.notnull(row.get("bridging_value")):
        try:
            compare_year_str = row["bridging_value"].split("_to_")[1]
            compare_year_num = extract_year_from_report_year(compare_year_str)
            distance_col = f"y{compare_year_num}_distance_km"
            if distance_col in row.index and pd.notnull(row[distance_col]):
                distance = row[distance_col]
        except (IndexError, ValueError):
            # Fallback if parsing fails
            if pd.notnull(row.get("compare_distance_km")):
                distance = row["compare_distance_km"]
    else:
        # Fallback for any other cases
        if pd.notnull(row.get("compare_distance_km")):
            distance = row["compare_distance_km"]

    return distance
