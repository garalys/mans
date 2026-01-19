"""
SET Impact Calculator

Calculates the impact of SET (Special Equipment Type) pricing on transportation costs.
SET loads have contracted pricing that may differ from spot market rates.
"""

import pandas as pd
import numpy as np


def calculate_set_impact(base_data: pd.DataFrame, compare_data: pd.DataFrame) -> float:
    """
    Calculate SET pricing impact on costs.

    This function calculates how SET pricing changes between periods affect
    total costs. Only considers loads where is_set=True.

    Input Table - base_data:
        | Column           | Type    | Description                    |
        |------------------|---------|--------------------------------|
        | dest_country     | string  | Destination country code       |
        | distance_band    | string  | Distance category              |
        | is_set           | boolean | Must be True for SET loads     |
        | distance_for_cpkm| float   | Distance in kilometers         |
        | total_cost_usd   | float   | Total cost in USD              |

    Input Table - compare_data:
        | Column           | Type    | Description                    |
        |------------------|---------|--------------------------------|
        | dest_country     | string  | Destination country code       |
        | distance_band    | string  | Distance category              |
        | is_set           | boolean | Must be True for SET loads     |
        | distance_for_cpkm| float   | Distance in kilometers         |
        | total_cost_usd   | float   | Total cost in USD              |

    Output:
        float: Total SET impact in USD
            Positive = compare period SET rates higher than base
            Negative = compare period SET rates lower than base

    Formula:
        For each (dest_country, distance_band) combination:
        1. base_cpkm = base_cost / base_distance
        2. compare_cpkm = compare_cost / compare_distance
        3. set_impact = compare_distance * (compare_cpkm - base_cpkm)
        4. Return sum of all set_impacts

    Example:
        If base SET rate was $0.40/km and compare is $0.45/km for 1000km,
        SET impact = 1000 * (0.45 - 0.40) = $50
    """
    # Filter for SET loads only (is_set=True)
    base_set = base_data[base_data["is_set"] == True].copy()
    compare_set = compare_data[compare_data["is_set"] == True].copy()

    # Group by destination and distance band
    base_grouped = base_set.groupby(["dest_country", "distance_band", "is_set"]).agg({
        "distance_for_cpkm": "sum",
        "total_cost_usd": "sum",
    }).reset_index()

    compare_grouped = compare_set.groupby(["dest_country", "distance_band", "is_set"]).agg({
        "distance_for_cpkm": "sum",
        "total_cost_usd": "sum",
    }).reset_index()

    # Calculate cost per km for both periods
    base_grouped["cpkm_base"] = np.where(
        base_grouped["distance_for_cpkm"] > 0,
        base_grouped["total_cost_usd"] / base_grouped["distance_for_cpkm"],
        0,
    )

    compare_grouped["cpkm_compare"] = np.where(
        compare_grouped["distance_for_cpkm"] > 0,
        compare_grouped["total_cost_usd"] / compare_grouped["distance_for_cpkm"],
        0,
    )

    # Merge base rates with compare data
    merged = pd.merge(
        compare_grouped,
        base_grouped[["dest_country", "distance_band", "is_set", "cpkm_base"]],
        on=["dest_country", "distance_band", "is_set"],
        how="left",
    )

    # Calculate SET impact: (compare rate - base rate) * compare distance
    merged["set_impact"] = merged["distance_for_cpkm"] * (
        merged["cpkm_compare"] - merged["cpkm_base"]
    )

    return merged["set_impact"].sum()
