"""
Normalized Cost Calculator

Calculates normalized costs by applying base period rates to comparison period volumes.
This isolates the mix impact from rate changes.
"""

import pandas as pd
import numpy as np


def calculate_normalized_cost(base_data: pd.DataFrame, compare_data: pd.DataFrame) -> float:
    """
    Calculate normalized cost using base year rates on comparison year volume.

    This function applies base period cost-per-kilometer rates to comparison period
    distances, grouped by destination country, distance band, and SET status.
    The result represents what the comparison period cost would have been if
    rates had remained constant.

    Input Table - base_data:
        | Column           | Type    | Description                    |
        |------------------|---------|--------------------------------|
        | dest_country     | string  | Destination country code       |
        | distance_band    | string  | Distance category              |
        | is_set           | boolean | Whether using SET pricing      |
        | distance_for_cpkm| float   | Distance in kilometers         |
        | total_cost_usd   | float   | Total cost in USD              |

    Input Table - compare_data:
        | Column           | Type    | Description                    |
        |------------------|---------|--------------------------------|
        | dest_country     | string  | Destination country code       |
        | distance_band    | string  | Distance category              |
        | is_set           | boolean | Whether using SET pricing      |
        | distance_for_cpkm| float   | Distance in kilometers         |

    Output:
        float: Total normalized cost in USD
            Calculated as: sum(compare_distance * base_rate) for each group

    Formula:
        For each (dest_country, distance_band, is_set) combination:
        1. base_rate = base_total_cost / base_distance
        2. normalized_cost = compare_distance * base_rate
        3. Return sum of all normalized_costs

    Example:
        If base period had rate of $0.50/km and compare period has 1000km,
        normalized cost for that group = 1000 * 0.50 = $500
    """
    # Group base data by key dimensions
    base_grouped = base_data.groupby(["dest_country", "distance_band", "is_set"]).agg({
        "distance_for_cpkm": "sum",
        "total_cost_usd": "sum",
    }).reset_index()

    # Group compare data by key dimensions
    compare_grouped = compare_data.groupby(["dest_country", "distance_band", "is_set"]).agg({
        "distance_for_cpkm": "sum",
    }).reset_index()

    # Calculate base rates (cost per km)
    base_grouped["rate"] = np.where(
        base_grouped["distance_for_cpkm"] > 0,
        base_grouped["total_cost_usd"] / base_grouped["distance_for_cpkm"],
        0,
    )

    # Merge base rates with compare volumes
    merged = pd.merge(
        compare_grouped,
        base_grouped[["dest_country", "distance_band", "is_set", "rate"]],
        on=["dest_country", "distance_band", "is_set"],
        how="left",
    )

    # Calculate normalized cost (compare volume * base rate)
    merged["normalized_cost"] = merged["distance_for_cpkm"] * merged["rate"]

    return merged["normalized_cost"].sum()
