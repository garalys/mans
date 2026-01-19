"""
Market Rate Impact Calculator

Calculates the impact of market rate changes on transportation costs.
Uses Transporeon contract price evolution to measure market movements.
"""

import pandas as pd
import numpy as np


def calculate_market_rate_impact(
    base_data: pd.DataFrame,
    compare_data: pd.DataFrame,
    country: str = "",
    return_total: bool = False,
) -> float:
    """
    Calculate market rate impact considering price evolution.

    This function measures how changes in market rates (Transporeon prices)
    affect costs, isolating this from mix and volume changes.

    Input Table - base_data:
        | Column                         | Type    | Description                     |
        |--------------------------------|---------|---------------------------------|
        | orig_country                   | string  | Origin country code             |
        | dest_country                   | string  | Destination country code        |
        | distance_band                  | string  | Distance category               |
        | is_set                         | boolean | Whether using SET pricing       |
        | business                       | string  | Business unit (for grouping)    |
        | distance_for_cpkm              | float   | Distance in kilometers          |
        | total_cost_usd                 | float   | Total cost in USD               |
        | transporeon_contract_price_eur | float   | Market rate price in EUR        |

    Input Table - compare_data:
        | Column                         | Type    | Description                     |
        |--------------------------------|---------|---------------------------------|
        | orig_country                   | string  | Origin country code             |
        | dest_country                   | string  | Destination country code        |
        | distance_band                  | string  | Distance category               |
        | is_set                         | boolean | Whether using SET pricing       |
        | business                       | string  | Business unit (for grouping)    |
        | distance_for_cpkm              | float   | Distance in kilometers          |
        | transporeon_contract_price_eur | float   | Market rate price in EUR        |

    Args:
        base_data: DataFrame with base period data
        compare_data: DataFrame with comparison period data
        country: Country code or 'EU' for EU-level calculation
        return_total: If True, return total impact; if False, return per-km impact

    Output:
        float: Market rate impact
            - If return_total=True: Total market impact in USD
            - If return_total=False: Market impact per kilometer ($/km)

    Formula:
        1. price_evolution = (compare_transporeon_price / base_transporeon_price) - 1
        2. market_impact = compare_distance * base_rate * price_evolution
        3. Return sum / total_distance (if return_total=False)
    """
    if country == "EU":
        return _calculate_eu_market_rate_impact(base_data, compare_data, return_total)
    else:
        return calculate_country_market_rate_impact(base_data, compare_data, return_total)


def _calculate_eu_market_rate_impact(
    base_data: pd.DataFrame,
    compare_data: pd.DataFrame,
    return_total: bool = False,
) -> float:
    """
    Calculate EU-level market rate impact.

    For EU calculations, groups by origin country, destination country,
    distance band, and SET status.

    Internal function used by calculate_market_rate_impact.
    """
    # Check if aggregating across businesses (Total)
    if "Total" in base_data["business"].unique():
        # Aggregate all businesses
        base_grouped = base_data.groupby(
            ["orig_country", "dest_country", "distance_band", "is_set"]
        ).agg({
            "distance_for_cpkm": "sum",
            "total_cost_usd": "sum",
            "transporeon_contract_price_eur": "first",
        }).reset_index()

        compare_grouped = compare_data.groupby(
            ["orig_country", "dest_country", "distance_band", "is_set"]
        ).agg({
            "distance_for_cpkm": "sum",
            "transporeon_contract_price_eur": "first",
        }).reset_index()
    else:
        # Keep business dimension
        base_grouped = base_data.groupby(
            ["orig_country", "dest_country", "distance_band", "is_set", "business"]
        ).agg({
            "distance_for_cpkm": "sum",
            "total_cost_usd": "sum",
            "transporeon_contract_price_eur": "mean",
        }).reset_index()

        compare_grouped = compare_data.groupby(
            ["orig_country", "dest_country", "distance_band", "is_set", "business"]
        ).agg({
            "distance_for_cpkm": "sum",
            "transporeon_contract_price_eur": "mean",
        }).reset_index()

    # Calculate base rates
    base_grouped["base_rate"] = np.where(
        base_grouped["distance_for_cpkm"] > 0,
        base_grouped["total_cost_usd"] / base_grouped["distance_for_cpkm"],
        0,
    )

    # Set up merge columns
    merge_cols = ["orig_country", "dest_country", "distance_band", "is_set"]
    if "business" in base_grouped.columns:
        merge_cols.append("business")

    # Merge base rates with compare data
    merged = pd.merge(
        compare_grouped,
        base_grouped[merge_cols + ["base_rate", "transporeon_contract_price_eur"]],
        on=merge_cols,
        how="left",
    )

    # Calculate price evolution and impact
    merged["price_evolution"] = (
        merged["transporeon_contract_price_eur_x"]
        / merged["transporeon_contract_price_eur_y"]
    ) - 1

    merged["market_impact"] = (
        merged["distance_for_cpkm"]
        * merged["base_rate"]
        * merged["price_evolution"]
    )

    total_distance = merged["distance_for_cpkm"].sum()
    total_market_impact = merged["market_impact"].sum()

    if return_total:
        return total_market_impact
    return total_market_impact / total_distance if total_distance > 0 else 0


def calculate_country_market_rate_impact(
    base_data: pd.DataFrame,
    compare_data: pd.DataFrame,
    return_total: bool = False,
) -> float:
    """
    Calculate country-level market rate impact.

    Groups by destination country, distance band, and SET status.

    Input Table - base_data:
        | Column                         | Type    | Description                     |
        |--------------------------------|---------|---------------------------------|
        | dest_country                   | string  | Destination country code        |
        | distance_band                  | string  | Distance category               |
        | is_set                         | boolean | Whether using SET pricing       |
        | distance_for_cpkm              | float   | Distance in kilometers          |
        | total_cost_usd                 | float   | Total cost in USD               |
        | transporeon_contract_price_eur | float   | Market rate price in EUR        |

    Input Table - compare_data:
        | Column                         | Type    | Description                     |
        |--------------------------------|---------|---------------------------------|
        | dest_country                   | string  | Destination country code        |
        | distance_band                  | string  | Distance category               |
        | is_set                         | boolean | Whether using SET pricing       |
        | distance_for_cpkm              | float   | Distance in kilometers          |
        | transporeon_contract_price_eur | float   | Market rate price in EUR        |

    Args:
        base_data: DataFrame with base period data
        compare_data: DataFrame with comparison period data
        return_total: If True, return total impact; if False, return per-km impact

    Output:
        float: Market rate impact (total USD or $/km)
    """
    # Group by key dimensions
    base_grouped = base_data.groupby(["dest_country", "distance_band", "is_set"]).agg({
        "distance_for_cpkm": "sum",
        "total_cost_usd": "sum",
        "transporeon_contract_price_eur": "first",
    }).reset_index()

    compare_grouped = compare_data.groupby(["dest_country", "distance_band", "is_set"]).agg({
        "distance_for_cpkm": "sum",
        "transporeon_contract_price_eur": "first",
    }).reset_index()

    # Calculate base rates
    base_grouped["base_rate"] = np.where(
        base_grouped["distance_for_cpkm"] > 0,
        base_grouped["total_cost_usd"] / base_grouped["distance_for_cpkm"],
        0,
    )

    # Merge
    merged = pd.merge(
        compare_grouped,
        base_grouped[["dest_country", "distance_band", "is_set", "base_rate", "transporeon_contract_price_eur"]],
        on=["dest_country", "distance_band", "is_set"],
        how="left",
    )

    # Calculate price evolution and impact
    merged["price_evolution"] = (
        merged["transporeon_contract_price_eur_x"]
        / merged["transporeon_contract_price_eur_y"]
    ) - 1

    merged["market_impact"] = (
        merged["distance_for_cpkm"]
        * merged["base_rate"]
        * merged["price_evolution"]
    )

    total_distance = merged["distance_for_cpkm"].sum()
    total_market_impact = merged["market_impact"].sum()

    if return_total:
        return total_market_impact
    return total_market_impact / total_distance if total_distance > 0 else 0
