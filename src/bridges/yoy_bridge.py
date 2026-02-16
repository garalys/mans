"""
Year-over-Year Bridge Calculator

Calculates bridge metrics comparing same weeks/months between consecutive years.
"""

import pandas as pd

from ..utils.date_utils import extract_year_from_report_year
from ..calculators.mix_calculator import compute_hierarchical_mix
from .bridge_metrics import calculate_detailed_bridge_metrics


def calculate_yoy_bridge_metrics(
    df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    base_year: str,
    compare_year: str,
    df_carrier: pd.DataFrame,
) -> None:
    """
    Calculate Year-over-Year bridge metrics for specified year pair.

    Updates bridge_df in-place with calculated metrics for all YoY rows
    matching the given base and compare years.

    Input Table - df (main data):
        | Column           | Type    | Description                    |
        |------------------|---------|--------------------------------|
        | report_year      | string  | Year in R20XX format           |
        | report_week      | string  | Week in WXX format             |
        | orig_country     | string  | Origin country code            |
        | business         | string  | Business unit                  |
        | distance_for_cpkm| float   | Distance in kilometers         |
        | total_cost_usd   | float   | Total cost in USD              |
        | executed_loads   | int     | Number of loads                |
        | vehicle_carrier  | string  | Carrier identifier             |
        | + other columns for impact calculations                      |

    Input Table - bridge_df (to be updated):
        | Column         | Type   | Description                       |
        |----------------|--------|-----------------------------------|
        | bridge_type    | string | Must be 'YoY' for rows to update  |
        | bridging_value | string | Format: 'R2024_to_R2025'          |
        | report_week    | string | Week being compared               |
        | orig_country   | string | Country code                      |
        | business       | string | Business unit                     |

    Input Table - df_carrier:
        | Column     | Type   | Description                    |
        |------------|--------|--------------------------------|
        | year       | string | Year in R20XX format           |
        | period     | string | Period (W01, M01, etc.)        |
        | country    | string | Country code                   |
        | percentage | float  | Carrier scaling percentage     |

    Args:
        df: Main transportation data DataFrame
        bridge_df: Bridge structure DataFrame (modified in-place)
        base_year: Base year in R20XX format (e.g., 'R2024')
        compare_year: Compare year in R20XX format (e.g., 'R2025')
        df_carrier: Carrier scaling reference data

    Output:
        None - bridge_df is modified in-place with these columns filled:
            - base_cpkm, compare_cpkm
            - mix_impact, normalised_cpkm
            - carrier_impact, demand_impact
            - market_rate_impact, set_impact, tech_impact
            - base_distance_km, base_costs_usd, base_loads, base_carriers
            - compare_distance_km, compare_costs_usd, compare_loads, compare_carriers
    """
    base_year_num = extract_year_from_report_year(base_year)
    compare_year_num = extract_year_from_report_year(compare_year)

    bridging_value = f"{base_year}_to_{compare_year}"

    # Pre-compute full-data mix grains per week (all countries, all businesses)
    full_mix_cache = {}

    # Process each matching row
    for idx, row in bridge_df[
        (bridge_df["bridge_type"] == "YoY")
        & (bridge_df["bridging_value"] == bridging_value)
    ].iterrows():
        week = row["report_week"]
        country = row["orig_country"]
        business = row["business"]

        # Pre-compute full-data mix grains (once per week)
        if week not in full_mix_cache:
            full_base = df[
                (df["report_year"] == base_year) & (df["report_week"] == week)
            ]
            full_compare = df[
                (df["report_year"] == compare_year) & (df["report_week"] == week)
            ]
            full_mix_cache[week] = {
                "base_mix": compute_hierarchical_mix(full_base) if not full_base.empty else None,
                "compare_mix": compute_hierarchical_mix(full_compare) if not full_compare.empty else None,
            }

        # Filter data for base and compare periods (per country, per business)
        base_data = df[
            (df["report_year"] == base_year)
            & (df["report_week"] == week)
            & (df["orig_country"] == country)
            & (df["business"] == business)
        ]

        compare_data = df[
            (df["report_year"] == compare_year)
            & (df["report_week"] == week)
            & (df["orig_country"] == country)
            & (df["business"] == business)
        ]

        # Calculate metrics — pass pre-computed full-data mix grains for percentages
        metrics = calculate_detailed_bridge_metrics(
            base_data,
            compare_data,
            country,
            base_year_num,
            compare_year_num,
            df_carrier,
            report_week=week,
            full_base_mix=full_mix_cache[week]["base_mix"],
            full_compare_mix=full_mix_cache[week]["compare_mix"],
        )

        # Update bridge_df with calculated metrics
        for metric, value in metrics.items():
            bridge_df.loc[idx, metric] = value
