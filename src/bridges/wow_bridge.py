"""
Week-over-Week Bridge Calculator

Calculates bridge metrics comparing consecutive weeks within the same year.
"""

import pandas as pd

from ..utils.date_utils import extract_year_from_report_year
from ..calculators.carrier_calculator import calculate_active_carriers
from ..calculators.mix_calculator import compute_hierarchical_mix
from .bridge_metrics import calculate_detailed_bridge_metrics


def calculate_wow_bridge_metrics(
    df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    df_carrier: pd.DataFrame,
) -> None:
    """
    Calculate Week-over-Week bridge metrics with optimized batch processing.

    Updates bridge_df in-place with metrics comparing consecutive weeks
    within the same year.

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
        | + other columns for impact calculations                      |

    Input Table - bridge_df (to be updated):
        | Column         | Type   | Description                       |
        |----------------|--------|-----------------------------------|
        | bridge_type    | string | Must be 'WoW' for rows to update  |
        | bridging_value | string | Format: 'R2025_W01_to_W02'        |
        | report_year    | string | Year of comparison                |
        | orig_country   | string | Country code                      |
        | business       | string | Business unit                     |

    Input Table - df_carrier:
        | Column     | Type   | Description                    |
        |------------|--------|--------------------------------|
        | year       | string | Year in R20XX format           |
        | period     | string | Period (W01, etc.)             |
        | country    | string | Country code                   |
        | percentage | float  | Carrier scaling percentage     |

    Args:
        df: Main transportation data DataFrame
        bridge_df: Bridge structure DataFrame (modified in-place)
        df_carrier: Carrier scaling reference data

    Output:
        None - bridge_df is modified in-place with:
            - w1_* columns: Week 1 metrics
            - w2_* columns: Week 2 metrics
            - Bridge component columns (mix_impact, etc.)
    """
    years = sorted(df["report_year"].unique())

    # Pre-calculate all required data for efficiency
    wow_data = {}
    # Pre-compute full-data mix grains per (year, week)
    full_mix_cache = {}

    for year in years:
        year_data = df[df["report_year"] == year]
        wow_data[year] = year_data.groupby(
            ["report_week", "orig_country", "business"]
        ).apply(
            lambda x: {
                "data": x,
                "distance": x["distance_for_cpkm"].sum(),
                "costs": x["total_cost_usd"].sum(),
                "loads": x["executed_loads"].sum(),
                "carriers": calculate_active_carriers(
                    x, x["report_week"].iloc[0], x["orig_country"].iloc[0], x["business"].iloc[0]
                ),
            }
        ).to_dict()

        # Pre-compute full-data mix grains per week (all countries, all businesses)
        full_mix_cache[year] = {}
        for week in year_data["report_week"].unique():
            week_data = year_data[year_data["report_week"] == week]
            full_mix_cache[year][week] = compute_hierarchical_mix(week_data)

    # Process WoW rows in batches
    batch_size = 1000
    wow_rows = bridge_df[bridge_df["bridge_type"] == "WoW"]

    for start_idx in range(0, len(wow_rows), batch_size):
        batch = wow_rows.iloc[start_idx : start_idx + batch_size]

        for idx, row in batch.iterrows():
            year = row["report_year"]
            parts = row["bridging_value"].split("_")
            w1, w2 = parts[1], parts[3]

            key = (w1, row["orig_country"], row["business"])
            compare_key = (w2, row["orig_country"], row["business"])

            if key in wow_data[year] and compare_key in wow_data[year]:
                base_data = wow_data[year][key]["data"]
                compare_data = wow_data[year][compare_key]["data"]

                # Get pre-computed full-data mix grains
                full_base_mix = full_mix_cache[year].get(w1)
                full_compare_mix = full_mix_cache[year].get(w2)

                metrics = calculate_detailed_bridge_metrics(
                    base_data,
                    compare_data,
                    row["orig_country"],
                    extract_year_from_report_year(year),
                    extract_year_from_report_year(year),
                    df_carrier,
                    report_week=w2,
                    bridge_type="WoW",
                    full_base_mix=full_base_mix,
                    full_compare_mix=full_compare_mix,
                )

                # Update WoW-specific columns
                _update_wow_metrics(bridge_df, idx, metrics, year)


def _update_wow_metrics(
    bridge_df: pd.DataFrame,
    idx: int,
    metrics: dict,
    year: str,
) -> None:
    """
    Update WoW-specific metrics in bridge_df.

    Maps base metrics to w1_* columns and compare metrics to w2_* columns.
    """
    # Map base metrics to w1 columns
    metrics_mapping = {
        "base_distance_km": "w1_distance_km",
        "base_costs_usd": "w1_costs_usd",
        "base_loads": "w1_loads",
        "base_carriers": "w1_carriers",
        "base_cpkm": "w1_cpkm",
    }

    for old_key, new_key in metrics_mapping.items():
        if old_key in metrics:
            bridge_df.loc[idx, new_key] = metrics[old_key]

    # Update w2 (comparison) metrics
    bridge_df.loc[idx, "w2_distance_km"] = metrics.get("compare_distance_km")
    bridge_df.loc[idx, "w2_costs_usd"] = metrics.get("compare_costs_usd")
    bridge_df.loc[idx, "w2_loads"] = metrics.get("compare_loads")
    bridge_df.loc[idx, "w2_carriers"] = metrics.get("compare_carriers")
    bridge_df.loc[idx, "w2_cpkm"] = metrics.get("compare_cpkm")

    # Update bridge components
    bridge_components = [
        "base_cpkm",
        "mix_impact",
        "normalised_cpkm",
        "country_mix",
        "corridor_mix",
        "distance_band_mix",
        "business_flow_mix",
        "equipment_type_mix",
        "supply_rates",
        "carrier_and_demand_impact",
        "carrier_impact",
        "demand_impact",
        "premium_impact",
        "market_rate_impact",
        "tech_impact",
        "compare_cpkm",
    ]

    for component in bridge_components:
        if component in metrics:
            bridge_df.loc[idx, component] = metrics[component]
