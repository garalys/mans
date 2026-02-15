"""
Total Business Aggregation Calculator

Calculates bridge metrics aggregated across all businesses (Total).
"""

import re
import pandas as pd
import numpy as np

from ..utils.date_utils import extract_year_from_report_year, get_mtd_date_range
from ..config.settings import (
    COUNTRY_COEFFICIENTS_VOLUME,
    COUNTRY_COEFFICIENTS_LCR,
    get_tech_savings_rate,
)
from ..config.logging_config import logger
from ..calculators.carrier_calculator import calculate_active_carriers
from ..calculators.normalized_cost_calculator import calculate_normalized_cost
from ..calculators.market_rate_calculator import calculate_market_rate_impact
from ..calculators.mix_calculator import (
    compute_hierarchical_mix,
    compute_normalised_distance,
    compute_seven_metrics,
    compute_mix_impacts,
    compute_cell_cpkm,
)
# from ..calculators.set_impact_calculator import calculate_set_impact  # Commented out - replaced by equipment_type_mix


def calculate_total_aggregated_metrics(
    df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    df_carrier: pd.DataFrame,
) -> None:
    """
    Calculate bridge metrics for 'Total' business (aggregated across all businesses).

    Updates the total bridge DataFrame with metrics aggregated across all
    business units for each country and time period.

    Input Table - df (main data):
        | Column           | Type    | Description                    |
        |------------------|---------|--------------------------------|
        | report_year      | string  | Year in R20XX format           |
        | report_week      | string  | Week in WXX format             |
        | report_month     | string  | Month in MXX format            |
        | report_day       | datetime| Date of the report             |
        | orig_country     | string  | Origin country code            |
        | business         | string  | Business unit                  |
        | distance_for_cpkm| float   | Distance in kilometers         |
        | total_cost_usd   | float   | Total cost in USD              |
        | executed_loads   | int     | Number of loads                |
        | + other columns for calculations                             |

    Input Table - bridge_df (Total bridge to update):
        | Column         | Type   | Description                       |
        |----------------|--------|-----------------------------------|
        | bridge_type    | string | 'YoY', 'WoW', or 'MTD'            |
        | business       | string | Always 'Total'                    |
        | orig_country   | string | Country code or 'EU'              |

    Input Table - df_carrier:
        | Column     | Type   | Description                    |
        |------------|--------|--------------------------------|
        | year       | string | Year in R20XX format           |
        | period     | string | Period identifier              |
        | country    | string | Country code                   |
        | percentage | float  | Carrier scaling percentage     |

    Args:
        df: Main transportation data DataFrame
        bridge_df: Total bridge structure DataFrame (modified in-place)
        df_carrier: Carrier scaling reference data

    Output:
        None - bridge_df is modified in-place with aggregated Total metrics.
    """
    # Pre-aggregate data by period type
    aggs = {period_type: {} for period_type in ["YoY", "WoW", "MTD"]}

    # Process in batches
    batch_size = 1000
    for period_type in ["YoY", "WoW", "MTD"]:
        rows = bridge_df[bridge_df["bridge_type"] == period_type]

        for start_idx in range(0, len(rows), batch_size):
            batch = rows.iloc[start_idx : start_idx + batch_size]

            for idx, row in batch.iterrows():
                country = row["orig_country"]

                # Get data based on period type
                if period_type == "YoY":
                    base_year, compare_year = row["bridging_value"].split("_to_")
                    time_period = row["report_week"]
                    key = (base_year, compare_year, time_period)

                    if key not in aggs[period_type]:
                        aggs[period_type][key] = {
                            "base": df[
                                (df["report_year"] == base_year)
                                & (df["report_week"] == time_period)
                            ],
                            "compare": df[
                                (df["report_year"] == compare_year)
                                & (df["report_week"] == time_period)
                            ],
                        }
                    base_data = aggs[period_type][key]["base"]
                    compare_data = aggs[period_type][key]["compare"]
                    compare_year_num = extract_year_from_report_year(compare_year)

                elif period_type == "WoW":
                    year = row["report_year"]
                    parts = row["bridging_value"].split("_")
                    w1, w2 = parts[1], parts[3]
                    time_period = w2
                    key = (year, w1, w2)

                    if key not in aggs[period_type]:
                        aggs[period_type][key] = {
                            "base": df[
                                (df["report_year"] == year) & (df["report_week"] == w1)
                            ],
                            "compare": df[
                                (df["report_year"] == year) & (df["report_week"] == w2)
                            ],
                        }
                    base_data = aggs[period_type][key]["base"]
                    compare_data = aggs[period_type][key]["compare"]
                    compare_year_num = extract_year_from_report_year(year)

                else:  # MTD
                    match = re.match(r"(.+)_(.+)_to_(.+)_.+", row["bridging_value"])
                    base_year, month, compare_year = match.groups()
                    start_date, end_date = get_mtd_date_range(
                        df, compare_year, int(month.replace("M", ""))
                    )

                    if start_date is None:
                        continue

                    key = (base_year, compare_year, month)
                    if key not in aggs[period_type]:
                        aggs[period_type][key] = {
                            "base": df[
                                (df["report_year"] == base_year)
                                & (df["report_month"] == month)
                                & (df["report_day"] >= start_date.replace(year=extract_year_from_report_year(base_year)))
                                & (df["report_day"] <= end_date.replace(year=extract_year_from_report_year(base_year)))
                            ],
                            "compare": df[
                                (df["report_year"] == compare_year)
                                & (df["report_month"] == month)
                                & (df["report_day"] >= start_date)
                                & (df["report_day"] <= end_date)
                            ],
                        }
                    base_data = aggs[period_type][key]["base"]
                    compare_data = aggs[period_type][key]["compare"]
                    time_period = month
                    compare_year_num = extract_year_from_report_year(compare_year)

                # Apply country filter if not EU
                if country != "EU":
                    base_data = base_data[base_data["orig_country"] == country]
                    compare_data = compare_data[compare_data["orig_country"] == country]

                # Skip if empty
                if base_data.empty or compare_data.empty:
                    continue

                # Calculate total metrics
                metrics = _calculate_total_metrics(base_data, compare_data, country, time_period)

                # Update bridge DataFrame
                _update_bridge_with_total_metrics(
                    bridge_df,
                    metrics,
                    idx,
                    base_data,
                    compare_data,
                    period_type,
                    time_period,
                    compare_year_num,
                    df_carrier,
                )


def _calculate_total_metrics(
    base_data: pd.DataFrame,
    compare_data: pd.DataFrame,
    country: str,
    time_period: str,
) -> dict:
    """Calculate total metrics aggregated across all businesses."""
    # Pre-aggregate common metrics
    aggs = {
        "base": base_data.groupby("business").agg({
            "distance_for_cpkm": "sum",
            "total_cost_usd": "sum",
            "executed_loads": "sum",
        }),
        "compare": compare_data.groupby("business").agg({
            "distance_for_cpkm": "sum",
            "total_cost_usd": "sum",
            "executed_loads": "sum",
        }),
    }

    total_metrics = {
        "base_distance_km": aggs["base"]["distance_for_cpkm"].sum(),
        "base_costs_usd": aggs["base"]["total_cost_usd"].sum(),
        "base_loads": aggs["base"]["executed_loads"].sum(),
        "distance_km": aggs["compare"]["distance_for_cpkm"].sum(),
        "costs_usd": aggs["compare"]["total_cost_usd"].sum(),
        "loads": aggs["compare"]["executed_loads"].sum(),
        "market_rate_impact": 0,
    }

    # Calculate carriers
    total_metrics["base_carriers"] = calculate_active_carriers(base_data, time_period, country)
    total_metrics["carriers"] = calculate_active_carriers(compare_data, time_period, country)

    # Calculate market rate impact
    if country == "EU":
        total_market_impact = 0
        total_compare_distance = 0

        for orig_country in ["DE", "ES", "FR", "IT", "UK"]:
            base_country = base_data[base_data["orig_country"] == orig_country]
            compare_country = compare_data[compare_data["orig_country"] == orig_country]

            if not base_country.empty and not compare_country.empty:
                market_rate = calculate_market_rate_impact(base_country, compare_country, orig_country)
                logger.info(f"{orig_country}: market rate: {market_rate}")
                compare_distance = compare_country["distance_for_cpkm"].sum()
                total_market_impact += market_rate * compare_distance
                total_compare_distance += compare_distance

        if total_compare_distance > 0:
            total_metrics["market_rate_impact"] = total_market_impact / total_compare_distance
    else:
        if total_metrics["distance_km"] > 0:
            market_rate = calculate_market_rate_impact(base_data, compare_data, country)
            total_metrics["market_rate_impact"] = market_rate

    # Calculate CPKMs
    if total_metrics["base_distance_km"] > 0:
        total_metrics["base_cpkm"] = total_metrics["base_costs_usd"] / total_metrics["base_distance_km"]
    else:
        total_metrics["base_cpkm"] = None

    if total_metrics["distance_km"] > 0:
        total_metrics["cpkm"] = total_metrics["costs_usd"] / total_metrics["distance_km"]
    else:
        total_metrics["cpkm"] = None

    return total_metrics


def _update_bridge_with_total_metrics(
    bridge_df: pd.DataFrame,
    metrics: dict,
    idx: int,
    base_data: pd.DataFrame,
    compare_data: pd.DataFrame,
    period_type: str,
    time_period: str,
    compare_year: int,
    df_carrier: pd.DataFrame,
) -> None:
    """Update bridge DataFrame with total aggregated metrics."""
    # Determine prefix mapping
    if period_type == "YoY":
        prefix_map = {"base": "base_", "compare": "compare_"}
        if time_period is None:
            time_period = bridge_df.loc[idx, "report_week"]
    elif period_type == "WoW":
        year = base_data["report_year"].iloc[0]
        prefix_map = {"base": "w1_", "compare": "w2_"}
        if time_period is None:
            time_period = bridge_df.loc[idx, "bridging_value"].split("_")[3]
        compare_year = extract_year_from_report_year(year)
    else:  # MTD
        prefix_map = {"base": "m1_", "compare": "m2_"}

    # Update metrics
    update_dict = {
        "base_cpkm": metrics["base_cpkm"],
        f"{prefix_map['base']}distance_km": metrics["base_distance_km"],
        f"{prefix_map['base']}costs_usd": metrics["base_costs_usd"],
        f"{prefix_map['base']}loads": metrics["base_loads"],
        f"{prefix_map['base']}carriers": metrics["base_carriers"],
        f"{prefix_map['base']}cpkm": metrics["base_cpkm"],
        "compare_cpkm": metrics["cpkm"],
        f"{prefix_map['compare']}distance_km": metrics["distance_km"],
        f"{prefix_map['compare']}costs_usd": metrics["costs_usd"],
        f"{prefix_map['compare']}loads": metrics["loads"],
        f"{prefix_map['compare']}carriers": metrics["carriers"],
        f"{prefix_map['compare']}cpkm": metrics["cpkm"],
        "market_rate_impact": metrics["market_rate_impact"],
    }

    bridge_df.loc[idx, list(update_dict.keys())] = list(update_dict.values())

    # Calculate bridge components if valid
    if metrics["base_cpkm"] is not None and metrics["cpkm"] is not None:
        country = bridge_df.loc[idx, "orig_country"]

        # Hierarchical Mix Impact (total level = across all businesses)
        base_filtered = base_data if country == "EU" else base_data[base_data["orig_country"] == country]
        compare_filtered = compare_data if country == "EU" else compare_data[compare_data["orig_country"] == country]

        base_mix = compute_hierarchical_mix(base_filtered)
        compare_mix = compute_hierarchical_mix(compare_filtered)
        norm_dist = compute_normalised_distance(base_mix, compare_mix)
        base_cpkm_cells = compute_cell_cpkm(base_filtered)
        compare_cpkm_cells = compute_cell_cpkm(compare_filtered)

        seven = compute_seven_metrics(
            base_mix, compare_mix, norm_dist, base_cpkm_cells, compare_cpkm_cells
        )

        agg_level = "country" if country != "EU" else "eu"
        mix_results = compute_mix_impacts(
            seven, metrics["distance_km"],
            bridge_level="total",
            aggregation_level=agg_level,
        )

        bridge_df.loc[idx, "mix_impact"] = mix_results["mix_impact"]
        bridge_df.loc[idx, "normalised_cpkm"] = mix_results["normalised_cpkm"]
        bridge_df.loc[idx, "country_mix"] = mix_results["country_mix"]
        bridge_df.loc[idx, "corridor_mix"] = mix_results["corridor_mix"]
        bridge_df.loc[idx, "distance_band_mix"] = mix_results["distance_band_mix"]
        bridge_df.loc[idx, "business_flow_mix"] = mix_results["business_flow_mix"]
        bridge_df.loc[idx, "equipment_type_mix"] = mix_results["equipment_type_mix"]

        # Carrier and demand impacts
        if metrics["carriers"] > 0 and metrics["base_carriers"] > 0:
            period = time_period if time_period and time_period.startswith(("W", "M")) else None

            percentage = 0.0
            if period is not None:
                year_str = f"R{compare_year}"
                lookup_mask = (
                    (df_carrier["year"] == year_str)
                    & (df_carrier["period"] == period)
                    & (df_carrier["country"] == country)
                )
                matching_rows = df_carrier[lookup_mask]
                if not matching_rows.empty:
                    percentage = matching_rows["percentage"].iloc[0]

            slope = COUNTRY_COEFFICIENTS_VOLUME.get(f"{country}_SLOPE", COUNTRY_COEFFICIENTS_VOLUME["EU_SLOPE"])
            intercept = COUNTRY_COEFFICIENTS_VOLUME.get(f"{country}_INTERCEPT", COUNTRY_COEFFICIENTS_VOLUME["EU_INTERCEPT"])
            slope_lcr = COUNTRY_COEFFICIENTS_LCR.get(f"{country}_SLOPE_LCR", COUNTRY_COEFFICIENTS_LCR["EU_SLOPE_LCR"])
            poly_lcr = COUNTRY_COEFFICIENTS_LCR.get(f"{country}_POLY_LCR", COUNTRY_COEFFICIENTS_LCR["EU_POLY_LCR"])
            intercept_lcr = COUNTRY_COEFFICIENTS_LCR.get(f"{country}_INTERCEPT_LCR", COUNTRY_COEFFICIENTS_LCR["EU_INTERCEPT_LCR"])

            base_lcr = metrics["base_loads"] / metrics["base_carriers"]
            compare_lcr = metrics["loads"] / metrics["carriers"]
            base_expected_carriers = (metrics["base_loads"] * slope + intercept) * (1 + percentage)
            compare_expected_carriers = (metrics["loads"] * slope + intercept) * (1 + percentage)
            base_expected_lcr = metrics["base_loads"] / base_expected_carriers
            compare_expected_lcr = metrics["loads"] / compare_expected_carriers

            carrier_impact_score = compare_lcr - base_lcr + base_expected_lcr - compare_expected_lcr
            demand_impact_score = compare_expected_lcr - base_expected_lcr

            carrier_impact = (
                (intercept_lcr + slope_lcr * (carrier_impact_score + base_lcr) + poly_lcr * ((carrier_impact_score + base_lcr) ** 2))
                / (intercept_lcr + slope_lcr * base_lcr + poly_lcr * (base_lcr ** 2)) - 1
            ) * metrics["base_cpkm"]

            demand_impact = (
                (intercept_lcr + slope_lcr * (demand_impact_score + base_lcr) + poly_lcr * ((demand_impact_score + base_lcr) ** 2))
                / (intercept_lcr + slope_lcr * base_lcr + poly_lcr * (base_lcr ** 2)) - 1
            ) * metrics["base_cpkm"]

            bridge_df.loc[idx, ["carrier_impact", "demand_impact"]] = [carrier_impact, demand_impact]

        # Tech impact
        try:
            # YoY / WoW → use explicit week
            if period_type in ["YoY", "WoW"] and time_period and time_period.startswith("W"):
                week_num = int(time_period.replace("W", ""))
                tech_rate = get_tech_savings_rate(compare_year, week_num)
                bridge_df.loc[idx, "tech_impact"] = metrics["cpkm"] * tech_rate

            # MTD → pick FIRST week of the month
            elif period_type == "MTD" and time_period and time_period.startswith("M"):
                if not compare_data.empty:
                    first_week = (
                        compare_data["report_week"]
                        .astype(str)
                        .sort_values()
                        .iloc[0]
                    )
                    week_num = int(first_week.replace("W", ""))
                    tech_rate = get_tech_savings_rate(compare_year, week_num)
                    bridge_df.loc[idx, "tech_impact"] = metrics["cpkm"] * tech_rate
                else:
                    bridge_df.loc[idx, "tech_impact"] = 0.0

        except (ValueError, TypeError, IndexError):
            bridge_df.loc[idx, "tech_impact"] = 0.0

