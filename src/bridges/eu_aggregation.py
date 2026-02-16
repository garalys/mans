"""
EU-Level Aggregation Calculator

Calculates bridge metrics aggregated across all EU5 countries.
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
from ..calculators.carrier_calculator import calculate_active_carriers
from ..calculators.normalized_cost_calculator import calculate_normalized_cost
from ..calculators.market_rate_calculator import calculate_market_rate_impact
from ..calculators.mix_calculator import (
    compute_hierarchical_mix,
    compute_normalised_distance,
    compute_seven_metrics,
    compute_mix_impacts,
    compute_cell_cpkm,
    enrich_mix_pcts,
)
# from ..calculators.set_impact_calculator import calculate_set_impact  # Commented out - replaced by equipment_type_mix


def calculate_eu_aggregated_metrics(
    df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    df_carrier: pd.DataFrame,
) -> None:
    """
    Calculate EU-level aggregated bridge metrics.

    Aggregates metrics across all EU5 countries (DE, ES, FR, IT, UK) for
    rows where orig_country='EU'.

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

    Input Table - bridge_df (to be updated):
        | Column         | Type   | Description                        |
        |----------------|--------|----------------------------------- |
        | bridge_type    | string | 'YoY', 'WoW', or 'MTD'             |
        | orig_country   | string | Must be 'EU' for rows to update    |
        | business       | string | Business unit                      |

    Input Table - df_carrier:
        | Column     | Type   | Description                    |
        |------------|--------|--------------------------------|
        | year       | string | Year in R20XX format           |
        | period     | string | Period identifier              |
        | country    | string | Country code                   |
        | percentage | float  | Carrier scaling percentage     |

    Args:
        df: Main transportation data DataFrame
        bridge_df: Bridge structure DataFrame (modified in-place)
        df_carrier: Carrier scaling reference data

    Output:
        None - bridge_df rows with orig_country='EU' are updated with
        aggregated metrics across all countries.
    """
    # Pre-calculate aggregations
    df_aggs = {period_type: {} for period_type in ["YoY", "WoW", "MTD"]}

    # Process EU rows by period type
    for period_type in ["YoY", "WoW", "MTD"]:
        eu_rows = bridge_df[
            (bridge_df["bridge_type"] == period_type)
            & (bridge_df["orig_country"] == "EU")
        ]

        # Process in batches
        batch_size = 1000
        for start_idx in range(0, len(eu_rows), batch_size):
            batch = eu_rows.iloc[start_idx : start_idx + batch_size]

            for idx, row in batch.iterrows():
                # Get time period data based on bridge type
                if period_type == "YoY":
                    base_year, compare_year = row["bridging_value"].split("_to_")
                    time_period = row["report_week"]
                    key = (base_year, compare_year, time_period)

                    if key not in df_aggs[period_type]:
                        df_aggs[period_type][key] = {
                            "base": df[
                                (df["report_year"] == base_year)
                                & (df["report_week"] == time_period)
                            ],
                            "compare": df[
                                (df["report_year"] == compare_year)
                                & (df["report_week"] == time_period)
                            ],
                        }

                    base_data = df_aggs[period_type][key]["base"]
                    compare_data = df_aggs[period_type][key]["compare"]

                elif period_type == "WoW":
                    year = row["report_year"]
                    parts = row["bridging_value"].split("_")
                    w1, w2 = parts[1], parts[3]
                    time_period = w2
                    key = (year, w1, w2)

                    if key not in df_aggs[period_type]:
                        df_aggs[period_type][key] = {
                            "base": df[
                                (df["report_year"] == year) & (df["report_week"] == w1)
                            ],
                            "compare": df[
                                (df["report_year"] == year) & (df["report_week"] == w2)
                            ],
                        }

                    base_data = df_aggs[period_type][key]["base"]
                    compare_data = df_aggs[period_type][key]["compare"]

                else:  # MTD
                    match = re.match(r"(.+)_(.+)_to_(.+)_.+", row["bridging_value"])
                    base_year, month, compare_year = match.groups()
                    start_date, end_date = get_mtd_date_range(
                        df, compare_year, int(month.replace("M", ""))
                    )

                    if start_date is None:
                        continue

                    key = (base_year, compare_year, month)
                    if key not in df_aggs[period_type]:
                        df_aggs[period_type][key] = {
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

                    base_data = df_aggs[period_type][key]["base"]
                    compare_data = df_aggs[period_type][key]["compare"]
                    time_period = month

                # Calculate EU metrics
                eu_metrics = _calculate_eu_business_metrics(
                    base_data, compare_data, time_period, row["business"]
                )

                # Update bridge with EU metrics
                _update_bridge_with_eu_metrics(
                    bridge_df,
                    eu_metrics,
                    idx,
                    base_data,
                    compare_data,
                    period_type,
                    time_period,
                    compare_year if period_type in ["YoY", "MTD"] else extract_year_from_report_year(row["report_year"]),
                    df_carrier,
                )


def _calculate_eu_business_metrics(
    base_data: pd.DataFrame,
    compare_data: pd.DataFrame,
    time_period: str,
    business: str,
) -> dict:
    """Calculate EU metrics for a specific business."""
    # Filter by business
    base_business = base_data[base_data["business"] == business]
    compare_business = compare_data[compare_data["business"] == business]

    # Aggregate metrics
    eu_base = base_business.agg({
        "distance_for_cpkm": "sum",
        "total_cost_usd": "sum",
        "executed_loads": "sum",
    })

    eu_compare = compare_business.agg({
        "distance_for_cpkm": "sum",
        "total_cost_usd": "sum",
        "executed_loads": "sum",
    })

    metrics = {
        "distance_km": eu_compare["distance_for_cpkm"],
        "costs_usd": eu_compare["total_cost_usd"],
        "loads": eu_compare["executed_loads"],
        "base_distance_km": eu_base["distance_for_cpkm"],
        "base_costs_usd": eu_base["total_cost_usd"],
        "base_loads": eu_base["executed_loads"],
    }

    # Calculate CPKMs
    metrics["base_cpkm"] = (
        metrics["base_costs_usd"] / metrics["base_distance_km"]
        if metrics["base_distance_km"] > 0
        else None
    )
    metrics["cpkm"] = (
        metrics["costs_usd"] / metrics["distance_km"]
        if metrics["distance_km"] > 0
        else None
    )

    # Calculate carriers
    metrics["base_carriers"] = calculate_active_carriers(base_business, time_period, "EU")
    metrics["carriers"] = calculate_active_carriers(compare_business, time_period, "EU")

    return metrics


def _update_bridge_with_eu_metrics(
    bridge_df: pd.DataFrame,
    metrics: dict,
    idx: int,
    base_data: pd.DataFrame,
    compare_data: pd.DataFrame,
    period_type: str,
    time_period: str,
    compare_year,
    df_carrier: pd.DataFrame,
) -> None:
    """Update bridge DataFrame with EU-aggregated metrics."""
    if base_data.empty or compare_data.empty:
        return

    # Determine prefix mapping based on period type
    if period_type == "YoY":
        prefix_map = {"base": f"base_", "compare": f"compare_"}
    elif period_type == "WoW":
        prefix_map = {"base": "w1_", "compare": "w2_"}
    else:  # MTD
        prefix_map = {"base": "m1_", "compare": "m2_"}

    # Update base metrics
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
    }

    bridge_df.loc[idx, list(update_dict.keys())] = list(update_dict.values())

    # Calculate bridge components if valid
    if metrics["base_cpkm"] is not None and metrics["cpkm"] is not None:
        business = bridge_df.loc[idx, "business"]
        impact_updates = _calculate_eu_impacts(
            base_data, compare_data, metrics, business, time_period, compare_year, df_carrier, bridge_type=period_type
        )
        bridge_df.loc[idx, list(impact_updates.keys())] = list(impact_updates.values())


def _calculate_eu_impacts(
    base_data: pd.DataFrame,
    compare_data: pd.DataFrame,
    metrics: dict,
    business: str,
    time_period: str,
    compare_year,
    df_carrier: pd.DataFrame,
    bridge_type: str,
) -> dict:
    """Calculate impact components for EU aggregation."""
    impact_updates = {}

    # Hierarchical Mix Impact (EU level)
    # Filter to this business (for distances, CPKMs)
    business_data = {
        "base": base_data[base_data["business"] == business],
        "compare": compare_data[compare_data["business"] == business],
    }

    # Compute grain from FILTERED (business) data — bridge's own distances/CPKMs
    filtered_base_mix = compute_hierarchical_mix(business_data["base"])
    filtered_compare_mix = compute_hierarchical_mix(business_data["compare"])

    # Compute full-data mix grains (all countries, all businesses) for percentages
    full_base_mix = compute_hierarchical_mix(base_data)
    full_compare_mix = compute_hierarchical_mix(compare_data)

    # Enrich filtered grain with full-data percentages
    base_mix = enrich_mix_pcts(filtered_base_mix, full_base_mix)
    compare_mix = enrich_mix_pcts(filtered_compare_mix, full_compare_mix)

    # Normalised distance + CPKMs from FILTERED data
    norm_dist = compute_normalised_distance(filtered_base_mix, filtered_compare_mix)
    base_cpkm_cells = compute_cell_cpkm(business_data["base"])
    compare_cpkm_cells = compute_cell_cpkm(business_data["compare"])

    seven = compute_seven_metrics(
        base_mix, compare_mix, norm_dist, base_cpkm_cells, compare_cpkm_cells
    )

    mix_results = compute_mix_impacts(
        seven, metrics["distance_km"],
        bridge_level="business",  # EU business-level
        aggregation_level="eu",
    )

    # Use business-filtered data for market rate impact
    market_rate_impact = calculate_market_rate_impact(
        business_data["base"], business_data["compare"], "EU"
    )

    impact_updates.update({
        "mix_impact": mix_results["mix_impact"],
        "normalised_cpkm": mix_results["normalised_cpkm"],
        "country_mix": mix_results["country_mix"],
        "corridor_mix": mix_results["corridor_mix"],
        "distance_band_mix": mix_results["distance_band_mix"],
        "business_flow_mix": mix_results["business_flow_mix"],
        "equipment_type_mix": mix_results["equipment_type_mix"],
        "market_rate_impact": market_rate_impact,
    })

    # Carrier and demand impacts
    if metrics["carriers"] > 0 and metrics["base_carriers"] > 0:
        country = "EU"
        period = time_period if time_period.startswith(("W", "M")) else None

        percentage = 0.0
        if period is not None:
            if isinstance(compare_year, str):
                year_str = compare_year if compare_year.startswith("R") else f"R{compare_year}"
            else:
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

        impact_updates["carrier_impact"] = carrier_impact
        impact_updates["demand_impact"] = demand_impact

    # Tech impact
    if time_period:

        compare_year_num = (
            compare_year
            if isinstance(compare_year, int)
            else extract_year_from_report_year(str(compare_year))
        )

        # --------------------------------------------------
        # WEEKLY (WoW / YoY)
        # --------------------------------------------------
        if time_period.startswith("W"):

            week_num = int(time_period.replace("W", ""))

            # --- WoW ---
            if bridge_type == "WoW":

                base_week_num = week_num - 1
                base_tech = get_tech_savings_rate(compare_year_num, base_week_num)
                compare_tech = get_tech_savings_rate(compare_year_num, week_num)

                # Only impact if tech rate changed
                if abs(base_tech - compare_tech) < 1e-9:
                    impact_updates["tech_impact"] = 0.0
                else:
                    impact_updates["tech_impact"] = (
                        metrics.get("base_cpkm", 0.0)
                        * (compare_tech - base_tech)
                    )

            # --- YoY ---
            else:
                tech_rate = get_tech_savings_rate(compare_year_num, week_num)
                impact_updates["tech_impact"] = metrics.get("cpkm", 0.0) * tech_rate

        # --------------------------------------------------
        # MTD (first week of month)
        # --------------------------------------------------
        elif time_period.startswith("M"):

            df = business_data.get("compare")

            if df is not None and not df.empty:

                first_week = (
                    df["report_week"]
                    .astype(str)
                    .sort_values()
                    .iloc[0]
                )

                week_num = int(first_week.replace("W", ""))
                tech_rate = get_tech_savings_rate(compare_year_num, week_num)

                impact_updates["tech_impact"] = (
                    metrics.get("cpkm", 0.0) * tech_rate
                )

            else:
                impact_updates["tech_impact"] = 0.0

    return impact_updates

