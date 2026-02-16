"""
Bridge Metrics Calculator

Calculates detailed metrics for bridge visualization including all impact components.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from ..config.logging_config import logger

from ..config.settings import (
    COUNTRY_COEFFICIENTS_VOLUME,
    COUNTRY_COEFFICIENTS_LCR,
    get_tech_savings_rate,
)
from ..calculators.normalized_cost_calculator import calculate_normalized_cost
from ..calculators.market_rate_calculator import calculate_market_rate_impact
from ..calculators.carrier_calculator import (
    calculate_active_carriers,
    calculate_carrier_demand_impacts,
)
from ..calculators.mix_calculator import (
    compute_hierarchical_mix,
    compute_normalised_distance,
    compute_seven_metrics,
    compute_mix_impacts,
    compute_cell_cpkm,
    enrich_mix_pcts,
)
# from ..calculators.set_impact_calculator import calculate_set_impact  # Commented out - replaced by equipment_type_mix


def calculate_detailed_bridge_metrics(
    base_data: pd.DataFrame,
    compare_data: pd.DataFrame,
    country: str,
    base_year: int,
    compare_year: int,
    df_carrier: pd.DataFrame,
    report_week: Optional[str] = None,
    report_month: Optional[str] = None,
    bridge_type: str = "YoY",
    full_base_mix: Optional[pd.DataFrame] = None,
    full_compare_mix: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Calculate detailed metrics for bridge visualization.

    This is the core function that computes all bridge components including
    mix impact, carrier/demand impacts, market rate impact, SET impact,
    and tech impact.

    Input Table - base_data:
        | Column                         | Type    | Description                  |
        |--------------------------------|---------|------------------------------|
        | distance_for_cpkm              | float   | Distance in kilometers       |
        | total_cost_usd                 | float   | Total cost in USD            |
        | executed_loads                 | int     | Number of loads              |
        | vehicle_carrier                | string  | Carrier identifier           |
        | report_week                    | string  | Week in WXX format           |
        | business                       | string  | Business unit                |
        | dest_country                   | string  | Destination country          |
        | distance_band                  | string  | Distance category            |
        | is_set                         | boolean | SET pricing flag             |
        | transporeon_contract_price_eur | float   | Market rate in EUR           |

    Input Table - compare_data:
        Same structure as base_data for comparison period.

    Input Table - df_carrier:
        | Column     | Type   | Description                    |
        |------------|--------|--------------------------------|
        | year       | string | Year in R20XX format           |
        | period     | string | Period (W01, M01, etc.)        |
        | country    | string | Country code                   |
        | percentage | float  | Carrier scaling percentage     |

    Args:
        base_data: DataFrame with base period transportation data
        compare_data: DataFrame with comparison period data
        country: Country code or 'EU'
        base_year: Base year as integer (e.g., 2024)
        compare_year: Comparison year as integer (e.g., 2025)
        df_carrier: Carrier scaling percentages reference data
        report_week: Week identifier for lookup (optional)
        report_month: Month identifier for lookup (optional)

    Output:
        Dict with computed metrics:
            | Key                 | Type  | Description                          |
            |---------------------|-------|--------------------------------------|
            | base_cpkm           | float | Base period cost per km              |
            | compare_cpkm        | float | Compare period cost per km           |
            | mix_impact          | float | Impact from mix changes ($/km)       |
            | normalised_cpkm     | float | Normalized cost per km               |
            | carrier_impact      | float | Carrier availability impact ($/km)   |
            | demand_impact       | float | Demand change impact ($/km)          |
            | market_rate_impact  | float | Market rate change impact ($/km)     |
            | set_impact          | float | SET pricing impact ($/km)            |
            | tech_impact         | float | Technology savings impact ($/km)     |
            | base_distance_km    | float | Base period total distance           |
            | base_costs_usd      | float | Base period total cost               |
            | base_loads          | int   | Base period load count               |
            | base_carriers       | int   | Base period carrier count            |
            | compare_distance_km | float | Compare period total distance        |
            | compare_costs_usd   | float | Compare period total cost            |
            | compare_loads       | int   | Compare period load count            |
            | compare_carriers    | int   | Compare period carrier count         |
    """
    metrics = {}

    # Calculate base period aggregates
    base_dist = base_data["distance_for_cpkm"].sum()
    base_cost = base_data["total_cost_usd"].sum()
    base_loads = base_data["executed_loads"].sum()
    base_carriers = calculate_active_carriers(
        base_data,
        base_data["report_week"].iloc[0] if not base_data.empty else None,
        country,
        base_data["business"].iloc[0] if not base_data.empty else None,
    )

    # Calculate compare period aggregates
    compare_dist = compare_data["distance_for_cpkm"].sum()
    compare_cost = compare_data["total_cost_usd"].sum()
    compare_loads = compare_data["executed_loads"].sum()
    compare_carriers = calculate_active_carriers(
        compare_data,
        compare_data["report_week"].iloc[0] if not compare_data.empty else None,
        country,
        compare_data["business"].iloc[0] if not compare_data.empty else None,
    )

    # Calculate CPKMs
    base_cpkm = base_cost / base_dist if base_dist > 0 else None
    compare_cpkm = compare_cost / compare_dist if compare_dist > 0 else None

    metrics.update({
        "base_cpkm": base_cpkm,
        "compare_cpkm": compare_cpkm,
    })

    # Calculate bridge components if we have valid data
    if base_dist > 0 and compare_dist > 0:
        # Hierarchical Mix Impact
        # Compute grain from FILTERED data (bridge's own distances and CPKMs)
        filtered_base_mix = compute_hierarchical_mix(base_data)
        filtered_compare_mix = compute_hierarchical_mix(compare_data)

        # Enrich with full-data percentages if provided (same pcts for all bridges in a period)
        if full_base_mix is not None:
            base_mix = enrich_mix_pcts(filtered_base_mix, full_base_mix)
            compare_mix = enrich_mix_pcts(filtered_compare_mix, full_compare_mix)
        else:
            base_mix = filtered_base_mix
            compare_mix = filtered_compare_mix

        # Normalised distance from FILTERED grains (bridge's own distances)
        norm_dist = compute_normalised_distance(filtered_base_mix, filtered_compare_mix)
        # Cell CPKMs from FILTERED data
        base_cpkm_cells = compute_cell_cpkm(base_data)
        compare_cpkm_cells = compute_cell_cpkm(compare_data)

        # Seven metrics: enriched mixes (full-data pcts) + filtered distances/CPKMs
        seven = compute_seven_metrics(
            base_mix, compare_mix, norm_dist, base_cpkm_cells, compare_cpkm_cells
        )

        # Determine aggregation level based on country
        agg_level = "country" if country != "EU" else "eu"

        # Mix impacts using FILTERED compare distance (bridge-specific)
        mix_results = compute_mix_impacts(
            seven, compare_dist,
            bridge_level="business",  # business-level bridges (called per-business)
            aggregation_level=agg_level,
        )

        metrics["mix_impact"] = mix_results["mix_impact"]
        metrics["normalised_cpkm"] = mix_results["normalised_cpkm"]
        metrics["country_mix"] = mix_results["country_mix"]
        metrics["corridor_mix"] = mix_results["corridor_mix"]
        metrics["distance_band_mix"] = mix_results["distance_band_mix"]
        metrics["business_flow_mix"] = mix_results["business_flow_mix"]
        metrics["equipment_type_mix"] = mix_results["equipment_type_mix"]

        # Market Rate Impact
        if base_data["distance_for_cpkm"].sum() > 0 and compare_data["distance_for_cpkm"].sum() > 0:
            market_rate_impact = calculate_market_rate_impact(base_data, compare_data)
            metrics["market_rate_impact"] = market_rate_impact
        else:
            metrics["market_rate_impact"] = 0

        # Carrier and Demand Impacts
        if compare_carriers > 0 and base_carriers > 0:
            period = report_week or report_month
            if period is None:
                if not compare_data.empty and "report_week" in compare_data.columns:
                    period = compare_data["report_week"].iloc[0]
                elif not compare_data.empty and "report_month" in compare_data.columns:
                    period = compare_data["report_month"].iloc[0]

            carrier_impact, demand_impact = calculate_carrier_demand_impacts(
                base_loads=base_loads,
                base_carriers=base_carriers,
                compare_loads=compare_loads,
                compare_carriers=compare_carriers,
                base_cpkm=base_cpkm,
                country=country,
                df_carrier=df_carrier,
                compare_year=compare_year,
                period=period or "",
            )
            metrics["carrier_impact"] = carrier_impact
            metrics["demand_impact"] = demand_impact
        else:
            metrics["carrier_impact"] = 0.0
            metrics["demand_impact"] = 0.0

        # SET Impact - commented out, replaced by equipment_type_mix
        # set_impact = calculate_set_impact(base_data, compare_data)
        # metrics["set_impact"] = set_impact / compare_dist if compare_dist > 0 else None

        # Tech Impact
        if bridge_type == "WoW" and not base_data.empty and not compare_data.empty:
            # WoW: tech impact only when tech percentages change
            base_week_num = int(base_data["report_week"].iloc[0].replace("W", ""))
            compare_week_num = int(compare_data["report_week"].iloc[0].replace("W", ""))
            base_tech_pct = get_tech_savings_rate(base_year, base_week_num)
            compare_tech_pct = get_tech_savings_rate(compare_year, compare_week_num)

            if base_tech_pct == compare_tech_pct:
                logger.info(
                    f"Base = Compare WOW"
                    f"WoW check → "
                    f"Base: Y{base_year} W{base_week_num} ({base_tech_pct}) | "
                    f"Compare: Y{compare_year} W{compare_week_num} ({compare_tech_pct})"
                    )                   
                metrics["tech_impact"] = 0.0
            else:
                logger.info(
                    f"Base != Compare WOW"
                    f"WoW check → "
                    f"Base: Y{base_year} W{base_week_num} ({base_tech_pct}) | "
                    f"Compare: Y{compare_year} W{compare_week_num} ({compare_tech_pct})"
                    )   
                # previous_week_cpkm * (this_week_tech_pct - previous_week_tech_pct)
                metrics["tech_impact"] = base_cpkm * (compare_tech_pct - base_tech_pct) if base_cpkm else 0.0
        else:
            week_num = int(base_data["report_week"].iloc[0].replace("W", "")) if not base_data.empty else 0
            tech_rate = get_tech_savings_rate(compare_year, week_num)
            tech_impact = compare_cpkm * tech_rate if compare_cpkm is not None else 0.0
            metrics["tech_impact"] = tech_impact

    else:
        # No valid data - set all impacts to None
        metrics.update({
            "mix_impact": None,
            "normalised_cpkm": None,
            "supply_rates": None,
            "carrier_and_demand_impact": None,
            "carrier_impact": None,
            "demand_impact": None,
            "premium_impact": None,
            "market_rate_impact": None,
            # "set_impact": None,  # Commented out - replaced by equipment_type_mix
            "tech_impact": None,
        })

    # Store raw values for flexible comparison
    metrics.update({
        "base_distance_km": base_dist,
        "base_costs_usd": base_cost if base_cost > 0 else 0,
        "base_loads": base_loads,
        "base_carriers": base_carriers,
        "compare_distance_km": compare_dist,
        "compare_costs_usd": compare_cost if compare_cost > 0 else 0,
        "compare_loads": compare_loads,
        "compare_carriers": compare_carriers,
    })

    return metrics
