"""
Metrics Utility Functions

Helper functions for creating bridge row structures and metric dictionaries.
"""

from typing import Dict, Any


def create_base_metrics_dict() -> Dict[str, Any]:
    """
    Create dictionary with base bridge metrics initialized to None.

    Output:
        Dict with the following keys, all initialized to None:
            - base_cpkm: Base period cost per kilometer
            - mix_impact: Impact from distance/lane mix changes
            - normalised_cpkm: Cost per km after mix adjustment
            - supply_rates: Supply-side rate impact
            - carrier_and_demand_impact: Combined carrier + demand impact
            - carrier_impact: Impact from carrier availability changes
            - demand_impact: Impact from demand changes
            - premium_impact: Impact from premium pricing
            - market_rate_impact: Impact from market rate changes
            - tech_impact: Impact from technology initiatives
            - set_impact: Impact from SET pricing changes
            - compare_cpkm: Comparison period cost per kilometer
    """
    return {
        "base_cpkm": None,
        "mix_impact": None,
        "normalised_cpkm": None,
        "supply_rates": None,
        "carrier_and_demand_impact": None,
        "carrier_impact": None,
        "demand_impact": None,
        "premium_impact": None,
        "market_rate_impact": None,
        "tech_impact": None,
        "set_impact": None,
        "compare_cpkm": None,
    }


def create_yoy_bridge_row(
    base_year: str,
    compare_year: str,
    week: str,
    country: str,
    business: str,
) -> Dict[str, Any]:
    """
    Create a Year-over-Year (YoY) bridge row structure.

    Args:
        base_year: Base year in R20XX format (e.g., 'R2024')
        compare_year: Comparison year in R20XX format (e.g., 'R2025')
        week: Week identifier in WXX format (e.g., 'W01')
        country: Origin country code or 'EU'
        business: Business unit identifier

    Output:
        Dict with bridge row structure:
            | Key             | Type   | Description                      |
            |-----------------|--------|----------------------------------|
            | report_year     | string | Comparison year                  |
            | report_week     | string | Week identifier                  |
            | orig_country    | string | Country code                     |
            | business        | string | Business unit                    |
            | bridge_type     | string | Always 'YoY'                     |
            | bridging_value  | string | 'R2024_to_R2025' format          |
            | + base metrics  | various| From create_base_metrics_dict()  |
    """
    return {
        "report_year": compare_year,
        "report_week": week,
        "orig_country": country,
        "business": business,
        "bridge_type": "YoY",
        "bridging_value": f"{base_year}_to_{compare_year}",
        **create_base_metrics_dict(),
    }


def create_wow_bridge_row(
    year: str,
    week1: str,
    week2: str,
    country: str,
    business: str,
) -> Dict[str, Any]:
    """
    Create a Week-over-Week (WoW) bridge row structure.

    Args:
        year: Report year in R20XX format
        week1: Base week in WXX format (e.g., 'W01')
        week2: Comparison week in WXX format (e.g., 'W02')
        country: Origin country code or 'EU'
        business: Business unit identifier

    Output:
        Dict with bridge row structure:
            | Key             | Type   | Description                      |
            |-----------------|--------|----------------------------------|
            | report_year     | string | Report year                      |
            | report_week     | string | Comparison week                  |
            | orig_country    | string | Country code                     |
            | business        | string | Business unit                    |
            | bridge_type     | string | Always 'WoW'                     |
            | bridging_value  | string | 'R2025_W01_to_W02' format        |
            | w1_*            | various| Week 1 metrics (initialized)     |
            | w2_*            | various| Week 2 metrics (initialized)     |
            | + base metrics  | various| From create_base_metrics_dict()  |
    """
    return {
        "report_year": year,
        "report_week": week2,
        "orig_country": country,
        "business": business,
        "bridge_type": "WoW",
        "bridging_value": f"{year}_{week1}_to_{week2}",
        **create_base_metrics_dict(),
        "w1_distance_km": None,
        "w1_costs_usd": None,
        "w1_loads": None,
        "w1_carriers": None,
        "w1_cpkm": None,
        "w2_distance_km": None,
        "w2_costs_usd": None,
        "w2_loads": None,
        "w2_carriers": None,
        "w2_cpkm": None,
    }


def create_mtd_bridge_row(
    base_year: str,
    compare_year: str,
    month: str,
    country: str,
    business: str,
) -> Dict[str, Any]:
    """
    Create a Month-to-Date (MTD) bridge row structure.

    Args:
        base_year: Base year in R20XX format
        compare_year: Comparison year in R20XX format
        month: Month identifier in MXX format (e.g., 'M01')
        country: Origin country code or 'EU'
        business: Business unit identifier

    Output:
        Dict with bridge row structure:
            | Key             | Type   | Description                      |
            |-----------------|--------|----------------------------------|
            | report_year     | string | Comparison year                  |
            | report_month    | string | Month identifier                 |
            | orig_country    | string | Country code                     |
            | business        | string | Business unit                    |
            | bridge_type     | string | Always 'MTD'                     |
            | bridging_value  | string | 'R2024_M01_to_R2025_M01' format  |
            | m1_*            | various| Month 1 metrics (initialized)    |
            | m2_*            | various| Month 2 metrics (initialized)    |
            | + base metrics  | various| From create_base_metrics_dict()  |
    """
    return {
        "report_year": compare_year,
        "report_month": month,
        "orig_country": country,
        "business": business,
        "bridge_type": "MTD",
        "bridging_value": f"{base_year}_{month}_to_{compare_year}_{month}",
        **create_base_metrics_dict(),
        "m1_distance_km": None,
        "m1_costs_usd": None,
        "m1_loads": None,
        "m1_carriers": None,
        "m1_cpkm": None,
        "m2_distance_km": None,
        "m2_costs_usd": None,
        "m2_loads": None,
        "m2_carriers": None,
        "m2_cpkm": None,
    }
