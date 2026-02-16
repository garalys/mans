"""
Carrier and Demand Impact Calculator

Calculates the impact of carrier availability and demand changes on costs.
Uses Load-to-Carrier Ratio (LCR) model to decompose cost changes.
"""

import pandas as pd
from typing import Tuple, Optional

from ..config.settings import COUNTRY_COEFFICIENTS_VOLUME, COUNTRY_COEFFICIENTS_LCR


def calculate_active_carriers(
    data: pd.DataFrame,
    period: str,
    country: str,
    business: Optional[str] = None,
) -> float:
    """
    Calculate active carriers count matching SQL logic.

    For EU-level calculations, uses fractional carrier counting based on
    each carrier's share of total loads. For country-level, uses unique
    carrier count.

    Input Table - data:
        | Column         | Type   | Description                       |
        |----------------|--------|-----------------------------------|
        | vehicle_carrier| string | Carrier identifier                |
        | executed_loads | int    | Number of loads executed          |
        | business       | string | Business unit (optional filter)   |

    Args:
        data: DataFrame containing carrier and load data
        period: Period identifier (e.g., 'W01', 'M01')
        country: Country code or 'EU'
        business: Optional business filter

    Output:
        float: Number of active carriers
            - For EU: Fractional count based on load distribution
            - For countries: Unique carrier count with executed loads > 0

    Formula (EU):
        For each carrier: fractional_carrier = carrier_loads / total_carrier_loads
        Total = sum of all fractional carriers

    Example:
        If carrier A has 100 loads (50% of their total) and carrier B has
        200 loads (100% of their total), EU carriers = 0.5 + 1.0 = 1.5
    """
    if country == "EU":
        # Calculate total loads by carrier
        carrier_total_loads = data.groupby("vehicle_carrier")["executed_loads"].sum()

        # For each carrier, divide its loads by its own total loads
        fractional_carriers = (
            data[data["executed_loads"] > 0]
            .groupby("vehicle_carrier")["executed_loads"]
            .apply(lambda x: x / carrier_total_loads[x.name])
            .sum()
        )

        return fractional_carriers
    else:
        # For country-level, count unique carriers with executed loads
        mask = data["executed_loads"] > 0
        if business is not None:
            mask &= data["business"] == business
        return data.loc[mask, "vehicle_carrier"].nunique()


def calculate_carrier_demand_impacts(
    base_loads: float,
    base_carriers: float,
    compare_loads: float,
    compare_carriers: float,
    base_cpkm: float,
    country: str,
    df_carrier: pd.DataFrame,
    compare_year: int,
    period: str,
) -> Tuple[float, float]:
    """
    Calculate carrier and demand impacts using LCR model.

    This function decomposes the change in costs into carrier availability
    impact and demand impact using a polynomial LCR model.

    Input Parameters:
        | Parameter        | Type  | Description                           |
        |------------------|-------|---------------------------------------|
        | base_loads       | float | Base period executed loads            |
        | base_carriers    | float | Base period active carriers           |
        | compare_loads    | float | Comparison period executed loads      |
        | compare_carriers | float | Comparison period active carriers     |
        | base_cpkm        | float | Base period cost per kilometer        |
        | country          | str   | Country code for coefficient lookup   |
        | df_carrier       | DF    | Carrier scaling percentages           |
        | compare_year     | int   | Year for lookup (e.g., 2025)          |
        | period           | str   | Period identifier (W01, M01, etc.)    |

    Input Table - df_carrier:
        | Column     | Type   | Description                           |
        |------------|--------|---------------------------------------|
        | year       | string | Year in R20XX format                  |
        | period     | string | Period (W01, M01, etc.)               |
        | country    | string | Country code                          |
        | percentage | float  | Carrier scaling adjustment            |

    Output:
        Tuple[float, float]: (carrier_impact, demand_impact) in $/km

    Formula:
        1. LCR = loads / carriers
        2. expected_carriers = (loads * slope + intercept) * (1 + percentage)
        3. expected_lcr = loads / expected_carriers
        4. carrier_impact_score = compare_lcr - base_lcr + base_expected_lcr - compare_expected_lcr
        5. demand_impact_score = compare_expected_lcr - base_expected_lcr
        6. Impact = (polynomial(new_lcr) / polynomial(base_lcr) - 1) * base_cpkm

    Note:
        Polynomial: intercept_lcr + slope_lcr * LCR + poly_lcr * LCR^2
    """
    if compare_carriers <= 0 or base_carriers <= 0:
        return 0.0, 0.0

    # Get volume coefficients for expected carrier calculation
    slope = COUNTRY_COEFFICIENTS_VOLUME.get(
        f"{country}_SLOPE", COUNTRY_COEFFICIENTS_VOLUME["EU_SLOPE"]
    )
    intercept = COUNTRY_COEFFICIENTS_VOLUME.get(
        f"{country}_INTERCEPT", COUNTRY_COEFFICIENTS_VOLUME["EU_INTERCEPT"]
    )

    # Get LCR coefficients for polynomial model
    slope_lcr = COUNTRY_COEFFICIENTS_LCR.get(
        f"{country}_SLOPE_LCR", COUNTRY_COEFFICIENTS_LCR["EU_SLOPE_LCR"]
    )
    poly_lcr = COUNTRY_COEFFICIENTS_LCR.get(
        f"{country}_POLY_LCR", COUNTRY_COEFFICIENTS_LCR["EU_POLY_LCR"]
    )
    intercept_lcr = COUNTRY_COEFFICIENTS_LCR.get(
        f"{country}_INTERCEPT_LCR", COUNTRY_COEFFICIENTS_LCR["EU_INTERCEPT_LCR"]
    )

    # Lookup carrier scaling percentage
    percentage = _lookup_carrier_percentage(df_carrier, compare_year, period, country)

    # Calculate Load-to-Carrier Ratios
    base_lcr = base_loads / base_carriers
    compare_lcr = compare_loads / compare_carriers

    # Calculate expected executing carriers based on loads
    base_expected_carriers = (base_loads * slope + intercept) * (1 + percentage)
    compare_expected_carriers = (compare_loads * slope + intercept) * (1 + percentage)

    # Calculate expected LCRs
    base_expected_lcr = base_loads / base_expected_carriers
    compare_expected_lcr = compare_loads / compare_expected_carriers

    # Decompose into carrier and demand impact scores
    carrier_impact_score = (
        compare_lcr - base_lcr + base_expected_lcr - compare_expected_lcr
    )
    demand_impact_score = compare_expected_lcr - base_expected_lcr

    # Calculate impacts using polynomial model
    carrier_impact = _calculate_lcr_impact(
        carrier_impact_score, base_lcr, base_cpkm, intercept_lcr, slope_lcr, poly_lcr
    )
    demand_impact = _calculate_lcr_impact(
        demand_impact_score, base_lcr, base_cpkm, intercept_lcr, slope_lcr, poly_lcr
    )

    return carrier_impact, demand_impact


def _lookup_carrier_percentage(
    df_carrier: pd.DataFrame,
    compare_year: int,
    period: str,
    country: str,
) -> float:
    """Look up carrier scaling percentage from reference table."""
    year_str = f"R{compare_year}"
    lookup_mask = (
        (df_carrier["year"] == year_str)
        & (df_carrier["period"] == period)
        & (df_carrier["country"] == country)
    )
    matching_rows = df_carrier[lookup_mask]

    if not matching_rows.empty:
        return matching_rows["percentage"].iloc[0]
    return 0.0


def _calculate_lcr_impact(
    impact_score: float,
    base_lcr: float,
    base_cpkm: float,
    intercept_lcr: float,
    slope_lcr: float,
    poly_lcr: float,
) -> float:
    """
    Calculate cost impact using polynomial LCR model.

    Formula:
        impact = (P(new_lcr) / P(base_lcr) - 1) * base_cpkm
        where P(x) = intercept + slope*x + poly*x^2
    """
    new_lcr = impact_score + base_lcr

    numerator = (
        intercept_lcr
        + slope_lcr * new_lcr
        + poly_lcr * (new_lcr ** 2)
    )
    denominator = (
        intercept_lcr
        + slope_lcr * base_lcr
        + poly_lcr * (base_lcr ** 2)
    )

    return (numerator / denominator - 1) * base_cpkm
