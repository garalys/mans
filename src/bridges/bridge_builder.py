"""
Bridge Structure Builder

Creates the scaffold structure for bridge analysis DataFrames.
Generates all combinations of time periods, countries, and businesses.
"""

import pandas as pd
from typing import List

from ..utils.metrics_utils import (
    create_yoy_bridge_row,
    create_wow_bridge_row,
    create_mtd_bridge_row,
)


def create_bridge_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create bridge structure for all country/business combinations.

    Generates rows for YoY, WoW, and MTD comparisons across all unique
    combinations of weeks, months, countries, and businesses in the data.

    Input Table - df:
        | Column       | Type   | Description                     |
        |--------------|--------|---------------------------------|
        | report_week  | string | Week in WXX format              |
        | report_month | string | Month in MXX format             |
        | orig_country | string | Origin country code             |
        | business     | string | Business unit identifier        |
        | report_year  | string | Year in R20XX format            |

    Output Table - Bridge DataFrame:
        | Column         | Type   | Description                       |
        |----------------|--------|-----------------------------------|
        | report_year    | string | Year/comparison year              |
        | report_week    | string | Week identifier (YoY/WoW)         |
        | report_month   | string | Month identifier (MTD)            |
        | orig_country   | string | Country code (incl. 'EU')         |
        | business       | string | Business unit                     |
        | bridge_type    | string | 'YoY', 'WoW', or 'MTD'            |
        | bridging_value | string | Period comparison identifier      |
        | base_cpkm      | float  | Base period CPKM (to fill)        |
        | compare_cpkm   | float  | Compare period CPKM (to fill)     |
        | *_impact       | float  | Various impact metrics (to fill)  |

    Note:
        - YoY: Only consecutive years (R2024 -> R2025, not R2024 -> R2026)
        - WoW: Sequential weeks within same year
        - MTD: Same month between consecutive years
        - 'EU' is added as a virtual country for aggregations
    """
    # Get unique values from data
    weeks = sorted(df["report_week"].unique())
    months = sorted(df["report_month"].unique())
    countries = sorted(df["orig_country"].unique())
    businesses = sorted(df["business"].unique())
    years = sorted(df["report_year"].unique())

    # Add EU as aggregate country
    countries.append("EU")

    bridge_rows = []

    for country in countries:
        for business in businesses:
            # Create YoY combinations - ONLY CONSECUTIVE YEARS
            for i in range(len(years) - 1):
                base_year = years[i]
                compare_year = years[i + 1]

                # YoY weekly combinations
                for week in weeks:
                    row = create_yoy_bridge_row(
                        base_year, compare_year, week, country, business
                    )
                    bridge_rows.append(row)

                # MTD combinations
                for month in months:
                    row = create_mtd_bridge_row(
                        base_year, compare_year, month, country, business
                    )
                    bridge_rows.append(row)

            # WoW combinations (within same year)
            for year in years:
                week_pairs = list(zip(weeks[:-1], weeks[1:]))
                for w1, w2 in week_pairs:
                    row = create_wow_bridge_row(year, w1, w2, country, business)
                    bridge_rows.append(row)

    return pd.DataFrame(bridge_rows)


def create_bridge_structure_for_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create bridge structure for 'Total' business aggregations.

    Similar to create_bridge_structure but only creates rows for
    'Total' business (aggregated across all business units).

    Input Table - df:
        | Column       | Type   | Description                     |
        |--------------|--------|---------------------------------|
        | report_week  | string | Week in WXX format              |
        | report_month | string | Month in MXX format             |
        | orig_country | string | Origin country code             |
        | report_year  | string | Year in R20XX format            |

    Output Table - Bridge DataFrame:
        Same structure as create_bridge_structure, but:
        - business column always contains 'Total'
        - Represents aggregated metrics across all businesses
    """
    weeks = sorted(df["report_week"].unique())
    months = sorted(df["report_month"].unique())
    countries = sorted(df["orig_country"].unique())
    years = sorted(df["report_year"].unique())

    # Add EU as aggregate country
    countries.append("EU")

    bridge_rows = []

    for country in countries:
        # YoY combinations - ONLY CONSECUTIVE YEARS
        for i in range(len(years) - 1):
            base_year = years[i]
            compare_year = years[i + 1]

            for week in weeks:
                row = create_yoy_bridge_row(
                    base_year, compare_year, week, country, "Total"
                )
                bridge_rows.append(row)

            for month in months:
                row = create_mtd_bridge_row(
                    base_year, compare_year, month, country, "Total"
                )
                bridge_rows.append(row)

        # WoW combinations
        for year in years:
            week_pairs = list(zip(weeks[:-1], weeks[1:]))
            for w1, w2 in week_pairs:
                row = create_wow_bridge_row(year, w1, w2, country, "Total")
                bridge_rows.append(row)

    return pd.DataFrame(bridge_rows)
