"""
Month-to-Date Bridge Calculator

Calculates bridge metrics comparing same months between consecutive years,
aligned to the maximum available date in the comparison month.
"""

import pandas as pd

from ..utils.date_utils import extract_year_from_report_year, get_mtd_date_range
from .bridge_metrics import calculate_detailed_bridge_metrics


def calculate_mtd_bridge_metrics(
    df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    df_carrier: pd.DataFrame,
) -> None:
    """
    Calculate Month-to-Date bridge metrics between consecutive years.

    Compares the same month between years, but only up to the latest
    available date in the comparison month (MTD alignment).

    Input Table - df (main data):
        | Column           | Type     | Description                   |
        |------------------|----------|-------------------------------|
        | report_year      | string   | Year in R20XX format          |
        | report_month     | string   | Month in MXX format           |
        | report_day       | datetime | Date of the report            |
        | orig_country     | string   | Origin country code           |
        | business         | string   | Business unit                 |
        | distance_for_cpkm| float    | Distance in kilometers        |
        | total_cost_usd   | float    | Total cost in USD             |
        | + other columns for impact calculations                     |

    Input Table - bridge_df (to be updated):
        | Column         | Type   | Description                       |
        |----------------|--------|-----------------------------------|
        | bridge_type    | string | Must be 'MTD' for rows to update  |
        | bridging_value | string | Format: 'R2024_M01_to_R2025_M01'  |
        | report_month   | string | Month being compared              |
        | orig_country   | string | Country code                      |
        | business       | string | Business unit                     |

    Input Table - df_carrier:
        | Column     | Type   | Description                    |
        |------------|--------|--------------------------------|
        | year       | string | Year in R20XX format           |
        | period     | string | Period (M01, etc.)             |
        | country    | string | Country code                   |
        | percentage | float  | Carrier scaling percentage     |

    Args:
        df: Main transportation data DataFrame
        bridge_df: Bridge structure DataFrame (modified in-place)
        df_carrier: Carrier scaling reference data

    Output:
        None - bridge_df is modified in-place with:
            - m1_* columns: Base month metrics
            - m2_* columns: Compare month metrics
            - Bridge component columns

    Note:
        MTD alignment means if current data goes to March 15, the base year
        comparison will also only include data through March 15.
    """
    years = sorted(df["report_year"].unique())

    # Process only consecutive year pairs
    for i in range(len(years) - 1):
        base_year = years[i]
        compare_year = years[i + 1]

        mtd_rows = bridge_df[
            (bridge_df["bridge_type"] == "MTD")
            & (bridge_df["bridging_value"].str.contains(f"{base_year}_.*_to_{compare_year}"))
        ]

        for idx, row in mtd_rows.iterrows():
            month = row["report_month"]
            country = row["orig_country"]
            business = row["business"]

            # Get MTD date range based on compare year data
            start_date, end_date = get_mtd_date_range(
                df, compare_year, int(month.replace("M", ""))
            )
            if start_date is None:
                continue

            # Filter base year data with MTD alignment
            base_data = df[
                (df["report_year"] == base_year)
                & (df["report_month"] == month)
                & (df["report_day"] >= start_date.replace(year=extract_year_from_report_year(base_year)))
                & (df["report_day"] <= end_date.replace(year=extract_year_from_report_year(base_year)))
                & (df["orig_country"] == country)
                & (df["business"] == business)
            ]

            # Filter compare year data
            compare_data = df[
                (df["report_year"] == compare_year)
                & (df["report_month"] == month)
                & (df["report_day"] >= start_date)
                & (df["report_day"] <= end_date)
                & (df["orig_country"] == country)
                & (df["business"] == business)
            ]

            # Skip if either dataset is empty
            if base_data.empty or compare_data.empty:
                continue

            # Calculate metrics
            metrics = calculate_detailed_bridge_metrics(
                base_data,
                compare_data,
                country,
                extract_year_from_report_year(base_year),
                extract_year_from_report_year(compare_year),
                df_carrier,
                report_month=month,
            )

            # Update MTD-specific columns
            _update_mtd_metrics(bridge_df, idx, metrics)


def _update_mtd_metrics(
    bridge_df: pd.DataFrame,
    idx: int,
    metrics: dict,
) -> None:
    """
    Update MTD-specific metrics in bridge_df.

    Maps base metrics to m1_* columns and compare metrics to m2_* columns.
    """
    # Map base metrics to m1 columns
    metrics_mapping = {
        "base_distance_km": "m1_distance_km",
        "base_costs_usd": "m1_costs_usd",
        "base_loads": "m1_loads",
        "base_carriers": "m1_carriers",
        "base_cpkm": "base_cpkm",
        "compare_distance_km": "m2_distance_km",
        "compare_costs_usd": "m2_costs_usd",
        "compare_loads": "m2_loads",
        "compare_carriers": "m2_carriers",
        "compare_cpkm": "compare_cpkm",
    }

    for old_key, new_key in metrics_mapping.items():
        if old_key in metrics:
            bridge_df.loc[idx, new_key] = metrics[old_key]

    # Update bridge components
    for metric, value in metrics.items():
        if metric not in metrics_mapping:
            bridge_df.loc[idx, metric] = value
