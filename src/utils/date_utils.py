"""
Date Utility Functions

Helper functions for date manipulation and period handling.
"""

import pandas as pd
from typing import Tuple, Optional

from ..config.logging_config import logger


def extract_year_from_report_year(report_year: str) -> int:
    """
    Extract numeric year from R20XX format.

    Input:
        report_year (str): Year string in format 'R20XX' (e.g., 'R2025')

    Output:
        int: Numeric year (e.g., 2025)

    Example:
        >>> extract_year_from_report_year('R2025')
        2025
    """
    return int(report_year.replace("R", ""))


def get_mtd_date_range(
    df: pd.DataFrame,
    year: str,
    month: int,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Get available date range for a month-to-date comparison.

    Returns the first day of the month and the maximum available date
    in the data for that month.

    Input Table - df:
        | Column       | Type     | Description                        |
        |--------------|----------|------------------------------------|
        | report_day   | datetime | Date of the report                 |
        | report_year  | string   | Year in R20XX format               |
        | report_month | string   | Month in MXX format (e.g., 'M01')  |

    Args:
        df: DataFrame containing date columns
        year: Year string in R20XX format (e.g., 'R2025')
        month: Month number (1-12)

    Output:
        Tuple[Optional[Timestamp], Optional[Timestamp]]:
            - start_date: First day of the month
            - end_date: Maximum available date in the month
            Returns (None, None) if no data found for the period

    Example:
        >>> start, end = get_mtd_date_range(df, 'R2025', 3)
        >>> print(start, end)
        2025-03-01 2025-03-15  # If data only goes to March 15
    """
    try:
        # Verify column exists
        if "report_day" not in df.columns:
            logger.error("Column 'report_day' not found in DataFrame")
            logger.info(f"Available columns: {df.columns.tolist()}")
            raise KeyError("Column 'report_day' is required but not found")

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df["report_day"]):
            logger.info("Converting report_day to datetime")
            df["report_day"] = pd.to_datetime(df["report_day"])

        # Get data for specific month
        month_data = df[
            (df["report_year"] == year)
            & (df["report_month"] == f"M{month:02d}")
        ]

        if month_data.empty:
            logger.warning(f"No data found for year {year}, month {month}")
            return None, None

        max_date = month_data["report_day"].max()
        start_date = max_date.replace(day=1)

        return start_date, max_date

    except Exception as e:
        logger.error(f"Error in get_mtd_date_range: {str(e)}")
        logger.error(f"Year: {year}, Month: {month}")
        raise


def get_previous_report_year(report_year: str, df: pd.DataFrame) -> Optional[str]:
    """
    Get the previous year's report_year string if it exists in the data.

    Input:
        report_year (str): Current year in R20XX format
        df (pd.DataFrame): DataFrame with report_year column

    Output:
        Optional[str]: Previous year in R20XX format, or None if not in data

    Example:
        >>> get_previous_report_year('R2026', df)
        'R2025'  # If R2025 exists in df
    """
    try:
        year_num = int(report_year.replace("R", ""))
        prev_year = f"R{year_num - 1}"
    except Exception:
        return None

    if prev_year in df["report_year"].unique():
        return prev_year

    return None
