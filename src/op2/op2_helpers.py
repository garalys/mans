"""
OP2 Helper Functions

Utility functions for OP2 benchmarking operations.
"""

import pandas as pd

from ..config.logging_config import logger


def get_set_impact_for_op2(final_bridge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract SET impact from YoY bridge for use in OP2 benchmarks.

    SET impact is sourced from the YoY bridge since it represents the
    change in SET pricing between years, which is independent of OP2.

    Input Table - final_bridge_df:
        | Column       | Type   | Description                         |
        |--------------|--------|-------------------------------------|
        | bridge_type  | string | Must be 'YoY' for filtering         |
        | report_year  | string | Year in R20XX format                |
        | report_week  | string | Week in WXX format                  |
        | orig_country | string | Origin country code                 |
        | business     | string | Business unit                       |
        | set_impact   | float  | SET pricing impact ($/km)           |

    Output Table:
        | Column       | Type   | Description                         |
        |--------------|--------|-------------------------------------|
        | report_year  | string | Year in R20XX format                |
        | report_week  | string | Week in WXX format                  |
        | orig_country | string | Origin country code                 |
        | business     | string | Business unit                       |
        | set_impact   | float  | SET pricing impact ($/km)           |

    Note:
        Only returns rows where set_impact is not null.
    """
    logger.info("Preparing YoY SET lookup for OP2 weekly bridge...")

    yoy_set_lookup = final_bridge_df[
        (final_bridge_df["bridge_type"] == "YoY")
        & (final_bridge_df["set_impact"].notna())
    ][["report_year", "report_week", "orig_country", "business", "set_impact"]]

    return yoy_set_lookup


def get_market_rate_impact_from_yoy(final_bridge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract market rate impact from YoY bridge for reference.

    Returns YoY market rate impact scaled to millions.

    Input Table - final_bridge_df:
        | Column            | Type   | Description                      |
        |-------------------|--------|----------------------------------|
        | bridge_type       | string | Must be 'YoY' for filtering      |
        | report_year       | string | Year in R20XX format             |
        | report_week       | string | Week in WXX format               |
        | orig_country      | string | Origin country code              |
        | business          | string | Business unit                    |
        | market_rate_impact| float  | Market rate impact ($/km)        |

    Output Table:
        Same as input selection with market_rate_impact scaled by 1,000,000
    """
    logger.info("Preparing YoY market rate lookup...")

    yoy_market_rate_lookup = final_bridge_df[
        (final_bridge_df["bridge_type"] == "YoY")
        & (final_bridge_df["set_impact"].notna())
    ][["report_year", "report_week", "orig_country", "business", "market_rate_impact","compare_distance_km"]]

    yoy_market_rate_lookup = yoy_market_rate_lookup.copy()
    yoy_market_rate_lookup["market_rate_impact"] = (
        yoy_market_rate_lookup["market_rate_impact"] * yoy_market_rate_lookup["compare_distance_km"]
    )

    return yoy_market_rate_lookup
