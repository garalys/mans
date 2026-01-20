"""
OP2 Impact Calculators

Functions for calculating carrier, demand, tech, and market rate impacts
for OP2 benchmarking.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from ..config.settings import (
    COUNTRY_COEFFICIENTS_VOLUME,
    COUNTRY_COEFFICIENTS_LCR,
    get_tech_savings_rate,
)
from ..config.logging_config import logger


def calculate_op2_carrier_demand_impacts(
    actual_loads: float,
    actual_carriers: float,
    op2_loads: float,
    op2_carriers: float,
    base_cpkm: float,
    country: str,
    df_carrier: pd.DataFrame,
    compare_year: int,
    report_week: str = None,
    report_month: str = None,
) -> Tuple[float, float]:
    """
    Calculate carrier and demand impacts for OP2 weekly bridge.

    OP2 side represents EXPECTED values, actual side is ACTUAL performance.

    Input Parameters:
        | Parameter        | Type   | Description                           |
        |------------------|--------|---------------------------------------|
        | actual_loads     | float  | Actual number of loads                |
        | actual_carriers  | float  | Actual number of carriers             |
        | op2_loads        | float  | OP2 planned loads                     |
        | op2_carriers     | float  | OP2 planned carriers                  |
        | base_cpkm        | float  | Base cost per kilometer               |
        | country          | string | Country code for coefficient lookup   |
        | df_carrier       | DF     | Carrier percentage adjustments        |
        | compare_year     | int    | Year for lookup (e.g., 2025)          |
        | report_week      | string | Week identifier (optional)            |
        | report_month     | string | Month identifier (optional)           |

    Input Table - df_carrier:
        | Column     | Type   | Description                    |
        |------------|--------|--------------------------------|
        | year       | string | Year in R20XX format           |
        | period     | string | Period (W01, M01, etc.)        |
        | country    | string | Country code                   |
        | percentage | float  | Carrier scaling percentage     |

    Output:
        Tuple[float, float]: (carrier_impact, demand_impact) in $/km

    Formula:
        Uses polynomial LCR model where OP2 is baseline:
        - actual_lcr = actual_loads / actual_carriers
        - op2_lcr = op2_loads / op2_carriers
        - expected_lcr = actual_loads / expected_carriers
        - carrier_impact_score = actual_lcr - expected_lcr
        - demand_impact_score = expected_lcr - op2_lcr
    """
    if (
        actual_loads <= 0
        or actual_carriers <= 0
        or op2_loads <= 0
        or op2_carriers <= 0
        or base_cpkm is None
    ):
        return 0.0, 0.0

    # Get volume coefficients
    slope = COUNTRY_COEFFICIENTS_VOLUME.get(
        f"{country}_SLOPE", COUNTRY_COEFFICIENTS_VOLUME["EU_SLOPE"]
    )
    intercept = COUNTRY_COEFFICIENTS_VOLUME.get(
        f"{country}_INTERCEPT", COUNTRY_COEFFICIENTS_VOLUME["EU_INTERCEPT"]
    )

    # Get LCR coefficients
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
    period = report_week or report_month
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

    # Calculate expected executing carriers with percentage adjustment
    actual_expected_carriers = (actual_loads * slope + intercept) * (1 + percentage)

    # Calculate LCRs
    actual_lcr = actual_loads / actual_carriers
    op2_lcr = op2_loads / op2_carriers
    actual_expected_lcr = actual_loads / actual_expected_carriers

    # Decompose impacts (OP2 is baseline)
    carrier_impact_score = actual_lcr - actual_expected_lcr
    demand_impact_score = actual_expected_lcr - op2_lcr

    # Calculate carrier impact using polynomial model
    carrier_impact = (
        (
            intercept_lcr
            + slope_lcr * (carrier_impact_score + op2_lcr)
            + poly_lcr * ((carrier_impact_score + op2_lcr) ** 2)
        )
        / (intercept_lcr + slope_lcr * op2_lcr + poly_lcr * (op2_lcr ** 2))
        - 1
    ) * base_cpkm

    # Calculate demand impact using polynomial model
    demand_impact = (
        (
            intercept_lcr
            + slope_lcr * (demand_impact_score + op2_lcr)
            + poly_lcr * ((demand_impact_score + op2_lcr) ** 2)
        )
        / (intercept_lcr + slope_lcr * op2_lcr + poly_lcr * (op2_lcr ** 2))
        - 1
    ) * base_cpkm

    return carrier_impact, demand_impact


def calculate_op2_tech_impact(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    by_business: bool = False,
) -> pd.DataFrame:
    """
    Calculate OP2 tech impact based on actual volumes and tech rates.

    Computes the delta between actual tech savings and OP2 planned savings.

    Input Table - df (actual data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_week      | string | Week in WXX format             |
        | orig_country     | string | Origin country code            |
        | dest_country     | string | Destination country code       |
        | business         | string | Business unit                  |
        | distance_band    | string | Distance category              |
        | total_cost_usd   | float  | Actual cost in USD             |

    Input Table - df_op2 (OP2 data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | Bridge type      | string | Must be 'weekly'               |
        | Tech Initiative  | float  | OP2 tech impact rate           |
        | Distance         | float  | OP2 distance                   |
        | + dimensional columns                                       |

    Output Table:
        | Column                 | Type   | Description                |
        |------------------------|--------|----------------------------|
        | report_year            | string | Year in R20XX format       |
        | report_week            | string | Week in WXX format         |
        | orig_country           | string | Origin country code        |
        | business (if by_business) | string | Business unit           |
        | op2_tech_impact_value  | float  | Tech impact delta in USD   |
    """
    # Prepare OP2 weekly data
    op2 = df_op2[df_op2["Bridge type"] == "weekly"].copy()

    op2 = op2.rename(columns={
        "Report Year": "report_year",
        "Week": "report_week",
        "Orig_EU5": "orig_country",
        "Dest_EU5": "dest_country",
        "Business Flow": "business",
        "Distance Band": "distance_band",
        "Distance": "op2_distance",
        "Tech Initiative": "op2_tech_impact",
    })

    logger.info("OP2 TECH INITIATIVE STATS")
    logger.info(op2["op2_tech_impact"].describe())

    # Normalize types
    for col in ["report_year", "report_week", "orig_country", "dest_country", "business", "distance_band"]:
        op2[col] = op2[col].astype(str)
    op2["business"] = op2["business"].str.upper()

    # Calculate OP2 tech impact value
    op2["op2_tech_impact_value"] = op2["op2_distance"] * op2["op2_tech_impact"]

    # Prepare actual data
    actual = df.groupby(
        ["report_year", "report_week", "orig_country", "dest_country", "business", "distance_band"],
        as_index=False,
    ).agg(actual_cost=("total_cost_usd", "sum"))

    # Normalize types
    for col in ["report_year", "report_week", "orig_country", "dest_country", "business", "distance_band"]:
        actual[col] = actual[col].astype(str)
    actual["business"] = actual["business"].str.upper()
    actual["distance_band"] = (
        actual["distance_band"]
        .str.strip()
        .str.replace(r"^\d+\.", "", regex=True)
    )

    # Calculate tech rate for actual data
    def _get_rate(row):
        week_num = int(row["report_week"].replace("W", ""))
        year_num = int(row["report_year"].replace("R", ""))
        return get_tech_savings_rate(year_num, week_num)

    actual["tech_rate"] = actual.apply(_get_rate, axis=1)
    actual["TY_tech_impact"] = actual["actual_cost"] * actual["tech_rate"]

    # Join OP2 tech impact
    merged = actual.merge(
        op2[["report_year", "report_week", "orig_country", "dest_country", "business", "distance_band", "op2_tech_impact_value"]],
        on=["report_year", "report_week", "orig_country", "dest_country", "business", "distance_band"],
        how="inner",
    )

    # Calculate delta
    merged["tech_impact_delta"] = merged["TY_tech_impact"] - merged["op2_tech_impact_value"]

    # Aggregate to output grain
    if by_business:
        out = merged.groupby(
            ["report_year", "report_week", "orig_country", "business"],
            as_index=False,
        ).agg(op2_tech_impact_value=("tech_impact_delta", "sum"))
        return out[["report_year", "report_week", "orig_country", "business", "op2_tech_impact_value"]]
    else:
        out = merged.groupby(
            ["report_year", "report_week", "orig_country"],
            as_index=False,
        ).agg(op2_tech_impact_value=("tech_impact_delta", "sum"))
        return out[["report_year", "report_week", "orig_country", "op2_tech_impact_value"]]


def calculate_op2_market_rate_impact(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
    by_business: bool = False,
) -> pd.DataFrame:
    """
    Calculate OP2 market rate impact as delta between YoY actual and OP2 planned.

    Formula: market_impact_delta = YoY_market_rate_impact - OP2_market_impact

    The YoY market rate impact is looked up from the already-calculated bridge.
    The OP2 market impact is calculated using OP2 rate multipliers.

    Input Table - df (actual data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_week      | string | Week in WXX format             |
        | orig_country     | string | Origin country code            |
        | dest_country     | string | Destination country code       |
        | business         | string | Business unit                  |
        | distance_band    | string | Distance category              |
        | distance_for_cpkm| float  | Distance in kilometers         |
        | total_cost_usd   | float  | Cost in USD                    |

    Input Table - df_op2 (OP2 data):
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | Bridge type     | string | Must be 'weekly'                |
        | CpKM            | float  | OP2 cost per kilometer          |
        | Market Rate     | float  | OP2 market rate value           |
        | + dimensional columns                                       |

    Input Table - final_bridge_df (YoY bridge with market rate impact):
        | Column              | Type   | Description                   |
        |---------------------|--------|-------------------------------|
        | bridge_type         | string | Filter for 'YoY'              |
        | report_year         | string | Year in R20XX format          |
        | report_week         | string | Week in WXX format            |
        | orig_country        | string | Origin country code           |
        | business            | string | Business unit                 |
        | market_rate_impact  | float  | YoY market rate impact ($/km) |
        | compare_distance_km | float  | Compare period distance       |

    Output Table:
        | Column            | Type   | Description                          |
        |-------------------|--------|--------------------------------------|
        | report_year       | string | Year in R20XX format                 |
        | report_week       | string | Week in WXX format                   |
        | orig_country      | string | Origin country code                  |
        | business (if by_business) | string | Business unit                 |
        | op2_market_impact | float  | YoY - OP2 market impact delta (USD)  |
    """
    from .op2_helpers import get_market_rate_impact_from_yoy

    # Step 1: Get YoY market rate impact (already in absolute USD from helper)
    yoy_market_rate = get_market_rate_impact_from_yoy(final_bridge_df)

    # Step 2: Calculate OP2 market impact using the existing formula
    # Prepare OP2 weekly data
    op2 = df_op2[df_op2["Bridge type"] == "weekly"].copy()

    op2 = op2.rename(columns={
        "Report Year": "report_year",
        "Week": "report_week",
        "Orig_EU5": "orig_country",
        "Dest_EU5": "dest_country",
        "Business Flow": "business",
        "Distance Band": "distance_band",
        "CpKM": "op2_cpkm",
        "Market Rate": "op2_market_rate",
    })

    dim_cols = ["report_year", "report_week", "orig_country", "dest_country", "business", "distance_band"]

    for col in dim_cols:
        op2[col] = op2[col].astype(str)
    op2["business"] = op2["business"].str.upper()

    # Calculate market rate multiplier
    op2["mr_by_cpkm"] = np.where(
        op2["op2_cpkm"] > 0,
        op2["op2_market_rate"] / op2["op2_cpkm"],
        np.nan,
    )
    op2 = op2[dim_cols + ["mr_by_cpkm"]]

    # Prepare actual data
    actual = df.copy()
    for col in dim_cols:
        actual[col] = actual[col].astype(str)
    actual["business"] = actual["business"].str.upper()
    actual["distance_band"] = (
        actual["distance_band"]
        .str.strip()
        .str.replace(r"^\d+\.", "", regex=True)
    )

    # Aggregate actual data
    group_cols = ["report_year", "report_week", "orig_country", "dest_country", "distance_band", "business"]
    actual_agg = actual.groupby(group_cols, as_index=False).agg(
        distance=("distance_for_cpkm", "sum"),
        cost=("total_cost_usd", "sum"),
    )

    actual_agg["cpkm"] = np.where(
        actual_agg["distance"] > 0,
        actual_agg["cost"] / actual_agg["distance"],
        0.0,
    )

    # Build LY via self-join
    actual_agg["year_num"] = actual_agg["report_year"].str.replace("R", "").astype(int)
    actual_agg["ly_report_year"] = "R" + (actual_agg["year_num"] - 1).astype(str)

    ly = actual_agg[group_cols + ["cpkm", "distance", "cost"]].rename(columns={
        "cpkm": "ly_cpkm",
        "distance": "ly_distance",
        "cost": "ly_cost",
    })

    merged = actual_agg.merge(
        ly,
        left_on=["ly_report_year", "report_week", "orig_country", "dest_country", "distance_band", "business"],
        right_on=["report_year", "report_week", "orig_country", "dest_country", "distance_band", "business"],
        how="left",
        suffixes=("", "_drop"),
    )
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_drop")])

    # Merge OP2 market rate signal
    merged = merged.merge(op2, on=dim_cols, how="left")

    # Apply OP2 market rate logic (absolute USD)
    merged["op2_market_impact_raw"] = np.where(
        merged["mr_by_cpkm"].notna(),
        np.where(
            merged["ly_distance"].fillna(0) == 0,
            merged["cost"] * merged["mr_by_cpkm"] - merged["cost"],
            merged["ly_cpkm"] * merged["distance"] * merged["mr_by_cpkm"],
        ),
        0.0,
    )
    merged["op2_market_impact_raw"] = merged["op2_market_impact_raw"].fillna(0.0)

    # Step 3: Aggregate OP2 market impact to output grain
    if by_business:
        op2_agg = merged.groupby(
            ["report_year", "report_week", "orig_country", "business"],
            as_index=False,
        ).agg(op2_market_impact_raw=("op2_market_impact_raw", "sum"))

        # Step 4: Join with YoY market rate and calculate delta
        out = op2_agg.merge(
            yoy_market_rate[["report_year", "report_week", "orig_country", "business", "market_rate_impact"]],
            on=["report_year", "report_week", "orig_country", "business"],
            how="left",
        )

        # Delta = YoY actual - OP2 planned
        out["op2_market_impact"] = out["market_rate_impact"].fillna(0) - out["op2_market_impact_raw"]
        return out[["report_year", "report_week", "orig_country", "business", "op2_market_impact"]]
    else:
        op2_agg = merged.groupby(
            ["report_year", "report_week", "orig_country"],
            as_index=False,
        ).agg(op2_market_impact_raw=("op2_market_impact_raw", "sum"))

        # Aggregate YoY market rate to country level
        yoy_country = yoy_market_rate.groupby(
            ["report_year", "report_week", "orig_country"],
            as_index=False,
        ).agg(market_rate_impact=("market_rate_impact", "sum"))

        # Step 4: Join with YoY market rate and calculate delta
        out = op2_agg.merge(
            yoy_country,
            on=["report_year", "report_week", "orig_country"],
            how="left",
        )

        # Delta = YoY actual - OP2 planned
        out["op2_market_impact"] = out["market_rate_impact"].fillna(0) - out["op2_market_impact_raw"]
        return out[["report_year", "report_week", "orig_country", "op2_market_impact"]]


def calculate_op2_tech_impact_monthly(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    by_business: bool = False,
) -> pd.DataFrame:
    """
    Calculate OP2 tech impact for monthly bridge based on actual volumes and tech rates.

    Computes the delta between actual tech savings and OP2 planned savings.

    Input Table - df (actual data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_month     | string | Month in MXX format            |
        | orig_country     | string | Origin country code            |
        | dest_country     | string | Destination country code       |
        | business         | string | Business unit                  |
        | distance_band    | string | Distance category              |
        | total_cost_usd   | float  | Actual cost in USD             |

    Input Table - df_op2 (OP2 data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | Bridge type      | string | Must be 'monthly'              |
        | Tech Initiative  | float  | OP2 tech impact rate           |
        | Distance         | float  | OP2 distance                   |
        | + dimensional columns                                       |

    Output Table:
        | Column                 | Type   | Description                |
        |------------------------|--------|----------------------------|
        | report_year            | string | Year in R20XX format       |
        | report_month           | string | Month in MXX format        |
        | orig_country           | string | Origin country code        |
        | business (if by_business) | string | Business unit           |
        | op2_tech_impact_value  | float  | Tech impact delta in USD   |
    """
    # Prepare OP2 monthly data
    op2 = df_op2[df_op2["Bridge type"] == "monthly"].copy()

    op2 = op2.rename(columns={
        "Report Year": "report_year",
        "Report Month": "report_month",
        "Orig_EU5": "orig_country",
        "Dest_EU5": "dest_country",
        "Business Flow": "business",
        "Distance Band": "distance_band",
        "Distance": "op2_distance",
        "Tech Initiative": "op2_tech_impact",
    })

    logger.info("OP2 MONTHLY TECH INITIATIVE STATS")
    if "op2_tech_impact" in op2.columns:
        logger.info(op2["op2_tech_impact"].describe())

    # Normalize types
    for col in ["report_year", "report_month", "orig_country", "dest_country", "business", "distance_band"]:
        if col in op2.columns:
            op2[col] = op2[col].astype(str)
    op2["business"] = op2["business"].str.upper()

    # Calculate OP2 tech impact value
    op2["op2_tech_impact_value"] = op2["op2_distance"] * op2["op2_tech_impact"].fillna(0)

    # Prepare actual data - keep report_week to lookup correct tech rate for each week
    actual = df.groupby(
        ["report_year", "report_month", "report_week", "orig_country", "dest_country", "business", "distance_band"],
        as_index=False,
    ).agg(actual_cost=("total_cost_usd", "sum"))

    # Normalize types
    for col in ["report_year", "report_month", "report_week", "orig_country", "dest_country", "business", "distance_band"]:
        actual[col] = actual[col].astype(str)
    actual["business"] = actual["business"].str.upper()
    actual["distance_band"] = (
        actual["distance_band"]
        .str.strip()
        .str.replace(r"^\d+\.", "", regex=True)
    )

    # Calculate tech rate using actual week from data
    def _get_rate(row):
        week_num = int(row["report_week"].replace("W", ""))
        year_num = int(row["report_year"].replace("R", ""))
        return get_tech_savings_rate(year_num, week_num)

    actual["tech_rate"] = actual.apply(_get_rate, axis=1)
    actual["TY_tech_impact"] = actual["actual_cost"] * actual["tech_rate"]

    # Aggregate to month level after calculating week-level tech impacts
    actual = actual.groupby(
        ["report_year", "report_month", "orig_country", "dest_country", "business", "distance_band"],
        as_index=False,
    ).agg(
        actual_cost=("actual_cost", "sum"),
        TY_tech_impact=("TY_tech_impact", "sum"),
    )

    # Join OP2 tech impact
    merged = actual.merge(
        op2[["report_year", "report_month", "orig_country", "dest_country", "business", "distance_band", "op2_tech_impact_value"]],
        on=["report_year", "report_month", "orig_country", "dest_country", "business", "distance_band"],
        how="inner",
    )

    # Calculate delta
    merged["tech_impact_delta"] = merged["TY_tech_impact"] - merged["op2_tech_impact_value"]

    # Aggregate to output grain
    if by_business:
        out = merged.groupby(
            ["report_year", "report_month", "orig_country", "business"],
            as_index=False,
        ).agg(op2_tech_impact_value=("tech_impact_delta", "sum"))
        return out[["report_year", "report_month", "orig_country", "business", "op2_tech_impact_value"]]
    else:
        out = merged.groupby(
            ["report_year", "report_month", "orig_country"],
            as_index=False,
        ).agg(op2_tech_impact_value=("tech_impact_delta", "sum"))
        return out[["report_year", "report_month", "orig_country", "op2_tech_impact_value"]]


def calculate_op2_market_rate_impact_monthly(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
    by_business: bool = False,
) -> pd.DataFrame:
    """
    Calculate OP2 market rate impact for monthly bridge as delta between MTD actual and OP2 planned.

    Formula: market_impact_delta = MTD_market_rate_impact - OP2_market_impact

    Input Table - df (actual data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_month     | string | Month in MXX format            |
        | orig_country     | string | Origin country code            |
        | dest_country     | string | Destination country code       |
        | business         | string | Business unit                  |
        | distance_band    | string | Distance category              |
        | distance_for_cpkm| float  | Distance in kilometers         |
        | total_cost_usd   | float  | Cost in USD                    |

    Input Table - df_op2 (OP2 data):
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | Bridge type     | string | Must be 'monthly'               |
        | CpKM            | float  | OP2 cost per kilometer          |
        | Market Rate     | float  | OP2 market rate value           |
        | + dimensional columns                                       |

    Input Table - final_bridge_df (MTD bridge with market rate impact):
        | Column              | Type   | Description                   |
        |---------------------|--------|-------------------------------|
        | bridge_type         | string | Filter for 'MTD'              |
        | report_year         | string | Year in R20XX format          |
        | report_month        | string | Month in MXX format           |
        | orig_country        | string | Origin country code           |
        | business            | string | Business unit                 |
        | market_rate_impact  | float  | MTD market rate impact ($/km) |
        | m2_distance_km      | float  | Compare period distance       |

    Output Table:
        | Column            | Type   | Description                          |
        |-------------------|--------|--------------------------------------|
        | report_year       | string | Year in R20XX format                 |
        | report_month      | string | Month in MXX format                  |
        | orig_country      | string | Origin country code                  |
        | business (if by_business) | string | Business unit                 |
        | op2_market_impact | float  | MTD - OP2 market impact delta (USD)  |
    """
    from .op2_helpers import get_market_rate_impact_from_mtd

    # Step 1: Get MTD market rate impact
    mtd_market_rate = get_market_rate_impact_from_mtd(final_bridge_df)

    # Step 2: Calculate OP2 market impact using the existing formula
    op2 = df_op2[df_op2["Bridge type"] == "monthly"].copy()

    op2 = op2.rename(columns={
        "Report Year": "report_year",
        "Report Month": "report_month",
        "Orig_EU5": "orig_country",
        "Dest_EU5": "dest_country",
        "Business Flow": "business",
        "Distance Band": "distance_band",
        "CpKM": "op2_cpkm",
        "Market Rate": "op2_market_rate",
    })

    dim_cols = ["report_year", "report_month", "orig_country", "dest_country", "business", "distance_band"]

    for col in dim_cols:
        if col in op2.columns:
            op2[col] = op2[col].astype(str)
    op2["business"] = op2["business"].str.upper()

    # Calculate market rate multiplier
    op2["mr_by_cpkm"] = np.where(
        op2["op2_cpkm"] > 0,
        op2["op2_market_rate"].fillna(0) / op2["op2_cpkm"],
        np.nan,
    )
    op2 = op2[dim_cols + ["mr_by_cpkm"]]

    # Prepare actual data
    actual = df.copy()
    for col in dim_cols:
        actual[col] = actual[col].astype(str)
    actual["business"] = actual["business"].str.upper()
    actual["distance_band"] = (
        actual["distance_band"]
        .str.strip()
        .str.replace(r"^\d+\.", "", regex=True)
    )

    # Aggregate actual data
    group_cols = ["report_year", "report_month", "orig_country", "dest_country", "distance_band", "business"]
    actual_agg = actual.groupby(group_cols, as_index=False).agg(
        distance=("distance_for_cpkm", "sum"),
        cost=("total_cost_usd", "sum"),
    )

    actual_agg["cpkm"] = np.where(
        actual_agg["distance"] > 0,
        actual_agg["cost"] / actual_agg["distance"],
        0.0,
    )

    # Build LY via self-join
    actual_agg["year_num"] = actual_agg["report_year"].str.replace("R", "").astype(int)
    actual_agg["ly_report_year"] = "R" + (actual_agg["year_num"] - 1).astype(str)

    ly = actual_agg[group_cols + ["cpkm", "distance", "cost"]].rename(columns={
        "cpkm": "ly_cpkm",
        "distance": "ly_distance",
        "cost": "ly_cost",
    })

    merged = actual_agg.merge(
        ly,
        left_on=["ly_report_year", "report_month", "orig_country", "dest_country", "distance_band", "business"],
        right_on=["report_year", "report_month", "orig_country", "dest_country", "distance_band", "business"],
        how="left",
        suffixes=("", "_drop"),
    )
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_drop")])

    # Merge OP2 market rate signal
    merged = merged.merge(op2, on=dim_cols, how="left")

    # Apply OP2 market rate logic (absolute USD)
    merged["op2_market_impact_raw"] = np.where(
        merged["mr_by_cpkm"].notna(),
        np.where(
            merged["ly_distance"].fillna(0) == 0,
            merged["cost"] * merged["mr_by_cpkm"] - merged["cost"],
            merged["ly_cpkm"] * merged["distance"] * merged["mr_by_cpkm"],
        ),
        0.0,
    )
    merged["op2_market_impact_raw"] = merged["op2_market_impact_raw"].fillna(0.0)

    # Step 3: Aggregate OP2 market impact to output grain
    if by_business:
        op2_agg = merged.groupby(
            ["report_year", "report_month", "orig_country", "business"],
            as_index=False,
        ).agg(op2_market_impact_raw=("op2_market_impact_raw", "sum"))

        # Step 4: Join with MTD market rate and calculate delta
        out = op2_agg.merge(
            mtd_market_rate[["report_year", "report_month", "orig_country", "business", "market_rate_impact"]],
            on=["report_year", "report_month", "orig_country", "business"],
            how="left",
        )

        out["op2_market_impact"] = out["market_rate_impact"].fillna(0) - out["op2_market_impact_raw"]
        return out[["report_year", "report_month", "orig_country", "business", "op2_market_impact"]]
    else:
        op2_agg = merged.groupby(
            ["report_year", "report_month", "orig_country"],
            as_index=False,
        ).agg(op2_market_impact_raw=("op2_market_impact_raw", "sum"))

        # Aggregate MTD market rate to country level
        mtd_country = mtd_market_rate.groupby(
            ["report_year", "report_month", "orig_country"],
            as_index=False,
        ).agg(market_rate_impact=("market_rate_impact", "sum"))

        out = op2_agg.merge(
            mtd_country,
            on=["report_year", "report_month", "orig_country"],
            how="left",
        )

        out["op2_market_impact"] = out["market_rate_impact"].fillna(0) - out["op2_market_impact_raw"]
        return out[["report_year", "report_month", "orig_country", "op2_market_impact"]]
