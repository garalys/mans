"""
OP2 EU-Level Aggregation Calculator

Creates EU-level OP2 benchmark bridges by aggregating country-level results.
"""

import pandas as pd
import numpy as np

from ..config.logging_config import logger
from ..config.settings import (
    COUNTRY_COEFFICIENTS_VOLUME,
    COUNTRY_COEFFICIENTS_LCR,
    get_tech_savings_rate,
)
from .op2_data_extractor import extract_op2_weekly_base_cpkm
from .op2_normalizer import compute_op2_normalized_cpkm_weekly
from .op2_impacts import calculate_op2_tech_impact, calculate_op2_market_rate_impact
from .op2_helpers import get_set_impact_for_op2


def create_op2_eu_weekly_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 weekly benchmark bridge at EU-total level.

    Aggregates country-level OP2 metrics to EU level. Uses OP2 weekly_agg
    for EU base figures (Distance, Loads, Cost, CpKM, Active Carriers, LCR).
    Other metrics are aggregated from country-level calculations.

    Input Table - df (actual data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_week      | string | Week in WXX format             |
        | orig_country     | string | Origin country code            |
        | total_cost_usd   | float  | Cost in USD                    |
        | distance_for_cpkm| float  | Distance in kilometers         |
        | executed_loads   | int    | Number of loads                |

    Input Table - df_op2:
        OP2 benchmark data with weekly_agg for EU base figures

    Input Table - df_carrier:
        Carrier scaling percentages

    Input Table - final_bridge_df:
        Completed YoY bridge for carrier and SET impact lookup

    Output Table:
        | Column                    | Type   | Description                    |
        |---------------------------|--------|--------------------------------|
        | report_year               | string | Year in R20XX format           |
        | report_week               | string | Week in WXX format             |
        | orig_country              | string | Always 'EU'                    |
        | business                  | string | Always 'Total'                 |
        | bridge_type               | string | Always 'op2_weekly'            |
        | + all standard OP2 bridge columns                                  |
    """
    logger.info("Creating OP2 EU-total weekly bridge...")

    # Get OP2 EU base metrics from weekly_agg
    op2_base_all = extract_op2_weekly_base_cpkm(df_op2)
    op2_eu_base = op2_base_all[op2_base_all["orig_country"] == "EU"].copy()

    if op2_eu_base.empty:
        logger.warning("No EU data found in OP2 weekly_agg. Skipping EU total bridge.")
        return pd.DataFrame()

    # Aggregate actual data across all EU5 countries
    eu_countries = ["DE", "ES", "FR", "IT", "UK"]
    actual_eu = (
        df[
            (df["report_week"].notna())
            & (df["orig_country"].isin(eu_countries))
        ]
        .groupby(["report_year", "report_week"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance_km=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    actual_eu["compare_cpkm"] = actual_eu["actual_cost"] / actual_eu["actual_distance_km"]

    # Get actual carriers from YoY bridge (aggregate EU5 countries)
    yoy_carriers = _get_eu_carriers_from_yoy_bridge(final_bridge_df, business="Total")
    actual_eu = actual_eu.merge(
        yoy_carriers,
        on=["report_year", "report_week"],
        how="left",
    )
    actual_eu["actual_carriers"] = actual_eu["actual_carriers"].fillna(0)

    # Aggregate normalized CPKM from country-level calculations
    country_norm = compute_op2_normalized_cpkm_weekly(df, df_op2, by_business=False)
    country_norm = country_norm[country_norm["orig_country"].isin(eu_countries)]
    eu_norm = country_norm.groupby(
        ["report_year", "report_week"],
        as_index=False,
    ).agg(
        op2_normalized_cost=("op2_normalized_cost", "sum"),
    )
    # Join with actual_eu to get distance for CPKM calculation
    eu_norm = eu_norm.merge(
        actual_eu[["report_year", "report_week", "actual_distance_km"]],
        on=["report_year", "report_week"],
        how="left",
    )
    eu_norm["op2_normalized_cpkm"] = eu_norm["op2_normalized_cost"] / eu_norm["actual_distance_km"]

    # Aggregate tech impact from country level
    country_tech = calculate_op2_tech_impact(df, df_op2, by_business=False)
    country_tech = country_tech[country_tech["orig_country"].isin(eu_countries)]
    eu_tech = country_tech.groupby(
        ["report_year", "report_week"],
        as_index=False,
    ).agg(op2_tech_impact_value=("op2_tech_impact_value", "sum"))

    # Aggregate market rate impact from country level
    country_market = calculate_op2_market_rate_impact(df, df_op2, final_bridge_df, by_business=False)
    country_market = country_market[country_market["orig_country"].isin(eu_countries)]
    eu_market = country_market.groupby(
        ["report_year", "report_week"],
        as_index=False,
    ).agg(op2_market_impact=("op2_market_impact", "sum"))

    # Merge all components
    bridge = (
        actual_eu
        .merge(op2_eu_base, on=["report_year", "report_week"], how="left")
        .merge(eu_norm[["report_year", "report_week", "op2_normalized_cpkm"]], on=["report_year", "report_week"], how="left")
        .merge(eu_tech, on=["report_year", "report_week"], how="left")
        .merge(eu_market, on=["report_year", "report_week"], how="left")
    )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_weekly"
    bridge["business"] = "Total"
    bridge["orig_country"] = "EU"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge["op2_normalized_cpkm"]

    # Get SET impact from YoY bridge for EU Total
    yoy_set_impact = get_set_impact_for_op2(final_bridge_df)
    yoy_set_eu = yoy_set_impact[
        (yoy_set_impact["orig_country"] == "EU")
        & (yoy_set_impact["business"] == "Total")
    ]
    bridge = bridge.merge(
        yoy_set_eu[["report_year", "report_week", "set_impact"]],
        on=["report_year", "report_week"],
        how="left",
    )

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100

    bridge["distance_variance_km"] = bridge["actual_distance_km"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100

    # Normalized metrics
    bridge["compare_cpkm_vs_normalised_op2_cpkm"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
    bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]

    bridge["tech_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_tech_impact_value"] / bridge["actual_distance_km"],
        0,
    )

    bridge["market_rate_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_market_impact"] / bridge["actual_distance_km"],
        0,
    )

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_week"] + "_OP2"
    bridge["w2_distance_km"] = bridge["actual_distance_km"]
    bridge["benchmark_gap"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate carrier and demand impacts using EU coefficients
    bridge[["carrier_impact", "demand_impact"]] = bridge.apply(
        lambda r: _calculate_eu_op2_carrier_demand_impacts(
            actual_loads=r["actual_loads"],
            actual_carriers=r["actual_carriers"],
            op2_loads=r["op2_base_loads"],
            op2_carriers=r["op2_carriers"],
            base_cpkm=r["op2_base_cpkm"],
            df_carrier=df_carrier,
            compare_year=int(r["report_year"].replace("R", "")),
            report_week=r["report_week"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    logger.info(f"Created {len(bridge)} OP2 EU-total weekly bridge rows")
    return bridge


def create_op2_eu_weekly_business_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 weekly benchmark bridge at EU x business level.

    OP2 doesn't have business-level splits for EU in the source data, so all
    metrics are aggregated from country x business level calculations.

    Input Tables:
        Same as create_op2_eu_weekly_bridge

    Output Table:
        Same structure as create_op2_eu_weekly_bridge but:
        - business column contains actual business values (not 'Total')
        - Granularity is EU x week x business
    """
    logger.info("Creating OP2 EU x business weekly bridge...")

    eu_countries = ["DE", "ES", "FR", "IT", "UK"]

    # Filter to EU5 countries with valid weeks
    df_eu = df[
        (df["report_week"].notna())
        & (df["orig_country"].isin(eu_countries))
    ].copy()
    # Normalize business to uppercase for consistent matching
    df_eu["business"] = df_eu["business"].str.upper()

    # Aggregate actual data by business across all EU5 countries
    actual_eu = (
        df_eu
        .groupby(["report_year", "report_week", "business"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance_km=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    actual_eu["compare_cpkm"] = actual_eu["actual_cost"] / actual_eu["actual_distance_km"]

    # Calculate actual carriers directly from source data (count unique carriers across EU countries by business)
    actual_eu["actual_carriers"] = actual_eu.apply(
        lambda r: df_eu[
            (df_eu["report_year"] == r["report_year"])
            & (df_eu["report_week"] == r["report_week"])
            & (df_eu["business"] == r["business"])
            & (df_eu["executed_loads"] > 0)
        ]["vehicle_carrier"].nunique(),
        axis=1,
    )

    # Aggregate OP2 base metrics from country x business level
    # (since EU doesn't have business splits in OP2 data)
    country_business_norm = compute_op2_normalized_cpkm_weekly(df, df_op2, by_business=True)
    country_business_norm = country_business_norm[country_business_norm["orig_country"].isin(eu_countries)]
    country_business_norm["business"] = country_business_norm["business"].str.upper()

    # Also aggregate base metrics from country x business
    from .op2_data_extractor import extract_op2_weekly_base_by_business
    country_business_base = extract_op2_weekly_base_by_business(df_op2)
    country_business_base = country_business_base[country_business_base["orig_country"].isin(eu_countries)]
    country_business_base["business"] = country_business_base["business"].str.upper()

    eu_base = country_business_base.groupby(
        ["report_year", "report_week", "business"],
        as_index=False,
    ).agg(
        op2_base_distance=("op2_base_distance", "sum"),
        op2_base_cost=("op2_base_cost", "sum"),
        op2_base_loads=("op2_base_loads", "sum"),
    )
    eu_base["op2_base_cpkm"] = eu_base["op2_base_cost"] / eu_base["op2_base_distance"]

    # Get EU-level carriers from weekly_agg
    op2_base_all = extract_op2_weekly_base_cpkm(df_op2)
    op2_eu_carriers = op2_base_all[op2_base_all["orig_country"] == "EU"][
        ["report_year", "report_week", "op2_carriers"]
    ]
    eu_base = eu_base.merge(
        op2_eu_carriers,
        on=["report_year", "report_week"],
        how="left",
    )

    # Aggregate normalized CPKM from country x business level
    eu_norm = country_business_norm.groupby(
        ["report_year", "report_week", "business"],
        as_index=False,
    ).agg(
        op2_normalized_cost=("op2_normalized_cost", "sum"),
    )
    # Join with actual_eu to get distance for CPKM calculation
    eu_norm = eu_norm.merge(
        actual_eu[["report_year", "report_week", "business", "actual_distance_km"]],
        on=["report_year", "report_week", "business"],
        how="left",
    )
    eu_norm["op2_normalized_cpkm"] = eu_norm["op2_normalized_cost"] / eu_norm["actual_distance_km"]

    # Aggregate tech impact from country x business level
    country_tech = calculate_op2_tech_impact(df, df_op2, by_business=True)
    country_tech = country_tech[country_tech["orig_country"].isin(eu_countries)]
    country_tech["business"] = country_tech["business"].str.upper()
    eu_tech = country_tech.groupby(
        ["report_year", "report_week", "business"],
        as_index=False,
    ).agg(op2_tech_impact_value=("op2_tech_impact_value", "sum"))

    # Aggregate market rate impact from country x business level
    country_market = calculate_op2_market_rate_impact(df, df_op2, final_bridge_df, by_business=True)
    country_market = country_market[country_market["orig_country"].isin(eu_countries)]
    country_market["business"] = country_market["business"].str.upper()
    eu_market = country_market.groupby(
        ["report_year", "report_week", "business"],
        as_index=False,
    ).agg(op2_market_impact=("op2_market_impact", "sum"))

    # Merge all components
    bridge = (
        actual_eu
        .merge(eu_base, on=["report_year", "report_week", "business"], how="left")
        .merge(eu_norm[["report_year", "report_week", "business", "op2_normalized_cpkm"]], on=["report_year", "report_week", "business"], how="left")
        .merge(eu_tech, on=["report_year", "report_week", "business"], how="left")
        .merge(eu_market, on=["report_year", "report_week", "business"], how="left")
    )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_weekly"
    bridge["orig_country"] = "EU"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge["op2_normalized_cpkm"]

    # Get SET impact from YoY bridge for EU by business
    yoy_set_impact = get_set_impact_for_op2(final_bridge_df)
    yoy_set_eu = yoy_set_impact[yoy_set_impact["orig_country"] == "EU"]
    bridge = bridge.merge(
        yoy_set_eu[["report_year", "report_week", "business", "set_impact"]],
        on=["report_year", "report_week", "business"],
        how="left",
    )

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = np.where(
        bridge["op2_base_loads"] > 0,
        (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100,
        0,
    )

    bridge["distance_variance_km"] = bridge["actual_distance_km"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = np.where(
        bridge["op2_base_distance"] > 0,
        (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100,
        0,
    )

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = np.where(
        bridge["op2_base_cost"] > 0,
        ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100,
        0,
    )

    # Normalized metrics
    bridge["compare_cpkm_vs_normalised_op2_cpkm"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
    bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]

    bridge["tech_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_tech_impact_value"] / bridge["actual_distance_km"],
        0,
    )

    bridge["market_rate_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_market_impact"] / bridge["actual_distance_km"],
        0,
    )

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_week"] + "_OP2"

    # Calculate carrier and demand impacts using EU coefficients
    bridge[["carrier_impact", "demand_impact"]] = bridge.apply(
        lambda r: _calculate_eu_op2_carrier_demand_impacts(
            actual_loads=r["actual_loads"],
            actual_carriers=r["actual_carriers"],
            op2_loads=r["op2_base_loads"] if pd.notna(r["op2_base_loads"]) else r["actual_loads"],
            op2_carriers=r["op2_carriers"] if pd.notna(r["op2_carriers"]) else r["actual_carriers"],
            base_cpkm=r["op2_normalized_cpkm"] if pd.notna(r["op2_normalized_cpkm"]) else r["compare_cpkm"],
            df_carrier=df_carrier,
            compare_year=int(r["report_year"].replace("R", "")),
            report_week=r["report_week"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    logger.info(f"Created {len(bridge)} OP2 EU x business weekly bridge rows")
    return bridge


def _get_eu_carriers_from_yoy_bridge(
    final_bridge_df: pd.DataFrame,
    business: str = "Total",
) -> pd.DataFrame:
    """
    Look up EU-level carrier counts from YoY bridge.

    Aggregates carrier counts from individual EU5 countries for a given business.

    Input Table - final_bridge_df:
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | bridge_type      | string | Filter for 'YoY'               |
        | orig_country     | string | Country code                   |
        | business         | string | Business unit                  |
        | report_year      | string | Compare year in R20XX format   |
        | report_week      | string | Week in WXX format             |
        | y{YYYY}_carriers | int    | Carrier count for year         |

    Output Table:
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | report_year     | string | Year in R20XX format            |
        | report_week     | string | Week in WXX format              |
        | actual_carriers | int    | Aggregated EU carrier count     |
    """
    eu_countries = ["DE", "ES", "FR", "IT", "UK"]

    # Get YoY bridge rows for EU5 countries and specified business
    yoy_rows = final_bridge_df[
        (final_bridge_df["bridge_type"] == "YoY")
        & (final_bridge_df["orig_country"].isin(eu_countries))
        & (final_bridge_df["business"] == business)
    ].copy()

    if yoy_rows.empty:
        return pd.DataFrame(columns=["report_year", "report_week", "actual_carriers"])

    # Extract compare year from bridging_value and get carrier column
    yoy_rows["compare_year"] = yoy_rows["bridging_value"].str.split("_to_").str[1]

    # Determine carrier column dynamically
    carrier_cols = [c for c in yoy_rows.columns if c.endswith("_carriers") and c.startswith("y")]

    results = []
    for (compare_year, week), group in yoy_rows.groupby(["compare_year", "report_week"]):
        # Find the carrier column for this compare year
        year_num = compare_year.replace("R", "")
        carrier_col = f"y{year_num}_carriers"

        if carrier_col in group.columns:
            total_carriers = group[carrier_col].sum()
        else:
            # Fallback: sum any available carrier column
            total_carriers = 0
            for col in carrier_cols:
                if col in group.columns:
                    total_carriers = group[col].sum()
                    break

        results.append({
            "report_year": compare_year,
            "report_week": week,
            "actual_carriers": total_carriers,
        })

    return pd.DataFrame(results)


def _get_eu_carriers_from_yoy_bridge_by_business(
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Look up EU-level carrier counts from YoY bridge by business.

    Aggregates carrier counts from individual EU5 countries for each business.

    Output Table:
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | report_year     | string | Year in R20XX format            |
        | report_week     | string | Week in WXX format              |
        | business        | string | Business unit                   |
        | actual_carriers | int    | Aggregated EU carrier count     |
    """
    eu_countries = ["DE", "ES", "FR", "IT", "UK"]

    # Get YoY bridge rows for EU5 countries (exclude Total)
    yoy_rows = final_bridge_df[
        (final_bridge_df["bridge_type"] == "YoY")
        & (final_bridge_df["orig_country"].isin(eu_countries))
        & (final_bridge_df["business"] != "Total")
    ].copy()

    if yoy_rows.empty:
        return pd.DataFrame(columns=["report_year", "report_week", "business", "actual_carriers"])

    # Extract compare year from bridging_value
    yoy_rows["compare_year"] = yoy_rows["bridging_value"].str.split("_to_").str[1]

    # Determine carrier column dynamically
    carrier_cols = [c for c in yoy_rows.columns if c.endswith("_carriers") and c.startswith("y")]

    results = []
    for (compare_year, week, business), group in yoy_rows.groupby(["compare_year", "report_week", "business"]):
        # Find the carrier column for this compare year
        year_num = compare_year.replace("R", "")
        carrier_col = f"y{year_num}_carriers"

        if carrier_col in group.columns:
            total_carriers = group[carrier_col].sum()
        else:
            total_carriers = 0
            for col in carrier_cols:
                if col in group.columns:
                    total_carriers = group[col].sum()
                    break

        results.append({
            "report_year": compare_year,
            "report_week": week,
            "business": business,
            "actual_carriers": total_carriers,
        })

    return pd.DataFrame(results)


def _calculate_eu_op2_carrier_demand_impacts(
    actual_loads: float,
    actual_carriers: float,
    op2_loads: float,
    op2_carriers: float,
    base_cpkm: float,
    df_carrier: pd.DataFrame,
    compare_year: int,
    report_week: str = None,
    report_month: str = None,
) -> tuple:
    """
    Calculate carrier and demand impacts for EU OP2 bridge using EU coefficients.

    Uses the same polynomial LCR model as country-level but with EU coefficients.

    Returns:
        Tuple[float, float]: (carrier_impact, demand_impact) in $/km
    """
    if (
        actual_loads <= 0
        or actual_carriers <= 0
        or op2_loads <= 0
        or op2_carriers <= 0
        or base_cpkm is None
        or pd.isna(base_cpkm)
    ):
        return 0.0, 0.0

    # Get EU volume coefficients
    slope = COUNTRY_COEFFICIENTS_VOLUME.get("EU_SLOPE", 0.01)
    intercept = COUNTRY_COEFFICIENTS_VOLUME.get("EU_INTERCEPT", 500)

    # Get EU LCR coefficients
    slope_lcr = COUNTRY_COEFFICIENTS_LCR.get("EU_SLOPE_LCR", 0.1)
    poly_lcr = COUNTRY_COEFFICIENTS_LCR.get("EU_POLY_LCR", 0.001)
    intercept_lcr = COUNTRY_COEFFICIENTS_LCR.get("EU_INTERCEPT_LCR", 1.0)

    # Lookup carrier scaling percentage for EU (use week or month)
    percentage = 0.0
    period = report_week or report_month
    if period is not None:
        year_str = f"R{compare_year}"
        lookup_mask = (
            (df_carrier["year"] == year_str)
            & (df_carrier["period"] == period)
            & (df_carrier["country"] == "EU")
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


def create_op2_eu_monthly_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 monthly benchmark bridge at EU-total level.

    Aggregates country-level OP2 metrics to EU level. Uses OP2 monthly_agg
    for EU base figures. Includes all impacts (tech, market, SET, carrier, demand).

    Input Table - df (actual data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_month     | string | Month in MXX format            |
        | orig_country     | string | Origin country code            |
        | total_cost_usd   | float  | Cost in USD                    |
        | distance_for_cpkm| float  | Distance in kilometers         |
        | executed_loads   | int    | Number of loads                |

    Input Table - df_op2:
        OP2 benchmark data with monthly_agg for EU base figures

    Input Table - df_carrier:
        Carrier scaling percentages

    Input Table - final_bridge_df:
        Completed MTD bridge for SET and market rate impact lookup

    Output Table:
        Standard OP2 monthly bridge with all impact columns
    """
    from .op2_data_extractor import extract_op2_monthly_base_cpkm
    from .op2_normalizer import compute_op2_normalized_cpkm_monthly
    from .op2_impacts import calculate_op2_tech_impact_monthly, calculate_op2_market_rate_impact_monthly
    from .op2_helpers import get_set_impact_for_op2_monthly

    logger.info("Creating OP2 EU-total monthly bridge...")

    eu_countries = ["DE", "ES", "FR", "IT", "UK"]

    # Get OP2 EU base metrics from monthly_agg
    op2_base_all = extract_op2_monthly_base_cpkm(df_op2)
    op2_eu_base = op2_base_all[op2_base_all["orig_country"] == "EU"].copy()

    if op2_eu_base.empty:
        logger.warning("No EU data found in OP2 monthly_agg. Skipping EU total bridge.")
        return pd.DataFrame()

    # Aggregate actual data across all EU5 countries
    df_eu = df[
        (df["report_month"].notna())
        & (df["orig_country"].isin(eu_countries))
    ].copy()

    actual_eu = (
        df_eu
        .groupby(["report_year", "report_month"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance_km=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    actual_eu["compare_cpkm"] = actual_eu["actual_cost"] / actual_eu["actual_distance_km"]

    # Calculate actual carriers directly from source data
    actual_eu["actual_carriers"] = actual_eu.apply(
        lambda r: df_eu[
            (df_eu["report_year"] == r["report_year"])
            & (df_eu["report_month"] == r["report_month"])
            & (df_eu["executed_loads"] > 0)
        ]["vehicle_carrier"].nunique(),
        axis=1,
    )

    # Aggregate normalized CPKM from country-level calculations
    country_norm = compute_op2_normalized_cpkm_monthly(df, df_op2, by_business=False)
    country_norm = country_norm[country_norm["orig_country"].isin(eu_countries)]
    eu_norm = country_norm.groupby(
        ["report_year", "report_month"],
        as_index=False,
    ).agg(op2_normalized_cost=("op2_normalized_cost", "sum"))
    eu_norm = eu_norm.merge(
        actual_eu[["report_year", "report_month", "actual_distance_km"]],
        on=["report_year", "report_month"],
        how="left",
    )
    eu_norm["op2_normalized_cpkm"] = eu_norm["op2_normalized_cost"] / eu_norm["actual_distance_km"]

    # Aggregate tech impact from country level
    country_tech = calculate_op2_tech_impact_monthly(df, df_op2, by_business=False)
    country_tech = country_tech[country_tech["orig_country"].isin(eu_countries)]
    eu_tech = country_tech.groupby(
        ["report_year", "report_month"],
        as_index=False,
    ).agg(op2_tech_impact_value=("op2_tech_impact_value", "sum"))

    # Aggregate market rate impact from country level
    country_market = calculate_op2_market_rate_impact_monthly(df, df_op2, final_bridge_df, by_business=False)
    country_market = country_market[country_market["orig_country"].isin(eu_countries)]
    eu_market = country_market.groupby(
        ["report_year", "report_month"],
        as_index=False,
    ).agg(op2_market_impact=("op2_market_impact", "sum"))

    # Merge all components
    bridge = (
        actual_eu
        .merge(op2_eu_base, on=["report_year", "report_month"], how="left")
        .merge(eu_norm[["report_year", "report_month", "op2_normalized_cpkm"]], on=["report_year", "report_month"], how="left")
        .merge(eu_tech, on=["report_year", "report_month"], how="left")
        .merge(eu_market, on=["report_year", "report_month"], how="left")
    )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_monthly"
    bridge["business"] = "Total"
    bridge["orig_country"] = "EU"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge["op2_normalized_cpkm"]

    # Get SET impact from MTD bridge for EU Total
    mtd_set_impact = get_set_impact_for_op2_monthly(final_bridge_df)
    mtd_set_eu = mtd_set_impact[
        (mtd_set_impact["orig_country"] == "EU")
        & (mtd_set_impact["business"] == "Total")
    ]
    bridge = bridge.merge(
        mtd_set_eu[["report_year", "report_month", "set_impact"]],
        on=["report_year", "report_month"],
        how="left",
    )

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = np.where(
        bridge["op2_base_loads"] > 0,
        (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100,
        0,
    )

    bridge["distance_variance_km"] = bridge["actual_distance_km"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = np.where(
        bridge["op2_base_distance"] > 0,
        (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100,
        0,
    )

    bridge["cpkm_variance"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]
    bridge["cpkm_variance_pct"] = np.where(
        bridge["op2_base_cpkm"] > 0,
        (bridge["cpkm_variance"] / bridge["op2_base_cpkm"]) * 100,
        0,
    )

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = np.where(
        bridge["op2_base_cost"] > 0,
        ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100,
        0,
    )

    # Normalized metrics
    bridge["normalized_variance"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
    bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate per-km impacts
    bridge["tech_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_tech_impact_value"].fillna(0) / bridge["actual_distance_km"],
        0,
    )

    bridge["market_rate_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_market_impact"].fillna(0) / bridge["actual_distance_km"],
        0,
    )

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_month"] + "_OP2"
    bridge["m2_distance_km"] = bridge["actual_distance_km"]
    bridge["benchmark_gap"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate carrier and demand impacts using EU coefficients
    bridge[["carrier_impact", "demand_impact"]] = bridge.apply(
        lambda r: _calculate_eu_op2_carrier_demand_impacts(
            actual_loads=r["actual_loads"],
            actual_carriers=r["actual_carriers"],
            op2_loads=r["op2_base_loads"],
            op2_carriers=r["op2_carriers"],
            base_cpkm=r["op2_base_cpkm"],
            df_carrier=df_carrier,
            compare_year=int(r["report_year"].replace("R", "")),
            report_month=r["report_month"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    # Null out non-applicable fields
    for col in ["premium_impact", "supply_rates", "report_week"]:
        bridge[col] = None

    logger.info(f"Created {len(bridge)} OP2 EU-total monthly bridge rows")
    return bridge


def create_op2_eu_monthly_business_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 monthly benchmark bridge at EU x business level.

    All metrics aggregated from country x business level calculations.
    Includes all impacts (tech, market, SET, carrier, demand).

    Input Tables:
        Same as create_op2_eu_monthly_bridge

    Output Table:
        Same structure as create_op2_eu_monthly_bridge but:
        - business column contains actual business values (not 'Total')
        - Granularity is EU x month x business
    """
    from .op2_data_extractor import extract_op2_monthly_base_cpkm, extract_op2_monthly_base_by_business
    from .op2_normalizer import compute_op2_normalized_cpkm_monthly
    from .op2_impacts import calculate_op2_tech_impact_monthly, calculate_op2_market_rate_impact_monthly
    from .op2_helpers import get_set_impact_for_op2_monthly

    logger.info("Creating OP2 EU x business monthly bridge...")

    eu_countries = ["DE", "ES", "FR", "IT", "UK"]

    # Filter to EU5 countries with valid months
    df_eu = df[
        (df["report_month"].notna())
        & (df["orig_country"].isin(eu_countries))
    ].copy()
    # Normalize business to uppercase for consistent matching
    df_eu["business"] = df_eu["business"].str.upper()

    # Aggregate actual data by business across all EU5 countries
    actual_eu = (
        df_eu
        .groupby(["report_year", "report_month", "business"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance_km=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    actual_eu["compare_cpkm"] = actual_eu["actual_cost"] / actual_eu["actual_distance_km"]

    # Calculate actual carriers directly from source data (count unique carriers across EU countries by business)
    actual_eu["actual_carriers"] = actual_eu.apply(
        lambda r: df_eu[
            (df_eu["report_year"] == r["report_year"])
            & (df_eu["report_month"] == r["report_month"])
            & (df_eu["business"] == r["business"])
            & (df_eu["executed_loads"] > 0)
        ]["vehicle_carrier"].nunique(),
        axis=1,
    )

    # Aggregate OP2 base metrics from country x business level
    country_business_norm = compute_op2_normalized_cpkm_monthly(df, df_op2, by_business=True)
    country_business_norm = country_business_norm[country_business_norm["orig_country"].isin(eu_countries)]
    country_business_norm["business"] = country_business_norm["business"].str.upper()

    # Also aggregate base metrics from country x business
    country_business_base = extract_op2_monthly_base_by_business(df_op2)
    country_business_base = country_business_base[country_business_base["orig_country"].isin(eu_countries)]
    country_business_base["business"] = country_business_base["business"].str.upper()

    eu_base = country_business_base.groupby(
        ["report_year", "report_month", "business"],
        as_index=False,
    ).agg(
        op2_base_distance=("op2_base_distance", "sum"),
        op2_base_cost=("op2_base_cost", "sum"),
        op2_base_loads=("op2_base_loads", "sum"),
    )
    eu_base["op2_base_cpkm"] = eu_base["op2_base_cost"] / eu_base["op2_base_distance"]

    # Get EU-level carriers from monthly_agg
    op2_base_all = extract_op2_monthly_base_cpkm(df_op2)
    op2_eu_carriers = op2_base_all[op2_base_all["orig_country"] == "EU"][
        ["report_year", "report_month", "op2_carriers"]
    ]
    eu_base = eu_base.merge(
        op2_eu_carriers,
        on=["report_year", "report_month"],
        how="left",
    )

    # Aggregate normalized CPKM from country x business level
    eu_norm = country_business_norm.groupby(
        ["report_year", "report_month", "business"],
        as_index=False,
    ).agg(
        op2_normalized_cost=("op2_normalized_cost", "sum"),
    )
    # Join with actual_eu to get distance for CPKM calculation
    eu_norm = eu_norm.merge(
        actual_eu[["report_year", "report_month", "business", "actual_distance_km"]],
        on=["report_year", "report_month", "business"],
        how="left",
    )
    eu_norm["op2_normalized_cpkm"] = eu_norm["op2_normalized_cost"] / eu_norm["actual_distance_km"]

    # Aggregate tech impact from country x business level
    country_tech = calculate_op2_tech_impact_monthly(df, df_op2, by_business=True)
    country_tech = country_tech[country_tech["orig_country"].isin(eu_countries)]
    country_tech["business"] = country_tech["business"].str.upper()
    eu_tech = country_tech.groupby(
        ["report_year", "report_month", "business"],
        as_index=False,
    ).agg(op2_tech_impact_value=("op2_tech_impact_value", "sum"))

    # Aggregate market rate impact from country x business level
    country_market = calculate_op2_market_rate_impact_monthly(df, df_op2, final_bridge_df, by_business=True)
    country_market = country_market[country_market["orig_country"].isin(eu_countries)]
    country_market["business"] = country_market["business"].str.upper()
    eu_market = country_market.groupby(
        ["report_year", "report_month", "business"],
        as_index=False,
    ).agg(op2_market_impact=("op2_market_impact", "sum"))

    # Merge all components
    bridge = (
        actual_eu
        .merge(eu_base, on=["report_year", "report_month", "business"], how="left")
        .merge(eu_norm[["report_year", "report_month", "business", "op2_normalized_cpkm"]], on=["report_year", "report_month", "business"], how="left")
        .merge(eu_tech, on=["report_year", "report_month", "business"], how="left")
        .merge(eu_market, on=["report_year", "report_month", "business"], how="left")
    )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_monthly"
    bridge["orig_country"] = "EU"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge["op2_normalized_cpkm"]

    # Populate OP2 columns
    bridge["op2_loads"] = bridge["op2_base_loads"]
    bridge["op2_distance_km"] = bridge["op2_base_distance"]
    bridge["op2_cost_usd"] = bridge["op2_base_cost"]
    bridge["op2_cpkm"] = bridge["op2_base_cpkm"]

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = np.where(
        bridge["op2_base_loads"] > 0,
        (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100,
        0,
    )

    bridge["distance_variance_km"] = bridge["actual_distance_km"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = np.where(
        bridge["op2_base_distance"] > 0,
        (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100,
        0,
    )

    bridge["cpkm_variance"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]
    bridge["cpkm_variance_pct"] = np.where(
        bridge["op2_base_cpkm"] > 0,
        (bridge["cpkm_variance"] / bridge["op2_base_cpkm"]) * 100,
        0,
    )

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = np.where(
        bridge["op2_base_cost"] > 0,
        ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100,
        0,
    )

    # Normalized metrics
    bridge["normalized_variance"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
    bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]

    # Get SET impact from MTD bridge for EU by business
    mtd_set_impact = get_set_impact_for_op2_monthly(final_bridge_df)
    mtd_set_eu = mtd_set_impact[mtd_set_impact["orig_country"] == "EU"]
    mtd_set_eu["business"] = mtd_set_eu["business"].str.upper()
    bridge = bridge.merge(
        mtd_set_eu[["report_year", "report_month", "business", "set_impact"]],
        on=["report_year", "report_month", "business"],
        how="left",
    )

    # Calculate per-km impacts
    bridge["tech_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_tech_impact_value"].fillna(0) / bridge["actual_distance_km"],
        0,
    )

    bridge["market_rate_impact"] = np.where(
        bridge["actual_distance_km"] > 0,
        bridge["op2_market_impact"].fillna(0) / bridge["actual_distance_km"],
        0,
    )

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_month"] + "_OP2"
    bridge["m2_distance_km"] = bridge["actual_distance_km"]
    bridge["benchmark_gap"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate carrier and demand impacts using EU coefficients
    bridge[["carrier_impact", "demand_impact"]] = bridge.apply(
        lambda r: _calculate_eu_op2_carrier_demand_impacts(
            actual_loads=r["actual_loads"],
            actual_carriers=r["actual_carriers"],
            op2_loads=r["op2_base_loads"] if pd.notna(r["op2_base_loads"]) else r["actual_loads"],
            op2_carriers=r["op2_carriers"] if pd.notna(r["op2_carriers"]) else r["actual_carriers"],
            base_cpkm=r["op2_normalized_cpkm"] if pd.notna(r["op2_normalized_cpkm"]) else r["compare_cpkm"],
            df_carrier=df_carrier,
            compare_year=int(r["report_year"].replace("R", "")),
            report_month=r["report_month"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    # Null out non-applicable fields for monthly
    for col in ["premium_impact", "supply_rates", "report_week"]:
        bridge[col] = None

    logger.info(f"Created {len(bridge)} OP2 EU x business monthly bridge rows")
    return bridge


def _get_quarter_from_month(month: str) -> str:
    """Map month (M01-M12) to quarter (Q1-Q4)."""
    month_num = int(month.replace("M", ""))
    if month_num <= 3:
        return "Q1"
    elif month_num <= 6:
        return "Q2"
    elif month_num <= 9:
        return "Q3"
    else:
        return "Q4"


def create_op2_eu_quarterly_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 quarterly benchmark bridge at EU-total level.

    Aggregates country-level OP2 metrics to EU level. Uses OP2 quarterly data
    (Period = Q1-Q4) for EU base figures. Other metrics aggregated from
    monthly EU bridges.

    Input Table - df (actual data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_month     | string | Month in MXX format            |
        | orig_country     | string | Origin country code            |
        | total_cost_usd   | float  | Cost in USD                    |
        | distance_for_cpkm| float  | Distance in kilometers         |
        | executed_loads   | int    | Number of loads                |

    Input Table - df_op2:
        OP2 benchmark data with monthly_agg containing Q1-Q4 periods

    Input Table - df_carrier:
        Carrier scaling percentages

    Input Table - final_bridge_df:
        Completed bridges for impact lookup

    Output Table:
        Standard OP2 quarterly bridge with all impact columns
    """
    from .op2_data_extractor import extract_op2_quarterly_base_cpkm

    logger.info("Creating OP2 EU-total quarterly bridge...")

    eu_countries = ["DE", "ES", "FR", "IT", "UK"]

    # Get OP2 EU quarterly base metrics
    op2_base_all = extract_op2_quarterly_base_cpkm(df_op2)
    op2_eu_base = op2_base_all[op2_base_all["orig_country"] == "EU"].copy()

    if op2_eu_base.empty:
        logger.warning("No EU quarterly data found in OP2. Skipping EU quarterly bridge.")
        return pd.DataFrame()

    # Add quarter column to actual data
    df_eu = df[
        (df["report_month"].notna())
        & (df["orig_country"].isin(eu_countries))
    ].copy()
    df_eu["report_quarter"] = df_eu["report_month"].apply(_get_quarter_from_month)

    # Aggregate actual quarterly data across EU
    actual_eu = (
        df_eu
        .groupby(["report_year", "report_quarter"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance_km=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    # Calculate carriers per quarter
    actual_eu["actual_carriers"] = actual_eu.apply(
        lambda r: df_eu[
            (df_eu["report_year"] == r["report_year"])
            & (df_eu["report_quarter"] == r["report_quarter"])
            & (df_eu["executed_loads"] > 0)
        ]["vehicle_carrier"].nunique(),
        axis=1,
    )

    actual_eu["compare_cpkm"] = actual_eu["actual_cost"] / actual_eu["actual_distance_km"]

    # Aggregate from EU monthly bridges
    monthly_eu_op2 = final_bridge_df[
        (final_bridge_df["bridge_type"] == "op2_monthly")
        & (final_bridge_df["business"] == "Total")
        & (final_bridge_df["orig_country"] == "EU")
    ].copy()

    if not monthly_eu_op2.empty:
        monthly_eu_op2["report_quarter"] = monthly_eu_op2["report_month"].apply(_get_quarter_from_month)

        quarterly_metrics = monthly_eu_op2.groupby(
            ["report_year", "report_quarter"],
            as_index=False,
        ).agg(
            op2_normalized_cost=("normalised_cpkm", lambda x: (x * monthly_eu_op2.loc[x.index, "actual_distance_km"]).sum()),
            norm_distance=("actual_distance_km", "sum"),
            op2_tech_impact_value=("tech_impact", lambda x: (x * monthly_eu_op2.loc[x.index, "actual_distance_km"]).sum()),
            op2_market_impact=("market_rate_impact", lambda x: (x * monthly_eu_op2.loc[x.index, "actual_distance_km"]).sum()),
            set_impact_sum=("set_impact", lambda x: (x * monthly_eu_op2.loc[x.index, "actual_distance_km"]).sum() if x.notna().any() else 0),
        )
        quarterly_metrics["op2_normalized_cpkm"] = quarterly_metrics["op2_normalized_cost"] / quarterly_metrics["norm_distance"]
    else:
        quarterly_metrics = pd.DataFrame()

    # Merge components
    bridge = actual_eu.merge(
        op2_eu_base,
        on=["report_year", "report_quarter"],
        how="left",
    )

    if not quarterly_metrics.empty:
        bridge = bridge.merge(
            quarterly_metrics[["report_year", "report_quarter", "op2_normalized_cpkm",
                              "op2_tech_impact_value", "op2_market_impact", "set_impact_sum", "norm_distance"]],
            on=["report_year", "report_quarter"],
            how="left",
        )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_quarterly"
    bridge["business"] = "Total"
    bridge["orig_country"] = "EU"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge.get("op2_normalized_cpkm", bridge["op2_base_cpkm"])

    # SET impact
    if "set_impact_sum" in bridge.columns and "norm_distance" in bridge.columns:
        bridge["set_impact"] = np.where(
            bridge["norm_distance"] > 0,
            bridge["set_impact_sum"] / bridge["norm_distance"],
            0,
        )
    else:
        bridge["set_impact"] = None

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = np.where(
        bridge["op2_base_loads"] > 0,
        (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100,
        0,
    )

    bridge["distance_variance_km"] = bridge["actual_distance_km"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = np.where(
        bridge["op2_base_distance"] > 0,
        (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100,
        0,
    )

    bridge["cpkm_variance"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]
    bridge["cpkm_variance_pct"] = np.where(
        bridge["op2_base_cpkm"] > 0,
        (bridge["cpkm_variance"] / bridge["op2_base_cpkm"]) * 100,
        0,
    )

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = np.where(
        bridge["op2_base_cost"] > 0,
        ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100,
        0,
    )

    # Normalized metrics
    if "op2_normalized_cpkm" in bridge.columns:
        bridge["normalized_variance"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
        bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]
    else:
        bridge["normalized_variance"] = 0
        bridge["mix_impact"] = 0

    # Per-km impacts
    if "op2_tech_impact_value" in bridge.columns:
        bridge["tech_impact"] = np.where(
            bridge["actual_distance_km"] > 0,
            bridge["op2_tech_impact_value"].fillna(0) / bridge["actual_distance_km"],
            0,
        )
    else:
        bridge["tech_impact"] = 0

    if "op2_market_impact" in bridge.columns:
        bridge["market_rate_impact"] = np.where(
            bridge["actual_distance_km"] > 0,
            bridge["op2_market_impact"].fillna(0) / bridge["actual_distance_km"],
            0,
        )
    else:
        bridge["market_rate_impact"] = 0

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_quarter"] + "_OP2"
    bridge["m2_distance_km"] = bridge["actual_distance_km"]
    bridge["benchmark_gap"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate carrier and demand impacts using EU coefficients
    bridge[["carrier_impact", "demand_impact"]] = bridge.apply(
        lambda r: _calculate_eu_op2_carrier_demand_impacts(
            actual_loads=r["actual_loads"],
            actual_carriers=r["actual_carriers"],
            op2_loads=r["op2_base_loads"],
            op2_carriers=r["op2_carriers"],
            base_cpkm=r["op2_base_cpkm"],
            df_carrier=df_carrier,
            compare_year=int(r["report_year"].replace("R", "")),
            report_month=r["report_quarter"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    # Null out non-applicable fields
    for col in ["premium_impact", "supply_rates", "report_week", "report_month"]:
        bridge[col] = None

    logger.info(f"Created {len(bridge)} OP2 EU-total quarterly bridge rows")
    return bridge


def create_op2_eu_quarterly_business_bridge(
    df: pd.DataFrame,
    df_op2: pd.DataFrame,
    df_carrier: pd.DataFrame,
    final_bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create OP2 quarterly benchmark bridge at EU x business level.

    OP2 doesn't have quarterly data at business level, so OP2 base metrics
    are aggregated from monthly_bridge data across EU countries.
    Other metrics aggregated from monthly EU business bridges.

    Input Tables:
        Same as create_op2_eu_quarterly_bridge

    Output Table:
        Same structure but with business dimension
    """
    from .op2_data_extractor import extract_op2_monthly_detailed_by_business, extract_op2_quarterly_base_cpkm

    logger.info("Creating OP2 EU x business quarterly bridge...")

    eu_countries = ["DE", "ES", "FR", "IT", "UK"]

    # Add quarter column to actual data
    df_eu = df[
        (df["report_month"].notna())
        & (df["orig_country"].isin(eu_countries))
    ].copy()
    df_eu["report_quarter"] = df_eu["report_month"].apply(_get_quarter_from_month)
    df_eu["business"] = df_eu["business"].str.upper()

    # Aggregate actual quarterly data by business across EU
    actual_eu = (
        df_eu
        .groupby(["report_year", "report_quarter", "business"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance_km=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    # Calculate carriers per quarter by business
    actual_eu["actual_carriers"] = actual_eu.apply(
        lambda r: df_eu[
            (df_eu["report_year"] == r["report_year"])
            & (df_eu["report_quarter"] == r["report_quarter"])
            & (df_eu["business"] == r["business"])
            & (df_eu["executed_loads"] > 0)
        ]["vehicle_carrier"].nunique(),
        axis=1,
    )

    actual_eu["compare_cpkm"] = actual_eu["actual_cost"] / actual_eu["actual_distance_km"]

    # Get OP2 base from monthly detailed data (Bridge type == 'monthly') across EU countries
    monthly_base = extract_op2_monthly_detailed_by_business(df_op2)
    monthly_base = monthly_base[monthly_base["orig_country"].isin(eu_countries)]
    monthly_base["report_quarter"] = monthly_base["report_month"].apply(_get_quarter_from_month)
    monthly_base["business"] = monthly_base["business"].str.upper()

    # Aggregate to EU quarterly by business
    op2_base = monthly_base.groupby(
        ["report_year", "report_quarter", "business"],
        as_index=False,
    ).agg(
        op2_base_distance=("op2_base_distance", "sum"),
        op2_base_cost=("op2_base_cost", "sum"),
        op2_base_loads=("op2_base_loads", "sum"),
    )
    op2_base["op2_base_cpkm"] = op2_base["op2_base_cost"] / op2_base["op2_base_distance"]

    # Get quarterly carriers from EU-level quarterly data
    quarterly_base = extract_op2_quarterly_base_cpkm(df_op2)
    eu_carriers = quarterly_base[quarterly_base["orig_country"] == "EU"][
        ["report_year", "report_quarter", "op2_carriers"]
    ]
    op2_base = op2_base.merge(
        eu_carriers,
        on=["report_year", "report_quarter"],
        how="left",
    )

    # Aggregate from EU monthly business bridges for impacts
    monthly_eu_op2 = final_bridge_df[
        (final_bridge_df["bridge_type"] == "op2_monthly")
        & (final_bridge_df["business"] != "Total")
        & (final_bridge_df["orig_country"] == "EU")
    ].copy()

    if not monthly_eu_op2.empty:
        monthly_eu_op2["report_quarter"] = monthly_eu_op2["report_month"].apply(_get_quarter_from_month)
        monthly_eu_op2["business"] = monthly_eu_op2["business"].str.upper()

        quarterly_metrics = monthly_eu_op2.groupby(
            ["report_year", "report_quarter", "business"],
            as_index=False,
        ).agg(
            op2_normalized_cost=("normalised_cpkm", lambda x: (x * monthly_eu_op2.loc[x.index, "actual_distance_km"]).sum()),
            norm_distance=("actual_distance_km", "sum"),
            op2_tech_impact_value=("tech_impact", lambda x: (x * monthly_eu_op2.loc[x.index, "actual_distance_km"]).sum()),
            op2_market_impact=("market_rate_impact", lambda x: (x * monthly_eu_op2.loc[x.index, "actual_distance_km"]).sum()),
            set_impact_sum=("set_impact", lambda x: (x * monthly_eu_op2.loc[x.index, "actual_distance_km"]).sum() if x.notna().any() else 0),
        )
        quarterly_metrics["op2_normalized_cpkm"] = quarterly_metrics["op2_normalized_cost"] / quarterly_metrics["norm_distance"]
    else:
        quarterly_metrics = pd.DataFrame()

    # Merge components
    bridge = actual_eu.merge(
        op2_base,
        on=["report_year", "report_quarter", "business"],
        how="left",
    )

    if not quarterly_metrics.empty:
        bridge = bridge.merge(
            quarterly_metrics[["report_year", "report_quarter", "business",
                              "op2_normalized_cpkm", "op2_tech_impact_value", "op2_market_impact",
                              "set_impact_sum", "norm_distance"]],
            on=["report_year", "report_quarter", "business"],
            how="left",
        )

    # Fill bridge schema
    bridge["bridge_type"] = "op2_quarterly"
    bridge["orig_country"] = "EU"
    bridge["base_cpkm"] = bridge["op2_base_cpkm"]
    bridge["normalised_cpkm"] = bridge.get("op2_normalized_cpkm", bridge["op2_base_cpkm"])

    # SET impact
    if "set_impact_sum" in bridge.columns and "norm_distance" in bridge.columns:
        bridge["set_impact"] = np.where(
            bridge["norm_distance"] > 0,
            bridge["set_impact_sum"] / bridge["norm_distance"],
            0,
        )
    else:
        bridge["set_impact"] = None

    # Calculate variance metrics
    bridge["loads_variance"] = bridge["actual_loads"] - bridge["op2_base_loads"]
    bridge["loads_variance_pct"] = np.where(
        bridge["op2_base_loads"] > 0,
        (bridge["loads_variance"] / bridge["op2_base_loads"]) * 100,
        0,
    )

    bridge["distance_variance_km"] = bridge["actual_distance_km"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = np.where(
        bridge["op2_base_distance"] > 0,
        (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100,
        0,
    )

    bridge["cpkm_variance"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]
    bridge["cpkm_variance_pct"] = np.where(
        bridge["op2_base_cpkm"] > 0,
        (bridge["cpkm_variance"] / bridge["op2_base_cpkm"]) * 100,
        0,
    )

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = np.where(
        bridge["op2_base_cost"] > 0,
        ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100,
        0,
    )

    # Normalized metrics
    if "op2_normalized_cpkm" in bridge.columns:
        bridge["normalized_variance"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
        bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]
    else:
        bridge["normalized_variance"] = 0
        bridge["mix_impact"] = 0

    # Per-km impacts
    if "op2_tech_impact_value" in bridge.columns:
        bridge["tech_impact"] = np.where(
            bridge["actual_distance_km"] > 0,
            bridge["op2_tech_impact_value"].fillna(0) / bridge["actual_distance_km"],
            0,
        )
    else:
        bridge["tech_impact"] = 0

    if "op2_market_impact" in bridge.columns:
        bridge["market_rate_impact"] = np.where(
            bridge["actual_distance_km"] > 0,
            bridge["op2_market_impact"].fillna(0) / bridge["actual_distance_km"],
            0,
        )
    else:
        bridge["market_rate_impact"] = 0

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_quarter"] + "_OP2"
    bridge["m2_distance_km"] = bridge["actual_distance_km"]
    bridge["benchmark_gap"] = bridge["compare_cpkm"] - bridge["op2_base_cpkm"]

    # Calculate carrier and demand impacts using EU coefficients
    bridge[["carrier_impact", "demand_impact"]] = bridge.apply(
        lambda r: _calculate_eu_op2_carrier_demand_impacts(
            actual_loads=r["actual_loads"],
            actual_carriers=r["actual_carriers"],
            op2_loads=r["op2_base_loads"] if pd.notna(r["op2_base_loads"]) else r["actual_loads"],
            op2_carriers=r["op2_carriers"] if pd.notna(r["op2_carriers"]) else r["actual_carriers"],
            base_cpkm=r["op2_normalized_cpkm"] if pd.notna(r.get("op2_normalized_cpkm")) else r["compare_cpkm"],
            df_carrier=df_carrier,
            compare_year=int(r["report_year"].replace("R", "")),
            report_month=r["report_quarter"],
        ),
        axis=1,
        result_type="expand",
    )

    bridge["carrier_and_demand_impact"] = bridge["carrier_impact"] + bridge["demand_impact"]

    # Null out non-applicable fields
    for col in ["premium_impact", "supply_rates", "report_week", "report_month"]:
        bridge[col] = None

    logger.info(f"Created {len(bridge)} OP2 EU x business quarterly bridge rows")
    return bridge
