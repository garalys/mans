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
            actual_distance=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    actual_eu["compare_cpkm"] = actual_eu["actual_cost"] / actual_eu["actual_distance"]

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
        actual_distance=("actual_distance", "sum"),
    )
    eu_norm["op2_normalized_cpkm"] = eu_norm["op2_normalized_cost"] / eu_norm["actual_distance"]

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

    bridge["distance_variance_km"] = bridge["actual_distance"] - bridge["op2_base_distance"]
    bridge["distance_variance_pct"] = (bridge["distance_variance_km"] / bridge["op2_base_distance"]) * 100

    bridge["cost_variance_mm"] = (bridge["actual_cost"] - bridge["op2_base_cost"]) / 1_000_000
    bridge["cost_variance_pct"] = ((bridge["actual_cost"] - bridge["op2_base_cost"]) / bridge["op2_base_cost"]) * 100

    # Normalized metrics
    bridge["compare_cpkm_vs_normalised_op2_cpkm"] = bridge["compare_cpkm"] - bridge["op2_normalized_cpkm"]
    bridge["mix_impact"] = bridge["op2_normalized_cpkm"] - bridge["op2_base_cpkm"]

    bridge["tech_impact"] = np.where(
        bridge["actual_distance"] > 0,
        bridge["op2_tech_impact_value"] / bridge["actual_distance"],
        0,
    )

    bridge["market_rate_impact"] = np.where(
        bridge["actual_distance"] > 0,
        bridge["op2_market_impact"] / bridge["actual_distance"],
        0,
    )

    bridge["bridging_value"] = bridge["report_year"] + "_" + bridge["report_week"] + "_OP2"
    bridge["w2_distance_km"] = bridge["actual_distance"]
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

    # Aggregate actual data by business across all EU5 countries
    actual_eu = (
        df[
            (df["report_week"].notna())
            & (df["orig_country"].isin(eu_countries))
        ]
        .groupby(["report_year", "report_week", "business"], as_index=False)
        .agg(
            actual_cost=("total_cost_usd", "sum"),
            actual_distance=("distance_for_cpkm", "sum"),
            actual_loads=("executed_loads", "sum"),
        )
    )

    actual_eu["compare_cpkm"] = actual_eu["actual_cost"] / actual_eu["actual_distance"]

    # Get actual carriers from YoY bridge by business
    yoy_carriers = _get_eu_carriers_from_yoy_bridge_by_business(final_bridge_df)
    actual_eu = actual_eu.merge(
        yoy_carriers,
        on=["report_year", "report_week", "business"],
        how="left",
    )
    actual_eu["actual_carriers"] = actual_eu["actual_carriers"].fillna(0)

    # Aggregate OP2 base metrics from country x business level
    # (since EU doesn't have business splits in OP2 data)
    country_business_norm = compute_op2_normalized_cpkm_weekly(df, df_op2, by_business=True)
    country_business_norm = country_business_norm[country_business_norm["orig_country"].isin(eu_countries)]

    # Also aggregate base metrics from country x business
    from .op2_data_extractor import extract_op2_weekly_base_by_business
    country_business_base = extract_op2_weekly_base_by_business(df_op2)
    country_business_base = country_business_base[country_business_base["orig_country"].isin(eu_countries)]

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
        actual_distance=("actual_distance", "sum"),
    )
    eu_norm["op2_normalized_cpkm"] = eu_norm["op2_normalized_cost"] / eu_norm["actual_distance"]

    # Aggregate tech impact from country x business level
    country_tech = calculate_op2_tech_impact(df, df_op2, by_business=True)
    country_tech = country_tech[country_tech["orig_country"].isin(eu_countries)]
    eu_tech = country_tech.groupby(
        ["report_year", "report_week", "business"],
        as_index=False,
    ).agg(op2_tech_impact_value=("op2_tech_impact_value", "sum"))

    # Aggregate market rate impact from country x business level
    country_market = calculate_op2_market_rate_impact(df, df_op2, final_bridge_df, by_business=True)
    country_market = country_market[country_market["orig_country"].isin(eu_countries)]
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

    bridge["distance_variance_km"] = bridge["actual_distance"] - bridge["op2_base_distance"]
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
        bridge["actual_distance"] > 0,
        bridge["op2_tech_impact_value"] / bridge["actual_distance"],
        0,
    )

    bridge["market_rate_impact"] = np.where(
        bridge["actual_distance"] > 0,
        bridge["op2_market_impact"] / bridge["actual_distance"],
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

    # Lookup carrier scaling percentage for EU
    percentage = 0.0
    if report_week is not None:
        year_str = f"R{compare_year}"
        lookup_mask = (
            (df_carrier["year"] == year_str)
            & (df_carrier["period"] == report_week)
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
