"""
Hierarchical Mix Calculator

Computes 5-level hierarchical mix percentages for transportation cost analysis:
    1. Country mix (orig_country share of total distance)
    2. Corridor mix (dest_country share within each orig_country)
    3. Distance band mix (distance_band share within each corridor)
    4. Business flow mix (business share within each distance band cell)
    5. Equipment type mix (is_set share within each business flow cell)

Supports both actual data and OP2 data (with hardcoded RET=100%, SET=0%).
Zero cascade: if any parent-level mix is 0, all descendant mixes are 0.
"""

import pandas as pd
import numpy as np

from ..config.logging_config import logger


GRAIN_COLS = ["orig_country", "dest_country", "distance_band", "business", "is_set"]


def compute_hierarchical_mix(
    data_df: pd.DataFrame,
    distance_col: str = "distance_for_cpkm",
    is_op2_base: bool = False,
) -> pd.DataFrame:
    """
    Compute hierarchical mix percentages at the most granular level.

    For OP2 base data (is_op2_base=True), the equipment type mix is hardcoded:
    RET (is_set=False) = 100%, SET (is_set=True) = 0%.

    Args:
        data_df: DataFrame with at minimum orig_country, dest_country,
                 distance_band, business, is_set, and a distance column.
        distance_col: Name of the distance column to aggregate.
        is_op2_base: If True, hardcode equipment type mix (RET=100%, SET=0%).

    Returns:
        DataFrame at full grain (orig_country, dest_country, distance_band,
        business, is_set) with columns:
            - distance: aggregated distance
            - country_mix_pct
            - corridor_mix_pct
            - distance_band_mix_pct
            - business_mix_pct
            - equipment_type_mix_pct
    """
    df = data_df.copy()

    # For OP2 data that has no is_set column, create it
    if "is_set" not in df.columns:
        df["is_set"] = False

    # Aggregate to full grain
    grain = df.groupby(GRAIN_COLS, as_index=False).agg(
        distance=(distance_col, "sum")
    )

    # Total distance
    total_distance = grain["distance"].sum()
    if total_distance == 0:
        grain["country_mix_pct"] = 0.0
        grain["corridor_mix_pct"] = 0.0
        grain["distance_band_mix_pct"] = 0.0
        grain["business_mix_pct"] = 0.0
        grain["equipment_type_mix_pct"] = 0.0
        return grain

    # 1. Country mix: each orig_country's share of total distance
    country_totals = grain.groupby("orig_country", as_index=False)["distance"].sum()
    country_totals = country_totals.rename(columns={"distance": "country_total"})
    country_totals["country_mix_pct"] = country_totals["country_total"] / total_distance

    grain = grain.merge(country_totals, on="orig_country", how="left")

    # 2. Corridor mix: each dest_country's share within its orig_country
    corridor_totals = grain.groupby(
        ["orig_country", "dest_country"], as_index=False
    )["distance"].sum()
    corridor_totals = corridor_totals.rename(columns={"distance": "corridor_total"})

    grain = grain.merge(corridor_totals, on=["orig_country", "dest_country"], how="left")

    grain["corridor_mix_pct"] = np.where(
        grain["country_total"] > 0,
        grain["corridor_total"] / grain["country_total"],
        0.0,
    )
    # Zero cascade: if country_mix_pct == 0, corridor_mix_pct = 0
    grain.loc[grain["country_mix_pct"] == 0, "corridor_mix_pct"] = 0.0

    # 3. Distance band mix: each distance_band's share within its corridor
    band_totals = grain.groupby(
        ["orig_country", "dest_country", "distance_band"], as_index=False
    )["distance"].sum()
    band_totals = band_totals.rename(columns={"distance": "band_total"})

    grain = grain.merge(
        band_totals, on=["orig_country", "dest_country", "distance_band"], how="left"
    )

    grain["distance_band_mix_pct"] = np.where(
        grain["corridor_total"] > 0,
        grain["band_total"] / grain["corridor_total"],
        0.0,
    )
    # Zero cascade
    grain.loc[grain["corridor_mix_pct"] == 0, "distance_band_mix_pct"] = 0.0

    # 4. Business flow mix: each business's share within its distance band cell
    business_totals = grain.groupby(
        ["orig_country", "dest_country", "distance_band", "business"], as_index=False
    )["distance"].sum()
    business_totals = business_totals.rename(columns={"distance": "business_total"})

    grain = grain.merge(
        business_totals,
        on=["orig_country", "dest_country", "distance_band", "business"],
        how="left",
    )

    grain["business_mix_pct"] = np.where(
        grain["band_total"] > 0,
        grain["business_total"] / grain["band_total"],
        0.0,
    )
    # Zero cascade
    grain.loc[grain["distance_band_mix_pct"] == 0, "business_mix_pct"] = 0.0

    # 5. Equipment type mix: is_set share within each business flow cell
    if is_op2_base:
        # OP2 has no SET: hardcode RET=100%, SET=0%
        grain["equipment_type_mix_pct"] = np.where(
            grain["is_set"] == False, 1.0, 0.0
        )
    else:
        grain["equipment_type_mix_pct"] = np.where(
            grain["business_total"] > 0,
            grain["distance"] / grain["business_total"],
            0.0,
        )
    # Zero cascade
    grain.loc[grain["business_mix_pct"] == 0, "equipment_type_mix_pct"] = 0.0

    # Clean up intermediate columns
    grain = grain.drop(
        columns=["country_total", "corridor_total", "band_total", "business_total"],
    )

    return grain


def compute_normalised_distance(
    base_grain: pd.DataFrame,
    compare_grain: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute normalised distance for compare year using base year distribution.

    Formula: normalised_distance(cell) = (total_compare / total_base) * base_distance(cell)

    Args:
        base_grain: Base year grain DataFrame (from compute_hierarchical_mix)
                    with 'distance' column.
        compare_grain: Compare year grain DataFrame with 'distance' column.

    Returns:
        DataFrame at full grain with 'normalised_distance' column added,
        merged with base and compare data.
    """
    total_base = base_grain["distance"].sum()
    total_compare = compare_grain["distance"].sum()

    if total_base == 0:
        scale = 0.0
    else:
        scale = total_compare / total_base

    # Start from base grain and add normalised distance
    result = base_grain[GRAIN_COLS + ["distance"]].copy()
    result = result.rename(columns={"distance": "base_distance"})
    result["normalised_distance"] = scale * result["base_distance"]

    # Merge compare distance
    compare_dist = compare_grain[GRAIN_COLS + ["distance"]].copy()
    compare_dist = compare_dist.rename(columns={"distance": "compare_distance"})

    result = result.merge(compare_dist, on=GRAIN_COLS, how="outer")
    result["base_distance"] = result["base_distance"].fillna(0)
    result["normalised_distance"] = result["normalised_distance"].fillna(0)
    result["compare_distance"] = result["compare_distance"].fillna(0)

    return result


def compute_seven_metrics(
    base_mix: pd.DataFrame,
    compare_mix: pd.DataFrame,
    norm_dist_df: pd.DataFrame,
    base_cpkm_df: pd.DataFrame,
    compare_cpkm_df: pd.DataFrame,
) -> dict:
    """
    Compute the seven normalisation metrics at the most granular level.

    Args:
        base_mix: Base mix DataFrame from compute_hierarchical_mix
        compare_mix: Compare mix DataFrame from compute_hierarchical_mix
        norm_dist_df: Normalised distance DataFrame from compute_normalised_distance
        base_cpkm_df: DataFrame with GRAIN_COLS + ['cpkm'] for base rates
        compare_cpkm_df: DataFrame with GRAIN_COLS + ['cpkm'] for compare rates

    Returns:
        Dict with keys: base_spend, rate_cost, country_cost, corridor_cost,
        distance_band_cost, business_flow_cost, equipment_type_cost
    """
    mix_cols = [
        "country_mix_pct", "corridor_mix_pct", "distance_band_mix_pct",
        "business_mix_pct", "equipment_type_mix_pct",
    ]

    # Merge everything into one working DataFrame
    work = norm_dist_df.copy()

    # Merge base mix
    base_mix_cols = GRAIN_COLS + mix_cols
    base_mix_renamed = base_mix[base_mix_cols].copy()
    base_mix_renamed = base_mix_renamed.rename(
        columns={c: f"base_{c}" for c in mix_cols}
    )
    work = work.merge(base_mix_renamed, on=GRAIN_COLS, how="left")

    # Merge compare mix
    compare_mix_renamed = compare_mix[base_mix_cols].copy()
    compare_mix_renamed = compare_mix_renamed.rename(
        columns={c: f"compare_{c}" for c in mix_cols}
    )
    work = work.merge(compare_mix_renamed, on=GRAIN_COLS, how="left")

    # Merge base CPKM
    work = work.merge(
        base_cpkm_df[GRAIN_COLS + ["cpkm"]].rename(columns={"cpkm": "base_cpkm_cell"}),
        on=GRAIN_COLS, how="left",
    )

    # Merge compare CPKM
    work = work.merge(
        compare_cpkm_df[GRAIN_COLS + ["cpkm"]].rename(columns={"cpkm": "compare_cpkm_cell"}),
        on=GRAIN_COLS, how="left",
    )

    # Fill NaN with 0
    for col in work.columns:
        if col not in GRAIN_COLS:
            work[col] = work[col].fillna(0)

    # Compare cost at cell level
    work["compare_cost_cell"] = work["compare_distance"] * work["compare_cpkm_cell"]

    # 1. Base spend: adjusted_distance * base_cpkm
    work["base_spend_cell"] = work["normalised_distance"] * work["base_cpkm_cell"]

    # 2. Rate cost: product(all 5 base mix) * compare_cost
    work["base_mix_product"] = (
        work["base_country_mix_pct"]
        * work["base_corridor_mix_pct"]
        * work["base_distance_band_mix_pct"]
        * work["base_business_mix_pct"]
        * work["base_equipment_type_mix_pct"]
    )
    work["rate_cost_cell"] = work["base_mix_product"] * work["compare_cost_cell"]

    # 3. Country cost: base(corridor,band,business,equip) * compare(country) * compare_cost
    work["country_cost_cell"] = (
        work["base_corridor_mix_pct"]
        * work["base_distance_band_mix_pct"]
        * work["base_business_mix_pct"]
        * work["base_equipment_type_mix_pct"]
        * work["compare_country_mix_pct"]
        * work["compare_cost_cell"]
    )

    # 4. Corridor cost: base(band,business,equip) * compare(country,corridor) * compare_cost
    work["corridor_cost_cell"] = (
        work["base_distance_band_mix_pct"]
        * work["base_business_mix_pct"]
        * work["base_equipment_type_mix_pct"]
        * work["compare_country_mix_pct"]
        * work["compare_corridor_mix_pct"]
        * work["compare_cost_cell"]
    )

    # 5. Distance band cost: base(business,equip) * compare(country,corridor,band) * compare_cost
    work["distance_band_cost_cell"] = (
        work["base_business_mix_pct"]
        * work["base_equipment_type_mix_pct"]
        * work["compare_country_mix_pct"]
        * work["compare_corridor_mix_pct"]
        * work["compare_distance_band_mix_pct"]
        * work["compare_cost_cell"]
    )

    # 6. Business flow cost: base(equip) * compare(country,corridor,band,business) * compare_cost
    work["business_flow_cost_cell"] = (
        work["base_equipment_type_mix_pct"]
        * work["compare_country_mix_pct"]
        * work["compare_corridor_mix_pct"]
        * work["compare_distance_band_mix_pct"]
        * work["compare_business_mix_pct"]
        * work["compare_cost_cell"]
    )

    # 7. Equipment type cost: product(all 5 compare mix) * compare_cost
    work["compare_mix_product"] = (
        work["compare_country_mix_pct"]
        * work["compare_corridor_mix_pct"]
        * work["compare_distance_band_mix_pct"]
        * work["compare_business_mix_pct"]
        * work["compare_equipment_type_mix_pct"]
    )
    work["equipment_type_cost_cell"] = work["compare_mix_product"] * work["compare_cost_cell"]

    # Sum over all cells
    metrics = {
        "base_spend": work["base_spend_cell"].sum(),
        "rate_cost": work["rate_cost_cell"].sum(),
        "country_cost": work["country_cost_cell"].sum(),
        "corridor_cost": work["corridor_cost_cell"].sum(),
        "distance_band_cost": work["distance_band_cost_cell"].sum(),
        "business_flow_cost": work["business_flow_cost_cell"].sum(),
        "equipment_type_cost": work["equipment_type_cost_cell"].sum(),
    }

    return metrics


def compute_mix_impacts(
    seven_metrics: dict,
    compare_total_distance: float,
    bridge_level: str = "total",
    aggregation_level: str = "eu",
) -> dict:
    """
    Compute per-km mix impacts from the seven metrics.

    Each impact = (this_metric - previous_metric) / compare_total_distance

    Args:
        seven_metrics: Dict from compute_seven_metrics
        compare_total_distance: Total compare distance for division
        bridge_level: 'total' or 'business' - controls which mix impacts to include
        aggregation_level: 'eu' or 'country' - controls whether country_mix is included

    Returns:
        Dict with per-km impact values:
            - base_cpkm (from base_spend)
            - normalised_cpkm (from base_spend)
            - rate_impact
            - country_mix
            - corridor_mix
            - distance_band_mix
            - business_flow_mix
            - equipment_type_mix
            - mix_impact (sum of applicable mix components)
    """
    if compare_total_distance == 0:
        return {
            "base_cpkm": None,
            "normalised_cpkm": None,
            "country_mix": None,
            "corridor_mix": None,
            "distance_band_mix": None,
            "business_flow_mix": None,
            "equipment_type_mix": None,
            "mix_impact": None,
        }

    m = seven_metrics
    dist = compare_total_distance

    # Per-km values
    base_spend_cpkm = m["base_spend"] / dist
    rate_cpkm = (m["rate_cost"] - m["base_spend"]) / dist
    country_cpkm = (m["country_cost"] - m["rate_cost"]) / dist
    corridor_cpkm = (m["corridor_cost"] - m["country_cost"]) / dist
    distance_band_cpkm = (m["distance_band_cost"] - m["corridor_cost"]) / dist
    business_flow_cpkm = (m["business_flow_cost"] - m["distance_band_cost"]) / dist
    equipment_type_cpkm = (m["equipment_type_cost"] - m["business_flow_cost"]) / dist

    result = {
        "base_cpkm": base_spend_cpkm,
        "country_mix": None,
        "corridor_mix": None,
        "distance_band_mix": None,
        "business_flow_mix": None,
        "equipment_type_mix": None,
    }

    mix_impact = 0.0

    # Country mix: included at EU level, not at country level
    if aggregation_level == "eu":
        result["country_mix"] = country_cpkm
        mix_impact += country_cpkm

    # Corridor mix: always included
    result["corridor_mix"] = corridor_cpkm
    mix_impact += corridor_cpkm

    # Distance band mix: always included
    result["distance_band_mix"] = distance_band_cpkm
    mix_impact += distance_band_cpkm

    # Business flow mix: included at total level, not at business level
    if bridge_level == "total":
        result["business_flow_mix"] = business_flow_cpkm
        mix_impact += business_flow_cpkm

    # Equipment type mix: always included
    result["equipment_type_mix"] = equipment_type_cpkm
    mix_impact += equipment_type_cpkm

    result["mix_impact"] = mix_impact
    # normalised_cpkm = base_cpkm + all applicable mix components
    result["normalised_cpkm"] = base_spend_cpkm + mix_impact

    return result


def compute_cell_cpkm(
    data_df: pd.DataFrame,
    distance_col: str = "distance_for_cpkm",
    cost_col: str = "total_cost_usd",
) -> pd.DataFrame:
    """
    Compute CPKM at cell level (full grain).

    Args:
        data_df: DataFrame with grain columns, distance, and cost.
        distance_col: Distance column name.
        cost_col: Cost column name.

    Returns:
        DataFrame with GRAIN_COLS + ['cpkm']
    """
    if "is_set" not in data_df.columns:
        data_df = data_df.copy()
        data_df["is_set"] = False

    grouped = data_df.groupby(GRAIN_COLS, as_index=False).agg(
        distance=(distance_col, "sum"),
        cost=(cost_col, "sum"),
    )

    grouped["cpkm"] = np.where(
        grouped["distance"] > 0,
        grouped["cost"] / grouped["distance"],
        0.0,
    )

    return grouped[GRAIN_COLS + ["cpkm"]]


def compute_op2_cell_cpkm(
    op2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cell-level CPKM from OP2 data.

    OP2 data has no is_set column; we set is_set=False for all rows
    (RET=100%, SET=0% means all OP2 rates apply to non-SET).

    Expects op2_df to have columns:
        orig_country, dest_country, distance_band, business, op2_cpkm, distance

    Returns:
        DataFrame with GRAIN_COLS + ['cpkm']
    """
    df = op2_df.copy()
    df["is_set"] = False

    # OP2 already has rates per cell; just aggregate in case of duplicates
    grouped = df.groupby(GRAIN_COLS, as_index=False).agg(
        cpkm=("op2_cpkm", "mean"),
    )

    return grouped[GRAIN_COLS + ["cpkm"]]
