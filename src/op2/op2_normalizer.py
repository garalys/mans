"""
OP2 Normalized CPKM Calculator

Applies OP2 rates to actual volumes to calculate normalized benchmarks.
"""

import pandas as pd

from ..config.logging_config import logger


def compute_op2_normalized_cpkm_weekly(
    actual_df: pd.DataFrame,
    df_op2: pd.DataFrame,
    by_business: bool = False,
) -> pd.DataFrame:
    """
    Apply OP2 weekly rates to actual weekly volumes.

    Calculates what the actual period's cost would have been at OP2 rates,
    normalized by actual distance.

    Input Table - actual_df (actual transportation data):
        | Column           | Type   | Description                    |
        |------------------|--------|--------------------------------|
        | report_year      | string | Year in R20XX format           |
        | report_week      | string | Week in WXX format             |
        | orig_country     | string | Origin country code            |
        | dest_country     | string | Destination country code       |
        | business         | string | Business unit                  |
        | distance_band    | string | Distance category              |
        | distance_for_cpkm| float  | Distance in kilometers         |

    Input Table - df_op2 (OP2 benchmark data):
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | Bridge type     | string | Must be 'weekly'                |
        | Report Year     | string | Year in R20XX format            |
        | Week            | string | Week in WXX format              |
        | Orig_EU5        | string | Origin country code             |
        | Dest_EU5        | string | Destination country code        |
        | Business Flow   | string | Business unit                   |
        | Distance Band   | string | Distance category               |
        | CpKM            | float  | OP2 cost per kilometer          |

    Args:
        actual_df: Actual transportation data
        df_op2: OP2 benchmark reference data
        by_business: If True, return results at business level

    Output Table (by_business=False):
        | Column               | Type   | Description                  |
        |----------------------|--------|------------------------------|
        | report_year          | string | Year in R20XX format         |
        | report_week          | string | Week in WXX format           |
        | orig_country         | string | Origin country code          |
        | op2_normalized_cost  | float  | Total normalized cost        |
        | op2_normalized_cpkm  | float  | Normalized CPKM              |

    Output Table (by_business=True):
        Same as above plus:
        | business             | string | Business unit                |
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
        "CpKM": "op2_cpkm",
    })

    # Normalize types
    for col in ["report_year", "report_week", "orig_country", "dest_country", "business", "distance_band"]:
        op2[col] = op2[col].astype(str)
    op2["business"] = op2["business"].str.upper()

    # Prepare actual data
    actual_df = actual_df.copy()
    actual_df["business"] = actual_df["business"].str.upper()
    actual_df["report_year"] = actual_df["report_year"].astype(str)
    actual_df["distance_band"] = (
        actual_df["distance_band"]
        .astype(str)
        .str.strip()
        .str.replace(r"^\d+\.", "", regex=True)
    )

    # Aggregate actual weekly volumes
    group_cols = [
        "report_year", "report_week", "orig_country",
        "dest_country", "business", "distance_band",
    ]
    actual = actual_df.groupby(group_cols, as_index=False).agg(
        actual_distance=("distance_for_cpkm", "sum")
    )

    # Join OP2 rates to actual volumes
    merged = actual.merge(
        op2[group_cols + ["op2_cpkm"]],
        on=group_cols,
        how="inner",
    )

    # Calculate normalized cost
    merged["normalized_cost"] = merged["op2_cpkm"] * merged["actual_distance"]

    # Aggregate to output grain
    if by_business:
        out = merged.groupby(
            ["report_year", "report_week", "orig_country", "business"],
            as_index=False,
        ).agg(
            op2_normalized_cost=("normalized_cost", "sum"),
            norm_distance=("actual_distance", "sum"),  # Renamed to avoid conflicts
        )
        out["op2_normalized_cpkm"] = out["op2_normalized_cost"] / out["norm_distance"]
        logger.info(out[[ "report_week", "orig_country", "business", "op2_normalized_cost","norm_distance"]].head())
        logger.info(f"sum op2 norm: {out['op2_normalized_cost'].sum()}")
        # Explicitly select and copy to avoid any column leakage
        result = out[["report_year", "report_week", "orig_country", "business", "op2_normalized_cost", "op2_normalized_cpkm"]].copy()
        logger.info(f"op2_norm returning columns: {result.columns.tolist()}")
        return result
    else:
        out = merged.groupby(
            ["report_year", "report_week", "orig_country"],
            as_index=False,
        ).agg(
            op2_normalized_cost=("normalized_cost", "sum"),
            norm_distance=("actual_distance", "sum"),  # Renamed to avoid conflicts
        )
        out["op2_normalized_cpkm"] = out["op2_normalized_cost"] / out["norm_distance"]
        logger.info(f"sum op2 norm: {out['op2_normalized_cpkm'].sum()}")
        # Explicitly select and copy to avoid any column leakage
        result = out[["report_year", "report_week", "orig_country", "op2_normalized_cost", "op2_normalized_cpkm"]].copy()
        logger.info(f"op2_norm returning columns: {result.columns.tolist()}")
        return result


def compute_op2_normalized_cpkm_monthly(
    actual_df: pd.DataFrame,
    df_op2: pd.DataFrame,
    by_business: bool = False,
) -> pd.DataFrame:
    """
    Apply OP2 monthly rates to actual monthly volumes.

    Input Table - actual_df:
        Same as weekly but with report_month instead of report_week

    Input Table - df_op2:
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | Bridge type     | string | Must be 'monthly_bridge'        |
        | Report Year     | string | Year in R20XX format            |
        | Report Month    | string | Month in MXX format             |
        | + same dimensional columns as weekly                        |

    Args:
        actual_df: Actual transportation data
        df_op2: OP2 benchmark reference data
        by_business: If True, return results at business level

    Output Table (by_business=False):
        | Column               | Type   | Description                  |
        |----------------------|--------|------------------------------|
        | report_year          | string | Year in R20XX format         |
        | report_month         | string | Month in MXX format          |
        | orig_country         | string | Origin country code          |
        | op2_normalized_cost  | float  | Total normalized cost        |
        | op2_normalized_cpkm  | float  | Normalized CPKM              |

    Output Table (by_business=True):
        Same as above plus:
        | business             | string | Business unit                |
    """
    # Prepare OP2 monthly data
    op2 = df_op2[df_op2["Bridge type"] == "monthly_bridge"].copy()

    op2 = op2.rename(columns={
        "Report Year": "report_year",
        "Report Month": "report_month",
        "Orig_EU5": "orig_country",
        "Dest_EU5": "dest_country",
        "Business Flow": "business",
        "Distance Band": "distance_band",
        "CpKM": "op2_cpkm",
    })

    # Normalize types
    for col in ["report_year", "report_month", "orig_country", "dest_country", "distance_band"]:
        op2[col] = op2[col].astype(str)
    op2["business"] = op2["business"].str.upper()

    # Prepare actual data
    actual_df = actual_df.copy()
    actual_df["report_year"] = actual_df["report_year"].astype(str)
    actual_df["report_month"] = actual_df["report_month"].astype(str)
    actual_df["business"] = actual_df["business"].str.upper()
    actual_df["distance_band"] = (
        actual_df["distance_band"]
        .astype(str)
        .str.strip()
        .str.replace(r"^\d+\.", "", regex=True)
    )

    # Aggregate actual monthly volumes
    group_cols = [
        "report_year", "report_month", "orig_country",
        "dest_country", "business", "distance_band",
    ]
    actual = actual_df.groupby(group_cols, as_index=False).agg(
        actual_distance=("distance_for_cpkm", "sum")
    )

    # Join OP2 rates to actual volumes
    merged = actual.merge(
        op2[group_cols + ["op2_cpkm"]],
        on=group_cols,
        how="inner",
    )

    # Calculate normalized cost
    merged["normalized_cost"] = merged["op2_cpkm"] * merged["actual_distance"]

    # Aggregate to output grain
    if by_business:
        out = merged.groupby(
            ["report_year", "report_month", "orig_country", "business"],
            as_index=False,
        ).agg(
            op2_normalized_cost=("normalized_cost", "sum"),
            norm_distance=("actual_distance", "sum"),
        )
        out["op2_normalized_cpkm"] = out["op2_normalized_cost"] / out["norm_distance"]
        return out[["report_year", "report_month", "orig_country", "business", "op2_normalized_cost", "op2_normalized_cpkm"]]
    else:
        out = merged.groupby(
            ["report_year", "report_month", "orig_country"],
            as_index=False,
        ).agg(
            op2_normalized_cost=("normalized_cost", "sum"),
            actual_distance=("actual_distance", "sum"),
        )
        out["op2_normalized_cpkm"] = out["op2_normalized_cost"] / out["actual_distance"]
        return out[["report_year", "report_month", "orig_country", "op2_normalized_cost", "op2_normalized_cpkm"]]
