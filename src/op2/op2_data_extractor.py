"""
OP2 Data Extractor

Functions for extracting and transforming OP2 benchmark data from raw OP2 tables.
"""

import pandas as pd


def extract_op2_weekly_base_cpkm(df_op2: pd.DataFrame) -> pd.DataFrame:
    """
    Extract OP2 base weekly CPKM from weekly_agg bridge type.

    Reads aggregated OP2 weekly metrics including CPKM, distance, cost,
    loads, and active carriers.

    Input Table - df_op2 (raw OP2 data):
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | Bridge type     | string | Must be 'weekly_agg' for filter |
        | Report Year     | string | Year in R20XX format            |
        | Week            | string | Week in WXX format              |
        | Orig_EU5        | string | Origin country code             |
        | CpKM            | float  | Cost per kilometer              |
        | Cost            | float  | Total cost                      |
        | Distance        | float  | Total distance                  |
        | Loads           | int    | Number of loads                 |
        | Active Carriers | int    | Number of active carriers       |
        | LCR             | float  | Load-to-Carrier Ratio           |

    Output Table:
        | Column            | Type   | Description                   |
        |-------------------|--------|-------------------------------|
        | report_year       | string | Year in R20XX format          |
        | report_week       | string | Week in WXX format            |
        | orig_country      | string | Origin country code           |
        | op2_base_cpkm     | float  | OP2 baseline CPKM             |
        | op2_base_cost     | float  | OP2 baseline total cost       |
        | op2_base_distance | float  | OP2 baseline distance         |
        | op2_base_loads    | int    | OP2 baseline loads            |
        | op2_carriers      | int    | OP2 active carriers           |
        | op2_lcr           | float  | OP2 Load-to-Carrier Ratio     |
    """
    base = df_op2[df_op2["Bridge type"] == "weekly_agg"].copy()

    base = base.rename(columns={
        "Report Year": "report_year",
        "Week": "report_week",
        "Orig_EU5": "orig_country",
        "CpKM": "op2_base_cpkm",
        "Cost": "op2_base_cost",
        "Distance": "op2_base_distance",
        "Loads": "op2_base_loads",
        "Active Carriers": "op2_carriers",
        "LCR": "op2_lcr",
    })

    return base[[
        "report_year",
        "report_week",
        "orig_country",
        "op2_base_cpkm",
        "op2_base_cost",
        "op2_base_distance",
        "op2_base_loads",
        "op2_carriers",
        "op2_lcr",
    ]]


def extract_op2_weekly_base_by_business(df_op2: pd.DataFrame) -> pd.DataFrame:
    """
    Extract OP2 weekly base metrics broken down by business.

    Aggregates OP2 weekly detailed data to the business level and
    joins carrier information from the aggregated table.

    Input Table - df_op2 (raw OP2 data):
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | Bridge type     | string | 'weekly' for detailed data      |
        | Report Year     | string | Year in R20XX format            |
        | Week            | string | Week in WXX format              |
        | Orig_EU5        | string | Origin country code             |
        | Business Flow   | string | Business unit identifier        |
        | Distance Band   | string | Distance Band                   |
        | Distance        | float  | Distance in kilometers          |
        | Cost            | float  | Total cost                      |
        | Loads           | int    | Number of loads                 |

    Output Table:
        | Column            | Type   | Description                   |
        |-------------------|--------|-------------------------------|
        | report_year       | string | Year in R20XX format          |
        | report_week       | string | Week in WXX format            |
        | orig_country      | string | Origin country code           |
        | business          | string | Business unit (uppercase)     |
        | op2_base_distance | float  | OP2 baseline distance         |
        | op2_base_cost     | float  | OP2 baseline cost             |
        | op2_base_loads    | int    | OP2 baseline loads            |
        | op2_base_cpkm     | float  | OP2 baseline CPKM             |
        | op2_carriers      | int    | OP2 active carriers           |
    """
    # Filter OP2 weekly detailed data
    op2 = df_op2[df_op2["Bridge type"] == "weekly"].copy()

    op2 = op2.rename(columns={
        "Report Year": "report_year",
        "Week": "report_week",
        "Orig_EU5": "orig_country",
        "Dest_EU5": "dest_country",
        "Business Flow": "business",
        "Distance Band": "distance_band",
        "Distance": "op2_distance",
        "Cost": "op2_cost",
        "Loads": "op2_base_loads",
    })

    # Normalize types
    op2["report_year"] = op2["report_year"].astype(str)
    op2["report_week"] = op2["report_week"].astype(str)
    op2["orig_country"] = op2["orig_country"].astype(str)
    op2["dest_country"] = op2["dest_country"].astype(str)
    op2["business"] = op2["business"].astype(str).str.upper()
    op2["distance_band"] = op2["distance_band"].astype(str)

    # Aggregate to business level
    out = op2.groupby(
        ["report_year", "report_week", "orig_country", "business"],
        as_index=False,
    ).agg(
        op2_base_distance=("op2_distance", "sum"),
        op2_base_cost=("op2_cost", "sum"),
        op2_base_loads=("op2_base_loads", "sum"),
    )

    out["op2_base_cpkm"] = out["op2_base_cost"] / out["op2_base_distance"]

    # Join carrier information from aggregated table
    carriers = extract_op2_weekly_base_cpkm(df_op2)
    out = out.merge(
        carriers[["report_year", "report_week", "orig_country", "op2_carriers"]],
        on=["report_year", "report_week", "orig_country"],
        how="inner",
    )

    return out


def extract_op2_monthly_base_cpkm(df_op2: pd.DataFrame) -> pd.DataFrame:
    """
    Extract OP2 base monthly CPKM from monthly_agg bridge type.

    Reads aggregated OP2 monthly metrics for monthly benchmarking.

    Input Table - df_op2 (raw OP2 data):
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | Bridge type     | string | Must be 'monthly_agg'           |
        | Report Year     | string | Year in R20XX format            |
        | Period          | string | Month in MXX format             |
        | Orig_EU5        | string | Origin country code             |
        | CpKM            | float  | Cost per kilometer              |
        | Cost            | float  | Total cost                      |
        | Distance        | float  | Total distance                  |
        | Loads           | int    | Number of loads                 |
        | Active Carriers | int    | Number of active carriers       |
        | LCR             | float  | Load-to-Carrier Ratio           |

    Output Table:
        | Column            | Type   | Description                   |
        |-------------------|--------|-------------------------------|
        | report_year       | string | Year in R20XX format          |
        | report_month      | string | Month in MXX format           |
        | orig_country      | string | Origin country code           |
        | op2_base_cpkm     | float  | OP2 baseline CPKM             |
        | op2_base_cost     | float  | OP2 baseline cost             |
        | op2_base_distance | float  | OP2 baseline distance         |
        | op2_base_loads    | int    | OP2 baseline loads            |
        | op2_carriers      | int    | OP2 active carriers           |
        | op2_lcr           | float  | OP2 Load-to-Carrier Ratio     |
    """
    base = df_op2[df_op2["Bridge type"] == "monthly_agg"].copy()

    base = base.rename(columns={
        "Report Year": "report_year",
        "Period": "report_month",
        "Orig_EU5": "orig_country",
        "CpKM": "op2_base_cpkm",
        "Cost": "op2_base_cost",
        "Distance": "op2_base_distance",
        "Loads": "op2_base_loads",
        "Active Carriers": "op2_carriers",
        "LCR": "op2_lcr",
    })

    return base[[
        "report_year",
        "report_month",
        "orig_country",
        "op2_base_cpkm",
        "op2_base_cost",
        "op2_base_distance",
        "op2_base_loads",
        "op2_carriers",
        "op2_lcr",
    ]]


def extract_op2_monthly_base_by_business(df_op2: pd.DataFrame) -> pd.DataFrame:
    """
    Extract OP2 monthly base metrics broken down by business.

    Aggregates OP2 monthly detailed data to the business level and
    joins carrier information from the aggregated table.

    Input Table - df_op2 (raw OP2 data):
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | Bridge type     | string | 'monthly_bridge' for detailed   |
        | Report Year     | string | Year in R20XX format            |
        | Report Month    | string | Month in MXX format             |
        | Orig_EU5        | string | Origin country code             |
        | Business Flow   | string | Business unit identifier        |
        | Distance Band   | string | Distance Band                   |
        | Distance        | float  | Distance in kilometers          |
        | Cost            | float  | Total cost                      |
        | Loads           | int    | Number of loads                 |

    Output Table:
        | Column            | Type   | Description                   |
        |-------------------|--------|-------------------------------|
        | report_year       | string | Year in R20XX format          |
        | report_month      | string | Month in MXX format           |
        | orig_country      | string | Origin country code           |
        | business          | string | Business unit (uppercase)     |
        | op2_base_distance | float  | OP2 baseline distance         |
        | op2_base_cost     | float  | OP2 baseline cost             |
        | op2_base_loads    | int    | OP2 baseline loads            |
        | op2_base_cpkm     | float  | OP2 baseline CPKM             |
        | op2_carriers      | int    | OP2 active carriers           |
    """
    # Filter OP2 monthly detailed data
    op2 = df_op2[df_op2["Bridge type"] == "monthly_bridge"].copy()

    op2 = op2.rename(columns={
        "Report Year": "report_year",
        "Report Month": "report_month",
        "Orig_EU5": "orig_country",
        "Dest_EU5": "dest_country",
        "Business Flow": "business",
        "Distance Band": "distance_band",
        "Distance": "op2_distance",
        "Cost": "op2_cost",
        "Loads": "op2_base_loads",
    })

    # Normalize types
    op2["report_year"] = op2["report_year"].astype(str)
    op2["report_month"] = op2["report_month"].astype(str)
    op2["orig_country"] = op2["orig_country"].astype(str)
    op2["dest_country"] = op2["dest_country"].astype(str)
    op2["business"] = op2["business"].astype(str).str.upper()
    op2["distance_band"] = op2["distance_band"].astype(str)

    # Aggregate to business level
    out = op2.groupby(
        ["report_year", "report_month", "orig_country", "business"],
        as_index=False,
    ).agg(
        op2_base_distance=("op2_distance", "sum"),
        op2_base_cost=("op2_cost", "sum"),
        op2_base_loads=("op2_base_loads", "sum"),
    )

    out["op2_base_cpkm"] = out["op2_base_cost"] / out["op2_base_distance"]

    # Join carrier information from aggregated table
    carriers = extract_op2_monthly_base_cpkm(df_op2)
    out = out.merge(
        carriers[["report_year", "report_month", "orig_country", "op2_carriers"]],
        on=["report_year", "report_month", "orig_country"],
        how="inner",
    )

    return out


def extract_op2_quarterly_base_cpkm(df_op2: pd.DataFrame) -> pd.DataFrame:
    """
    Extract OP2 base quarterly CPKM from monthly_agg bridge type.

    Reads aggregated OP2 quarterly metrics (Period = Q1, Q2, Q3, Q4).
    Only available at country and EU level, not at business level.

    Input Table - df_op2 (raw OP2 data):
        | Column          | Type   | Description                     |
        |-----------------|--------|---------------------------------|
        | Bridge type     | string | Must be 'monthly_agg'           |
        | Report Year     | string | Year in R20XX format            |
        | Period          | string | Quarter in Q1-Q4 format         |
        | Orig_EU5        | string | Origin country code             |
        | CpKM            | float  | Cost per kilometer              |
        | Cost            | float  | Total cost                      |
        | Distance        | float  | Total distance                  |
        | Loads           | int    | Number of loads                 |
        | Active Carriers | int    | Number of active carriers       |
        | LCR             | float  | Load-to-Carrier Ratio           |

    Output Table:
        | Column            | Type   | Description                   |
        |-------------------|--------|-------------------------------|
        | report_year       | string | Year in R20XX format          |
        | report_quarter    | string | Quarter in Q1-Q4 format       |
        | orig_country      | string | Origin country code           |
        | op2_base_cpkm     | float  | OP2 baseline CPKM             |
        | op2_base_cost     | float  | OP2 baseline cost             |
        | op2_base_distance | float  | OP2 baseline distance         |
        | op2_base_loads    | int    | OP2 baseline loads            |
        | op2_carriers      | int    | OP2 active carriers           |
        | op2_lcr           | float  | OP2 Load-to-Carrier Ratio     |
    """
    # Filter for quarterly periods (Q1, Q2, Q3, Q4)
    base = df_op2[
        (df_op2["Bridge type"] == "monthly_agg")
        & (df_op2["Period"].isin(["Q1", "Q2", "Q3", "Q4"]))
    ].copy()

    base = base.rename(columns={
        "Report Year": "report_year",
        "Period": "report_quarter",
        "Orig_EU5": "orig_country",
        "CpKM": "op2_base_cpkm",
        "Cost": "op2_base_cost",
        "Distance": "op2_base_distance",
        "Loads": "op2_base_loads",
        "Active Carriers": "op2_carriers",
        "LCR": "op2_lcr",
    })

    return base[[
        "report_year",
        "report_quarter",
        "orig_country",
        "op2_base_cpkm",
        "op2_base_cost",
        "op2_base_distance",
        "op2_base_loads",
        "op2_carriers",
        "op2_lcr",
    ]]
