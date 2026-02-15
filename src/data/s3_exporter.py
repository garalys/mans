"""
S3 Data Exporter

Functions for exporting processed data to AWS S3 for QuickSight visualization.
"""

import io
import pandas as pd

from .s3_loader import get_s3_client
from ..config.logging_config import logger


def export_to_quicksight(
    bridge_df: pd.DataFrame,
    destination_bucket: str,
    destination_key: str,
) -> None:
    """
    Export bridge data to S3 in a format optimized for QuickSight visualization.

    Input Table - Bridge DataFrame:
        | Column                    | Type   | Description                              |
        |---------------------------|--------|------------------------------------------|
        | report_year               | string | Year in R20XX format                     |
        | report_week               | string | Week in WXX format (for YoY/WoW)         |
        | report_month              | string | Month in MXX format (for MTD)            |
        | orig_country              | string | Origin country code or 'EU'              |
        | business                  | string | Business unit or 'Total'                 |
        | bridge_type               | string | 'YoY', 'WoW', 'MTD', 'op2_weekly', etc.  |
        | bridging_value            | string | Period comparison identifier             |
        | base_cpkm                 | float  | Base period cost per kilometer           |
        | compare_cpkm              | float  | Comparison period cost per kilometer     |
        | mix_impact                | float  | Impact from mix changes ($/km)           |
        | country_mix               | float  | Country mix impact ($/km) - EU only      |
        | corridor_mix              | float  | Corridor mix impact ($/km)               |
        | distance_band_mix         | float  | Distance band mix impact ($/km)          |
        | business_flow_mix         | float  | Business flow mix impact ($/km)          |
        | equipment_type_mix        | float  | Equipment type mix impact ($/km)         |
        | normalised_cpkm           | float  | Normalized cost per kilometer            |
        | carrier_impact            | float  | Impact from carrier changes ($/km)       |
        | demand_impact             | float  | Impact from demand changes ($/km)        |
        | carrier_and_demand_impact | float  | Combined carrier + demand impact ($/km)  |
        | market_rate_impact        | float  | Impact from market rate changes ($/km)   |
        | tech_impact               | float  | Impact from tech initiatives ($/km)      |
        | premium_impact            | float  | Impact from premium pricing ($/km)       |
        | *_impact_mm               | float  | Impacts in millions USD                  |
        | base_distance_km          | float  | Base period distance                     |
        | compare_distance_km       | float  | Comparison period distance               |
        | base_costs_usd            | float  | Base period total cost                   |
        | compare_costs_usd         | float  | Comparison period total cost             |
        | base_loads                | int    | Base period load count                   |
        | compare_loads             | int    | Comparison period load count             |
        | base_carriers             | int    | Base period active carriers              |
        | compare_carriers          | int    | Comparison period active carriers        |

    Output:
        CSV file written to S3 at destination_bucket/destination_key

    Args:
        bridge_df: DataFrame containing bridge analysis results
        destination_bucket: S3 bucket name for output
        destination_key: S3 key (path) for output file

    Raises:
        Exception: If S3 upload fails
    """
    s3_client = get_s3_client()

    try:
        csv_buffer = io.StringIO()
        bridge_df.to_csv(csv_buffer, index=False)

        s3_client.put_object(
            Bucket=destination_bucket,
            Key=destination_key,
            Body=csv_buffer.getvalue(),
        )

        logger.info(f"Successfully exported {len(bridge_df)} rows to s3://{destination_bucket}/{destination_key}")

    except Exception as e:
        logger.error(f"Error exporting to S3: {str(e)}")
        raise
