"""
S3 Data Loader

Functions for loading data from AWS S3 buckets.
"""

import io
import boto3
import pandas as pd
from typing import Tuple

from ..config.logging_config import logger


def get_s3_client():
    """
    Initialize and return an S3 client.

    Returns:
        boto3 S3 client instance
    """
    return boto3.client("s3")


def load_source_data(
    source_bucket: str,
    source_key: str,
    carrier_scaling_bucket: str,
    carrier_scaling_key: str,
    op2_bucket: str,
    op2_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess input data from S3.

    Input Table - Main Data (source_bucket/source_key):
        | Column           | Type     | Description                           |
        |------------------|----------|---------------------------------------|
        | report_day       | date     | Date of the report                    |
        | report_year      | string   | Year in R20XX format (e.g., 'R2025')  |
        | report_month     | string   | Month in MXX format (e.g., 'M01')     |
        | report_week      | string   | Week in WXX format (e.g., 'W01')      |
        | orig_country     | string   | Origin country code (DE, ES, FR, etc.)|
        | dest_country     | string   | Destination country code              |
        | business         | string   | Business unit identifier              |
        | distance_band    | string   | Distance category                     |
        | is_set           | boolean  | Whether load uses SET pricing         |
        | distance_for_cpkm| float    | Distance in kilometers                |
        | total_cost_usd   | float    | Total cost in USD                     |
        | executed_loads   | int      | Number of executed loads              |
        | vehicle_carrier  | string   | Carrier identifier                    |
        | transporeon_contract_price_eur | float | Market rate price in EUR     |

    Input Table - Carrier Scaling (carrier_scaling_bucket/carrier_scaling_key):
        | Column     | Type   | Description                              |
        |------------|--------|------------------------------------------|
        | year       | string | Year in R20XX format                     |
        | period     | string | Period (W01, M01, etc.)                  |
        | country    | string | Country code                             |
        | percentage | float  | Carrier scaling percentage adjustment    |

    Input Table - OP2 Data (op2_bucket/op2_key):
        | Column        | Type   | Description                           |
        |---------------|--------|---------------------------------------|
        | Bridge type   | string | Type: 'weekly', 'weekly_agg', etc.    |
        | Report Year   | string | Year in R20XX format                  |
        | Week          | string | Week in WXX format                    |
        | Orig_EU5      | string | Origin country                        |
        | Dest_EU5      | string | Destination country                   |
        | Business Flow | string | Business unit                         |
        | Distance Band | string | Distance category                     |
        | CpKM          | float  | Cost per kilometer                    |
        | Distance      | float  | Distance in kilometers                |
        | Cost          | float  | Total cost                            |
        | Loads         | int    | Number of loads                       |
        | Active Carriers| int   | Number of active carriers             |

    Output:
        Tuple of three DataFrames:
        - df: Main transportation data with report_day converted to datetime
        - df_carrier: Carrier scaling percentages
        - df_op2: OP2 benchmark data

    Raises:
        ValueError: If required columns are missing from main data
        Exception: If S3 read fails
    """
    s3_client = get_s3_client()

    try:
        # Load main data
        response = s3_client.get_object(Bucket=source_bucket, Key=source_key)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))

        # Validate required columns
        required_columns = [
            "report_day",
            "report_year",
            "report_month",
            "report_week",
            "orig_country",
            "business",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert report_day to datetime
        try:
            df["report_day"] = pd.to_datetime(df["report_day"])
        except Exception as e:
            logger.error(f"Error converting report_day to datetime: {str(e)}")
            logger.info(f"Sample of report_day values: {df['report_day'].head()}")
            raise

        # Load carrier scaling data
        response_carrier = s3_client.get_object(
            Bucket=carrier_scaling_bucket, Key=carrier_scaling_key
        )
        df_carrier = pd.read_csv(io.BytesIO(response_carrier["Body"].read()))

        # Load OP2 data
        op2_response = s3_client.get_object(Bucket=op2_bucket, Key=op2_key)
        df_op2 = pd.read_csv(io.BytesIO(op2_response["Body"].read()))

        return df, df_carrier, df_op2

    except Exception as e:
        logger.error(f"Error loading data from S3: {str(e)}")
        raise
