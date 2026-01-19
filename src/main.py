"""
Transportation Cost Analytics - Main Entry Point

This module orchestrates the bridge analysis pipeline, calculating
YoY, WoW, MTD, and OP2 benchmark bridges for transportation cost analysis.

Usage:
    python -m src.main

Environment Variables Required:
    SOURCE_BUCKET: S3 bucket containing source data
    SOURCE_KEY: S3 key for source data file
    CARRIER_SCALING_BUCKET: S3 bucket for carrier scaling data
    CARRIER_SCALING_KEY: S3 key for carrier scaling file
    OP2_BUCKET: S3 bucket for OP2 benchmark data
    OP2_KEY: S3 key for OP2 file
    DESTINATION_BUCKET: S3 bucket for output
    DESTINATION_KEY: S3 key for output file
    + Country coefficients (see config/settings.py)
    + Tech savings rates (TECH_SAVINGS_RATE_*)
"""

import os
import time
import pandas as pd
from datetime import datetime

from .config.logging_config import logger
from .data.s3_loader import load_source_data
from .data.s3_exporter import export_to_quicksight
from .bridges.bridge_builder import create_bridge_structure, create_bridge_structure_for_totals
from .bridges.yoy_bridge import calculate_yoy_bridge_metrics
from .bridges.wow_bridge import calculate_wow_bridge_metrics
from .bridges.mtd_bridge import calculate_mtd_bridge_metrics
from .bridges.eu_aggregation import calculate_eu_aggregated_metrics
from .bridges.total_aggregation import calculate_total_aggregated_metrics
from .bridges.impact_adjuster import adjust_carrier_demand_impacts
from .op2.op2_weekly_bridge import create_op2_weekly_bridge, create_op2_weekly_country_business_bridge
from .op2.op2_eu_aggregation import create_op2_eu_weekly_bridge, create_op2_eu_weekly_business_bridge
from .op2.op2_impact_adjuster import adjust_op2_carrier_demand_impacts
from .utils.date_utils import extract_year_from_report_year


def run_analytics_pipeline(
    source_bucket: str,
    source_key: str,
    carrier_scaling_bucket: str,
    carrier_scaling_key: str,
    op2_bucket: str,
    op2_key: str,
    destination_bucket: str,
    destination_key: str,
) -> dict:
    """
    Execute the full analytics pipeline.

    Pipeline Steps:
        1. Load data from S3 (source, carrier scaling, OP2)
        2. Create bridge structures for all combinations
        3. Calculate YoY bridge metrics
        4. Calculate WoW bridge metrics
        5. Calculate MTD bridge metrics
        6. Calculate EU aggregated metrics
        7. Calculate Total business aggregations
        8. Adjust carrier/demand impacts
        9. Create OP2 weekly bridges (country-total, country-business)
        10. Create OP2 EU aggregations (EU-total, EU-business)
        11. Export results to S3

    Input:
        S3 paths for source data, carrier scaling, OP2 benchmarks

    Output:
        dict: Processing summary including:
            - message: Status message
            - rows_processed: Number of output rows
            - years_processed: Number of years in data
            - source_file: Source S3 path
            - destination_file: Output S3 path
            - memory_usage_mb: Final DataFrame memory usage
    """
    # Step 1: Load data from S3
    logger.info("Loading and processing data from S3...")
    start_time = time.time()

    df, df_carrier, df_op2 = load_source_data(
        source_bucket, source_key,
        carrier_scaling_bucket, carrier_scaling_key,
        op2_bucket, op2_key,
    )

    # Filter to relevant years and weeks (configurable)
    df = df[df["report_year"].isin(["R2025", "R2026"])]
    df = df[df["report_week"].isin(["W01", "W02", "W03", "W04", "W05"])]

    # Clean carrier data
    df_carrier.columns = df_carrier.columns.str.strip()
    df_carrier["percentage"] = pd.to_numeric(df_carrier["percentage"], errors="coerce")

    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"Loading from S3 took: {time.time() - start_time:.2f} seconds")

    # Step 2: Create bridge structures
    logger.info("Creating bridge structure...")
    start_time = time.time()
    bridge_df = create_bridge_structure(df)
    logger.info(f"Bridge structure creation took: {time.time() - start_time:.2f} seconds")

    logger.info("Creating total bridge structure...")
    start_time = time.time()
    total_bridge_df = create_bridge_structure_for_totals(df)
    logger.info(f"Total bridge structure creation took: {time.time() - start_time:.2f} seconds")

    # Step 3: Calculate YoY metrics
    logger.info("Calculating YoY metrics...")
    start_time = time.time()
    years = sorted(df["report_year"].unique())

    for i in range(len(years) - 1):
        base_year = years[i]
        compare_year = years[i + 1]
        logger.info(f"Processing year pair {base_year}-{compare_year}")
        calculate_yoy_bridge_metrics(df, bridge_df, base_year, compare_year, df_carrier)

    logger.info(f"YoY calculations took: {time.time() - start_time:.2f} seconds")

    # Step 4: Calculate WoW metrics
    logger.info("Calculating WoW metrics...")
    start_time = time.time()
    calculate_wow_bridge_metrics(df, bridge_df, df_carrier)
    logger.info(f"WoW calculations took: {time.time() - start_time:.2f} seconds")

    # Step 5: Calculate MTD metrics
    logger.info("Calculating MTD metrics...")
    start_time = time.time()
    calculate_mtd_bridge_metrics(df, bridge_df, df_carrier)
    logger.info(f"MTD calculations took: {time.time() - start_time:.2f} seconds")

    # Step 6: Calculate EU aggregated metrics
    logger.info("Calculating EU aggregated metrics...")
    start_time = time.time()
    calculate_eu_aggregated_metrics(df, bridge_df, df_carrier)
    logger.info(f"EU aggregated calculations took: {time.time() - start_time:.2f} seconds")

    # Step 7: Calculate total metrics
    logger.info("Calculating total metrics...")
    start_time = time.time()
    calculate_total_aggregated_metrics(df, total_bridge_df, df_carrier)
    logger.info(f"Total metrics calculations took: {time.time() - start_time:.2f} seconds")

    # Step 8: Combine and adjust impacts
    logger.info("Combining and adjusting final results...")
    final_bridge_df = pd.concat([bridge_df, total_bridge_df], ignore_index=True)
    final_bridge_df = adjust_carrier_demand_impacts(final_bridge_df)

    # Step 9: Create OP2 weekly bridges
    logger.info("Creating OP2 weekly COUNTRY-TOTAL bridge...")
    start_time = time.time()

    op2_weekly_country_total_df = create_op2_weekly_bridge(
        df=df,
        df_op2=df_op2,
        df_carrier=df_carrier,
        final_bridge_df=final_bridge_df,
    )

    logger.info(f"OP2 weekly COUNTRY-TOTAL rows: {len(op2_weekly_country_total_df)}")
    final_bridge_df = pd.concat([final_bridge_df, op2_weekly_country_total_df], ignore_index=True)
    logger.info(f"OP2 weekly COUNTRY-TOTAL creation took: {time.time() - start_time:.2f} seconds")

    logger.info("Creating OP2 weekly COUNTRY-BUSINESS bridge...")
    start_time = time.time()

    op2_weekly_country_business_df = create_op2_weekly_country_business_bridge(
        df=df,
        df_op2=df_op2,
        df_carrier=df_carrier,
        final_bridge_df=final_bridge_df,
    )

    logger.info(f"OP2 weekly COUNTRY-BUSINESS rows: {len(op2_weekly_country_business_df)}")

    # Validation guards
    assert op2_weekly_country_business_df["business"].ne("Total").any()
    assert op2_weekly_country_business_df["orig_country"].ne("EU").all()

    final_bridge_df = pd.concat([final_bridge_df, op2_weekly_country_business_df], ignore_index=True)

    # Step 10: Create OP2 EU aggregations
    logger.info("Creating OP2 EU-TOTAL weekly bridge...")
    start_time = time.time()

    op2_eu_total_df = create_op2_eu_weekly_bridge(
        df=df,
        df_op2=df_op2,
        df_carrier=df_carrier,
        final_bridge_df=final_bridge_df,
    )

    logger.info(f"OP2 EU-TOTAL rows: {len(op2_eu_total_df)}")
    final_bridge_df = pd.concat([final_bridge_df, op2_eu_total_df], ignore_index=True)
    logger.info(f"OP2 EU-TOTAL creation took: {time.time() - start_time:.2f} seconds")

    logger.info("Creating OP2 EU-BUSINESS weekly bridge...")
    start_time = time.time()

    op2_eu_business_df = create_op2_eu_weekly_business_bridge(
        df=df,
        df_op2=df_op2,
        df_carrier=df_carrier,
        final_bridge_df=final_bridge_df,
    )

    logger.info(f"OP2 EU-BUSINESS rows: {len(op2_eu_business_df)}")
    final_bridge_df = pd.concat([final_bridge_df, op2_eu_business_df], ignore_index=True)
    logger.info(f"OP2 EU-BUSINESS creation took: {time.time() - start_time:.2f} seconds")

    # Adjust OP2 impacts
    final_bridge_df = adjust_op2_carrier_demand_impacts(final_bridge_df)

    # Step 11: Export to S3
    logger.info("Exporting results to S3...")
    start_time = time.time()
    export_to_quicksight(final_bridge_df, destination_bucket, destination_key)
    logger.info(f"Export to S3 took: {time.time() - start_time:.2f} seconds")

    logger.info("Processing completed successfully")

    # Calculate memory usage
    memory_usage = final_bridge_df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"Final DataFrame memory usage: {memory_usage:.2f} MB")

    return {
        "message": "Processing completed successfully",
        "rows_processed": len(final_bridge_df),
        "years_processed": len(years),
        "source_file": f"s3://{source_bucket}/{source_key}",
        "destination_file": f"s3://{destination_bucket}/{destination_key}",
        "memory_usage_mb": round(memory_usage, 2),
    }


def main():
    """
    Main entry point.

    Reads configuration from environment variables and executes the pipeline.

    Returns:
        dict: Status code and processing results
    """
    try:
        # Get environment variables
        source_bucket = os.environ.get("SOURCE_BUCKET")
        source_key = os.environ.get("SOURCE_KEY")
        carrier_scaling_bucket = os.environ.get("CARRIER_SCALING_BUCKET")
        carrier_scaling_key = os.environ.get("CARRIER_SCALING_KEY")
        op2_bucket = os.environ.get("OP2_BUCKET")
        op2_key = os.environ.get("OP2_KEY")
        destination_bucket = os.environ.get("DESTINATION_BUCKET")
        destination_key = os.environ.get("DESTINATION_KEY")

        # Log the start of execution
        logger.info("Starting analytics processing")
        logger.info(f"Source: s3://{source_bucket}/{source_key}")
        logger.info(f"Scaling: s3://{carrier_scaling_bucket}/{carrier_scaling_key}")
        logger.info(f"Destination: s3://{destination_bucket}/{destination_key}")

        # Validate required environment variables
        required_vars = [source_bucket, source_key, destination_bucket, destination_key]
        if not all(required_vars):
            raise ValueError(
                "Missing required environment variables. "
                "Please check SOURCE_BUCKET, SOURCE_KEY, DESTINATION_BUCKET, and DESTINATION_KEY are set."
            )

        # Run the pipeline
        results = run_analytics_pipeline(
            source_bucket, source_key,
            carrier_scaling_bucket, carrier_scaling_key,
            op2_bucket, op2_key,
            destination_bucket, destination_key,
        )

        # Log successful completion
        logger.info({
            "message": "Execution completed successfully",
            "results": results,
            "timestamp": datetime.now().isoformat(),
        })

        return {"statusCode": 200, "body": results}

    except Exception as e:
        # Log the error with full details
        logger.error({
            "message": "Error in execution",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }, exc_info=True)

        return {"statusCode": 500, "body": f"Error processing file: {str(e)}"}


if __name__ == "__main__":
    logger.info("=== STARTING MAIN FUNCTION ===")
    main()
    logger.info("=== MAIN FUNCTION COMPLETED ===")
