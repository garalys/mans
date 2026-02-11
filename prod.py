import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import boto3
from datetime import datetime, timedelta
import os
import logging
import duckdb
import io


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Monthly Report Generator
# ------------------------------------------------------------------------------
class MonthlyReportGenerator:
    def __init__(self, s3_bucket, output_bucket, sns_topic_arn, volume_bucket):
        self.s3_bucket = s3_bucket
        self.output_bucket = output_bucket
        self.volume_bucket = volume_bucket
        self.sns_topic_arn = sns_topic_arn

        self.s3_client = boto3.client("s3")
        self.sns_client = boto3.client("sns")

        logger.info("=" * 80)
        logger.info("MonthlyReportGenerator initialized")
        logger.info(f"Input bucket: {self.s3_bucket}")
        logger.info(f"Volume bucket: {self.volume_bucket}")
        logger.info(f"Output bucket: {self.output_bucket}")
        logger.info(f"SNS Topic ARN: {self.sns_topic_arn}")
        logger.info("=" * 80)

    # --------------------------------------------------------------------------
    def get_previous_month(self):
        """Get previous month's year and month in R2025/M10 format"""
        today = datetime.now()
        first_day_current_month = today.replace(day=1)
        last_day_prev_month = first_day_current_month - timedelta(days=1)

        year = f"R{last_day_prev_month.year}"
        month = f"M{last_day_prev_month.month:02d}"

        logger.info(f"Target period: {year} {month}")
        return year, month

    # --------------------------------------------------------------------------
    def load_data(self, year, month):
        """Load and process data from S3"""
        logger.info("=" * 80)
        logger.info("STARTING DATA LOAD")
        logger.info("=" * 80)

        try:
            # -------------------- Route data --------------------
            input_path = f"s3://{self.s3_bucket}/input-data/data.csv"
            logger.info(f"Loading route data from: {input_path}")
            obj = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key="input-data/data.csv"
            )

            df_raw = pd.read_csv(io.BytesIO(obj["Body"].read()))
            logger.info(f"✓ Loaded {len(df_raw)} rows from S3")
            conn = duckdb.connect()

            query = f"""
                SELECT
                    report_year,
                    report_month,
                    route,
                    vrid,
                    vehicle_carrier,
                    is_set,
                    orig_eu5_country,
                    dest_eu5_country,
                    supply_type,
                    SUM(cost_for_cpkm_eur) * 1.17459 AS total_cost,
                    SUM(distance_for_cpkm) AS total_distance,
                    SUM(executed_loads) AS executed_load
                FROM df_raw
                WHERE 
                report_year = '{year}'
                AND report_month = '{month}'
                GROUP BY ALL
            """

            df_routes = conn.execute(query).df()
            conn.close()


            logger.info(f"✓ Route data loaded: {len(df_routes)} records")

            if df_routes.empty:
                logger.warning("⚠ No route data found for the specified period!")
            else:
                logger.info(f"  - Unique routes: {df_routes['route'].nunique()}")
                logger.info(f"  - Unique carriers: {df_routes['vehicle_carrier'].nunique()}")
                logger.info(
                    f"  - Supply types: {df_routes['supply_type'].unique().tolist()}"
                )

            # -------------------- Volume data --------------------
            volume_path = f"s3://{self.volume_bucket}/volume/volume_by_vrid.csv000"
            logger.info(f"Loading volume data from: {volume_path}")

            df_cubes = pd.read_csv(volume_path)

            logger.info(f"✓ Volume data loaded: {len(df_cubes)} records")
            logger.info(f"  - Unique VRIDs: {df_cubes['vrid'].nunique()}")

            # -------------------- Processing --------------------
            logger.info("Processing data...")
            logger.info("  - Calculating CPKM...")

            df_routes["cpkm"] = np.where(
                df_routes["total_distance"] > 0,
                df_routes["total_cost"] / df_routes["total_distance"],
                0,
            )

            logger.info("  - Calculating CPKM z-scores per route...")
            df_routes["cpkm_route_z"] = (
                df_routes.groupby(
                    ["report_year", "report_month", "route"]
                )["cpkm"]
                .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            )

            logger.info("  - Merging with volume data...")
            df_routes = (
                df_routes.merge(df_cubes, on="vrid", how="left")
                .reset_index(drop=True)
            )

            missing_cubes = df_routes["cubes"].isna().sum()
            if missing_cubes > 0:
                logger.warning(
                    f"⚠ {missing_cubes} records missing cube data after merge"
                )

            logger.info(f"✓ Data processing complete: {len(df_routes)} records")
            logger.info("=" * 80)

            return df_routes

        except Exception as e:
            logger.error("❌ Error loading data", exc_info=True)
            raise

    # --------------------------------------------------------------------------
    def _execute_query(self, query):
        """Execute SQL query using DuckDB"""
        try:
            logger.info("Initializing DuckDB connection...")
            conn = duckdb.connect()
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            conn.execute("SET s3_region='eu-north-1';")

            logger.info("Executing query...")
            df = conn.execute(query).df()

            conn.close()
            logger.info("✓ Query executed successfully")
            return df

        except Exception:
            logger.error("❌ Query execution failed", exc_info=True)
            raise

    # --------------------------------------------------------------------------
    def compute_route_metrics(self, df):
        logger.info("Computing route metrics...")

        metrics = (
            df.groupby(["route", "supply_type"])
            .agg(
                total_loads=("executed_load", "sum"),
                total_cost=("total_cost", "sum"),
                cpkm_var=("cpkm", "var"),
                cpkm_std=("cpkm", "std"),
                obs=("cpkm", "count"),
            )
            .reset_index()
        )

        logger.info(f"✓ Computed metrics for {len(metrics)} route-supply combinations")
        return metrics

    # --------------------------------------------------------------------------
    def add_cost_over_mad(self, df, cpkm_col="cpkm", mad_cutoff=3.0):
        logger.info(f"Calculating MAD-based costs (cutoff={mad_cutoff})...")

        df = df.copy()

        g = df.groupby("route")[cpkm_col]
        median = g.transform("median")
        mad = (df[cpkm_col] - median).abs().groupby(df["route"]).transform("median")

        mad_threshold = median + mad_cutoff * mad / 0.6745

        df["cost_over_mad"] = np.where(
            df[cpkm_col] > mad_threshold,
            (df[cpkm_col] - mad_threshold)
            * df["total_distance"]
            * df["executed_load"],
            0.0,
        )

        logger.info(f"✓ Total cost over MAD: €{df['cost_over_mad'].sum():,.0f}")
        return df

    # --------------------------------------------------------------------------
    def filter_routes_with_mad_variance(self, df, routes, cpkm_col="cpkm", min_obs=3):
        logger.info(f"Filtering routes with MAD variance (min_obs={min_obs})...")

        valid_routes = []

        for route in routes:
            vals = df[df["route"] == route][cpkm_col].dropna()

            if len(vals) < min_obs:
                continue

            median = np.median(vals)
            mad = np.median(np.abs(vals - median))

            if mad > 0 and not np.isnan(mad):
                valid_routes.append(route)

        logger.info(
            f"✓ {len(valid_routes)} routes with valid MAD variance "
            f"(from {len(routes)} total)"
        )

        return valid_routes

    # --------------------------------------------------------------------------
    def plot_supply_type_analysis(
        self, pdf, df, supply_type, cpkm_column="cpkm", mad_cutoff=3, top_n=8
    ):
        logger.info("-" * 80)
        logger.info(f"Processing supply type: {supply_type}")

        df_supply = df[df["supply_type"] == supply_type].copy()

        if df_supply.empty:
            logger.warning(f"⚠ No data found for supply type: {supply_type}")
            return

        route_metrics = self.compute_route_metrics(df_supply)
        df_supply = self.add_cost_over_mad(
            df_supply, cpkm_col=cpkm_column, mad_cutoff=mad_cutoff
        )

        valid_routes = self.filter_routes_with_mad_variance(
            df_supply, df_supply["route"].unique(), min_obs=5
        )

        if not valid_routes:
            logger.warning("⚠ No valid routes with MAD variance")
            return

        routes_top_mad = (
            df_supply[df_supply["route"].isin(valid_routes)]
            .groupby("route")["cost_over_mad"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

        self._create_mad_plots(
            pdf, df_supply, routes_top_mad, supply_type, cpkm_column, mad_cutoff
        )

    # --------------------------------------------------------------------------
    def _create_mad_plots(self, pdf, df, routes, supply_type, cpkm_column, mad_cutoff):
        sns.set_theme(style="whitegrid")

        n_cols = 2
        n_rows = int(np.ceil(len(routes) / n_cols))

        fig = plt.figure(figsize=(24, 12 * n_rows))
        gs = fig.add_gridspec(n_rows * 2, n_cols, height_ratios=[3, 2] * n_rows)

        fig.suptitle(
            f"MAD Outlier Analysis - {supply_type}",
            fontsize=22,
            fontweight="bold",
        )

        for idx, route in enumerate(routes):
            row = (idx // n_cols) * 2
            col = idx % n_cols

            ax_plot = fig.add_subplot(gs[row, col])
            ax_table = fig.add_subplot(gs[row + 1, col])
            ax_table.axis("off")

            df_r = df[df["route"] == route]
            vals = df_r[cpkm_column].dropna()

            if vals.empty:
                continue

            median = np.median(vals)
            mad = np.median(np.abs(vals - median))
            mad_threshold = median + mad_cutoff * mad / 0.6745

            sns.histplot(
                data=df_r,
                x=cpkm_column,
                hue="vehicle_carrier",
                bins=50,
                kde=True,
                element="step",
                alpha=0.55,
                ax=ax_plot,
            )

            ax_plot.axvline(
                mad_threshold,
                color="red",
                linestyle="--",
                linewidth=3,
                label=f"MAD cutoff ({mad_cutoff})",
            )

            ax_plot.set_title(
                f"{route} | "
                f"{df_r.orig_eu5_country.iloc[0]} → "
                f"{df_r.dest_eu5_country.iloc[0]}",
                fontsize=15,
                fontweight="bold",
            )

            summary_data = [
                ["Loads", int(df_r.executed_load.sum())],
                ["Total cost €", f"€{df_r.total_cost.sum():,.0f}"],
                ["Median CPKM", f"{median:.3f}"],
                ["MAD", f"{mad:.3f}"],
                ["MAD Threshold", f"{mad_threshold:.3f}"],
            ]

            table = ax_table.table(
                cellText=summary_data,
                colLabels=["Metric", "Value"],
                cellLoc="center",
                bbox=[0.0, 0.55, 1.0, 0.4],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.1, 1.6)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # --------------------------------------------------------------------------
    def generate_report(self):
        logger.info("=" * 80)
        logger.info("STARTING REPORT GENERATION")
        logger.info("=" * 80)

        year, month = self.get_previous_month()
        df_routes = self.load_data(year, month)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"monthly_report_{year}_{month}_{timestamp}.pdf"
        local_pdf_path = f"/tmp/{pdf_filename}"

        with PdfPages(local_pdf_path) as pdf:
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(
                0.5,
                0.5,
                f"Monthly CPKM Report\n{year} {month}",
                ha="center",
                va="center",
                fontsize=24,
                fontweight="bold",
            )
            pdf.savefig(fig)
            plt.close()

            for supply_type in ["1.AZNG", "2.RLB", "3.3P"]:
                self.plot_supply_type_analysis(pdf, df_routes, supply_type)

            info = pdf.infodict()
            info["Title"] = f"Monthly CPKM Report {year} {month}"
            info["Author"] = "Automated Report System"
            info["CreationDate"] = datetime.now()

        s3_key = f"reports/{year}/{month}/{pdf_filename}"
        self.s3_client.upload_file(local_pdf_path, self.output_bucket, s3_key)

        pdf_url = self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.output_bucket, "Key": s3_key},
            ExpiresIn=604800,
        )

        self.send_notification(year, month, pdf_url, s3_key)
        return pdf_url

    # --------------------------------------------------------------------------
    def send_notification(self, year, month, pdf_url, s3_key):
        message = f"""Monthly CPKM Report Generated

                Period: {year} {month}

                Download Link (valid for 7 days):
                {pdf_url}

                S3 Location:
                s3://{self.output_bucket}/{s3_key}
                """

        self.sns_client.publish(
            TopicArn=self.sns_topic_arn,
            Subject=f"Monthly CPKM Report - {year} {month}",
            Message=message,
        )


# ------------------------------------------------------------------------------
def main():
    logger.info("=" * 80)
    logger.info("ECS TASK STARTED")
    logger.info("=" * 80)

    generator = MonthlyReportGenerator(
        s3_bucket=os.environ["INPUT_BUCKET"],
        output_bucket=os.environ["OUTPUT_BUCKET"],
        sns_topic_arn=os.environ["SNS_TOPIC_ARN"],
        volume_bucket=os.environ["VOLUME_BUCKET"],
    )

    generator.generate_report()

    logger.info("=" * 80)
    logger.info("ECS TASK COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
