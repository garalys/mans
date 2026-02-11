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
# Consistent seaborn theme for all pages
# ------------------------------------------------------------------------------
SEABORN_PALETTE = "muted"
TABLE_HEADER_COLOR = "#4472C4"
TABLE_HEADER_FONT_COLOR = "#FFFFFF"
TABLE_ROW_EVEN = "#F2F2F2"
TABLE_ROW_ODD = "#FFFFFF"


def _style_table(table, fontsize=10):
    """Apply consistent professional styling to matplotlib tables."""
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER_COLOR)
            cell.set_text_props(color=TABLE_HEADER_FONT_COLOR, fontweight="bold")
        else:
            cell.set_facecolor(TABLE_ROW_EVEN if row % 2 == 0 else TABLE_ROW_ODD)


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
        """Load and process data from S3."""
        logger.info("=" * 80)
        logger.info("STARTING DATA LOAD")
        logger.info("=" * 80)

        try:
            input_path = f"s3://{self.s3_bucket}/input-data/data.csv"
            logger.info(f"Loading route data from: {input_path}")
            obj = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key="input-data/data.csv"
            )

            df_raw = pd.read_csv(io.BytesIO(obj["Body"].read()))
            logger.info(f"Loaded {len(df_raw)} rows from S3")
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

            logger.info(f"Route data loaded: {len(df_routes)} records")

            if df_routes.empty:
                logger.warning("No route data found for the specified period!")
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

            logger.info(f"Volume data loaded: {len(df_cubes)} records")
            logger.info(f"  - Unique VRIDs: {df_cubes['vrid'].nunique()}")

            # -------------------- Processing --------------------
            logger.info("Processing data...")

            df_routes["cpkm"] = np.where(
                df_routes["total_distance"] > 0,
                df_routes["total_cost"] / df_routes["total_distance"],
                0,
            )

            df_routes["cpkm_route_z"] = (
                df_routes.groupby(
                    ["report_year", "report_month", "route"]
                )["cpkm"]
                .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            )

            df_routes = (
                df_routes.merge(df_cubes, on="vrid", how="left")
                .reset_index(drop=True)
            )

            missing_cubes = df_routes["cubes"].isna().sum()
            if missing_cubes > 0:
                logger.warning(
                    f"{missing_cubes} records missing cube data after merge"
                )

            logger.info(f"Data processing complete: {len(df_routes)} records")
            logger.info("=" * 80)

            return df_routes

        except Exception as e:
            logger.error("Error loading data", exc_info=True)
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
            logger.info("Query executed successfully")
            return df

        except Exception:
            logger.error("Query execution failed", exc_info=True)
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

        logger.info(f"Computed metrics for {len(metrics)} route-supply combinations")
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

        logger.info(f"Total cost over MAD: {df['cost_over_mad'].sum():,.0f}")
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
            f"{len(valid_routes)} routes with valid MAD variance "
            f"(from {len(routes)} total)"
        )

        return valid_routes

    # --------------------------------------------------------------------------
    # Supply Type Overview (table only, no box/violin)
    # --------------------------------------------------------------------------
    def _create_supply_type_summary(self, pdf, df, year, month):
        """Summary table with P2.5/P97.5 instead of raw min/max."""
        logger.info("Creating supply type summary page...")

        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        supply_types = sorted(df["supply_type"].dropna().unique())

        rows = []
        for st in supply_types:
            vals = df.loc[df["supply_type"] == st, "cpkm"].dropna()
            sub = df[df["supply_type"] == st]
            n_carriers = sub["vehicle_carrier"].nunique()
            total_loads = int(sub["executed_load"].sum())
            lcr = total_loads / n_carriers if n_carriers > 0 else 0
            rows.append({
                "Supply Type": st,
                "Loads": total_loads,
                "Total Cost": f"{sub['total_cost'].sum():,.0f}",
                "Volume m3": f"{sub['cubes'].sum():,.0f}" if "cubes" in sub.columns else "N/A",
                "Routes": sub["route"].nunique(),
                "Carriers": n_carriers,
                "LCR": f"{lcr:.1f}",
                "Mean CPKM": f"{vals.mean():.4f}",
                "Median CPKM": f"{vals.median():.4f}",
                "P2.5": f"{np.percentile(vals, 2.5):.4f}",
                "P97.5": f"{np.percentile(vals, 97.5):.4f}",
                "P25": f"{np.percentile(vals, 25):.4f}",
                "P75": f"{np.percentile(vals, 75):.4f}",
                "Std CPKM": f"{vals.std():.4f}",
            })

        summary_df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(24, 4 + len(supply_types) * 0.8))
        ax.axis("off")
        ax.set_title(
            f"Supply Type Overview  -  {year} {month}",
            fontsize=20, fontweight="bold", pad=20,
        )

        table = ax.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc="center",
            loc="center",
        )
        _style_table(table, fontsize=10)
        table.scale(1, 2.0)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        logger.info("Supply type summary page created")

    # --------------------------------------------------------------------------
    # Top routes by cost/volume - histograms with hue=supply_type + box plots
    # --------------------------------------------------------------------------
    def _create_top_routes_overview(
        self, pdf, df, year, month, top_n=15, top_filter="total_cost",
        title_label="Cost",
    ):
        """Histograms + box plots + descriptive stats table per route."""
        logger.info(f"Creating top {top_n} routes by {title_label}...")

        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        top_routes = (
            df.groupby("route", as_index=False)[top_filter]
            .sum()
            .sort_values(top_filter, ascending=False)
            .head(top_n)["route"]
            .tolist()
        )

        df_top = df[df["route"].isin(top_routes)].copy()

        # Consistent supply type colors
        supply_types = sorted(df["supply_type"].dropna().unique())
        palette = sns.color_palette(SEABORN_PALETTE, len(supply_types))
        color_map = {st: palette[i] for i, st in enumerate(supply_types)}

        # One page per route: histogram + box plot + stats table
        for route in top_routes:
            df_r = df_top[df_top["route"] == route]
            vals = df_r["cpkm"].dropna()

            if vals.empty:
                continue

            origin = df_r["orig_eu5_country"].iloc[0]
            dest = df_r["dest_eu5_country"].iloc[0]

            fig = plt.figure(figsize=(24, 16))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])

            ax_hist = fig.add_subplot(gs[0, 0])
            ax_box = fig.add_subplot(gs[0, 1])
            ax_table = fig.add_subplot(gs[1, :])
            ax_table.axis("off")

            fig.suptitle(
                f"Top Routes by {title_label}  -  {year} {month}\n"
                f"{route} | {origin} -> {dest}",
                fontsize=18, fontweight="bold",
            )

            # ---- Histogram with hue=supply_type ----
            bin_edges = np.histogram_bin_edges(vals, bins=30)

            for st in df_r["supply_type"].dropna().unique():
                st_vals = df_r.loc[df_r["supply_type"] == st, "cpkm"].dropna()
                if st_vals.empty:
                    continue
                label = f"{st} (u={st_vals.mean():.2f}, m={st_vals.median():.2f})"
                ax_hist.hist(
                    st_vals, bins=bin_edges, alpha=0.6, label=label,
                    edgecolor="black", color=color_map.get(st),
                )

            ax_hist.set_title("CPKM Distribution by Supply Type", fontsize=13, fontweight="bold")
            ax_hist.set_xlabel("CPKM", fontsize=11)
            ax_hist.set_ylabel("Number of Loads", fontsize=11)
            ax_hist.legend(fontsize=9)
            ax_hist.grid(True, alpha=0.3)

            # ---- Box plot by supply type ----
            st_present = sorted(df_r["supply_type"].dropna().unique())
            if len(st_present) > 0:
                sns.boxplot(
                    data=df_r, x="supply_type", y="cpkm",
                    palette=color_map, ax=ax_box,
                    showfliers=True, fliersize=3,
                    order=st_present,
                )
            ax_box.set_title("CPKM Box Plot by Supply Type", fontsize=13, fontweight="bold")
            ax_box.set_xlabel("Supply Type", fontsize=11)
            ax_box.set_ylabel("CPKM", fontsize=11)

            # ---- Descriptive stats table per supply type + overall ----
            table_rows = []
            for st in st_present:
                sub = df_r[df_r["supply_type"] == st]
                st_vals = sub["cpkm"].dropna()
                n_carriers = sub["vehicle_carrier"].nunique()
                st_loads = int(sub["executed_load"].sum())
                lcr = st_loads / n_carriers if n_carriers > 0 else 0
                table_rows.append([
                    st,
                    f"{sub['total_cost'].sum():,.0f}",
                    f"{sub['cubes'].sum():,.0f}" if "cubes" in sub.columns else "N/A",
                    st_loads,
                    n_carriers,
                    f"{lcr:.1f}",
                    f"{st_vals.mean():.3f}" if len(st_vals) > 0 else "-",
                    f"{st_vals.median():.3f}" if len(st_vals) > 0 else "-",
                    f"{np.percentile(st_vals, 25):.3f}" if len(st_vals) > 0 else "-",
                    f"{np.percentile(st_vals, 75):.3f}" if len(st_vals) > 0 else "-",
                ])

            # Overall row
            n_carriers_all = df_r["vehicle_carrier"].nunique()
            total_loads_all = int(df_r["executed_load"].sum())
            lcr_all = total_loads_all / n_carriers_all if n_carriers_all > 0 else 0
            table_rows.append([
                "OVERALL",
                f"{df_r['total_cost'].sum():,.0f}",
                f"{df_r['cubes'].sum():,.0f}" if "cubes" in df_r.columns else "N/A",
                total_loads_all,
                n_carriers_all,
                f"{lcr_all:.1f}",
                f"{vals.mean():.3f}",
                f"{vals.median():.3f}",
                f"{np.percentile(vals, 25):.3f}",
                f"{np.percentile(vals, 75):.3f}",
            ])

            tbl = ax_table.table(
                cellText=table_rows,
                colLabels=[
                    "Supply Type", "Total Cost", "Volume m3", "Loads",
                    "Carriers", "LCR", "Mean CPKM", "Median CPKM", "P25", "P75",
                ],
                cellLoc="center",
                loc="center",
            )
            _style_table(tbl, fontsize=10)
            tbl.scale(1, 1.8)

            # Bold the OVERALL row
            n_data_rows = len(table_rows)
            for col_idx in range(10):
                tbl[n_data_rows, col_idx].set_text_props(fontweight="bold")
                tbl[n_data_rows, col_idx].set_facecolor("#D6E4F0")

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        logger.info(f"Top routes by {title_label} created")

    # --------------------------------------------------------------------------
    # Per supply type MAD analysis
    # --------------------------------------------------------------------------
    def plot_supply_type_analysis(
        self, pdf, df, supply_type, cpkm_column="cpkm", mad_cutoff=3, top_n=8
    ):
        logger.info("-" * 80)
        logger.info(f"Processing supply type: {supply_type}")

        df_supply = df[df["supply_type"] == supply_type].copy()

        if df_supply.empty:
            logger.warning(f"No data found for supply type: {supply_type}")
            return

        route_metrics = self.compute_route_metrics(df_supply)
        df_supply = self.add_cost_over_mad(
            df_supply, cpkm_col=cpkm_column, mad_cutoff=mad_cutoff
        )

        # Only routes with MAD variance
        valid_routes = self.filter_routes_with_mad_variance(
            df_supply, df_supply["route"].unique(), min_obs=5
        )

        if not valid_routes:
            logger.warning("No valid routes with MAD variance")
            return

        routes_top_mad = (
            df_supply[df_supply["route"].isin(valid_routes)]
            .groupby("route")["cost_over_mad"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

        # Only pass routes that have MAD variance
        routes_filtered = self.filter_routes_with_mad_variance(
            df_supply, routes_top_mad, cpkm_col=cpkm_column, min_obs=3
        )

        if not routes_filtered:
            logger.warning("No routes with MAD variance after filtering")
            return

        # Plot 1: by carrier (outliers only get names, rest = "OK")
        self._create_mad_plots(
            pdf, df_supply, routes_filtered, supply_type, cpkm_column,
            mad_cutoff, hue_column="vehicle_carrier",
            title_suffix=f"MAD Outlier Analysis - {supply_type}",
        )

        # Plot 2: by is_set
        self._create_mad_plots(
            pdf, df_supply, routes_filtered, supply_type, cpkm_column,
            mad_cutoff, hue_column="is_set",
            title_suffix=f"MAD Outlier Analysis - {supply_type} - Split by SET",
        )

    # --------------------------------------------------------------------------
    def _create_mad_plots(
        self, pdf, df, routes, supply_type, cpkm_column, mad_cutoff,
        hue_column="vehicle_carrier", title_suffix="",
    ):
        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        n_cols = 2
        n_rows = int(np.ceil(len(routes) / n_cols))

        fig = plt.figure(figsize=(24, 14 * n_rows))
        gs = fig.add_gridspec(n_rows * 2, n_cols, height_ratios=[3, 2.5] * n_rows)

        fig.suptitle(
            title_suffix,
            fontsize=22,
            fontweight="bold",
            y=0.995,
        )

        for idx, route in enumerate(routes):
            row = (idx // n_cols) * 2
            col = idx % n_cols

            ax_plot = fig.add_subplot(gs[row, col])
            ax_table = fig.add_subplot(gs[row + 1, col])
            ax_table.axis("off")

            df_r = df[df["route"] == route].copy()
            vals = df_r[cpkm_column].dropna()

            if vals.empty:
                ax_plot.set_title(f"{route} (no data)")
                ax_plot.axis("off")
                continue

            # ---- MAD stats ----
            median = np.median(vals)
            mad = np.median(np.abs(vals - median))

            if mad == 0 or np.isnan(mad):
                ax_plot.set_title(f"{route} (no MAD variance)")
                ax_plot.axis("off")
                continue

            mad_threshold = median + mad_cutoff * mad / 0.6745
            df_r["mad_score"] = 0.6745 * (df_r[cpkm_column] - median) / mad
            df_r["cost_over_threshold"] = np.where(
                df_r[cpkm_column] > mad_threshold,
                (df_r[cpkm_column] - mad_threshold)
                * df_r["total_distance"]
                * df_r["executed_load"],
                0.0,
            )

            outliers = df_r[df_r["mad_score"] > mad_cutoff]

            # ---- Carrier hue: only outlier carriers get names, rest = "OK" ----
            plot_hue = hue_column
            if hue_column == "vehicle_carrier":
                outlier_carriers = set(outliers["vehicle_carrier"].unique())
                df_r["plot_carrier"] = np.where(
                    df_r["vehicle_carrier"].isin(outlier_carriers),
                    df_r["vehicle_carrier"],
                    "OK",
                )
                plot_hue = "plot_carrier"

            # ---- Distribution ----
            sns.histplot(
                data=df_r,
                x=cpkm_column,
                hue=plot_hue,
                bins=50,
                kde=True,
                stat="count",
                element="step",
                alpha=0.55,
                ax=ax_plot,
            )

            ax_plot.axvline(
                mad_threshold,
                color="red",
                linestyle="--",
                linewidth=2.5,
                label=f"MAD cutoff ({mad_cutoff})",
            )

            ax_plot.set_title(
                f"{route} | "
                f"{df_r.orig_eu5_country.iloc[0]} -> "
                f"{df_r.dest_eu5_country.iloc[0]}",
                fontsize=15,
                fontweight="bold",
            )
            ax_plot.set_xlabel("CPKM", fontsize=12)
            ax_plot.set_ylabel("Number of loads", fontsize=12)

            # ---- Summary table with cubes, mean, LCR ----
            n_carriers = df_r["vehicle_carrier"].nunique()
            total_loads = int(df_r["executed_load"].sum())
            lcr = total_loads / n_carriers if n_carriers > 0 else 0
            total_cubes = df_r["cubes"].sum() if "cubes" in df_r.columns else 0

            summary_data = [
                ["Loads", f"{total_loads}"],
                ["Carriers", f"{n_carriers}"],
                ["LCR", f"{lcr:.1f}"],
                ["Total Cost", f"{df_r.total_cost.sum():,.0f}"],
                ["Volume m3", f"{total_cubes:,.0f}"],
                ["Cost over MAD", f"{df_r.cost_over_threshold.sum():,.0f}"],
                ["Mean CPKM", f"{vals.mean():.3f}"],
                ["Median CPKM", f"{median:.3f}"],
                ["MAD", f"{mad:.3f}"],
                ["MAD Threshold", f"{mad_threshold:.3f}"],
            ]

            summary_tbl = ax_table.table(
                cellText=summary_data,
                colLabels=["Metric", "Value"],
                cellLoc="center",
                bbox=[0.00, 0.60, 1.00, 0.38],
            )
            _style_table(summary_tbl, fontsize=10)
            summary_tbl.scale(1.1, 1.4)

            # ---- Outlier detail table (ALL outliers, no limit) ----
            if hue_column == "vehicle_carrier":
                group_column = "is_set"
            else:
                group_column = hue_column

            if not outliers.empty:
                out_tbl_df = (
                    outliers
                    .groupby(["vehicle_carrier", group_column])
                    .agg(
                        Loads=("executed_load", "sum"),
                        Avg_MAD=("mad_score", "mean"),
                        Max_MAD=("mad_score", "max"),
                        Cost=("total_cost", "sum"),
                        Distance=("total_distance", "sum"),
                        Cost_Over_Threshold=("cost_over_threshold", "sum"),
                    )
                    .sort_values("Cost_Over_Threshold", ascending=False)
                )
                out_tbl_df["Outlier_CPKM"] = np.where(
                    out_tbl_df["Distance"] > 0,
                    out_tbl_df["Cost"] / out_tbl_df["Distance"],
                    np.nan,
                )
                out_tbl_df["CPKM_Over_MAD"] = np.where(
                    out_tbl_df["Distance"] > 0,
                    out_tbl_df["Cost_Over_Threshold"] / out_tbl_df["Distance"],
                    np.nan,
                )

                out_rows = [
                    [
                        carrier, st, int(r.Loads),
                        f"{r.Avg_MAD:.2f}", f"{r.Max_MAD:.2f}",
                        f"{r.Cost:,.0f}", f"{r.Distance:,.0f}",
                        f"{r.Cost_Over_Threshold:,.0f}",
                        f"{r.Outlier_CPKM:,.2f}", f"{r.CPKM_Over_MAD:,.2f}",
                    ]
                    for (carrier, st), r in out_tbl_df.iterrows()
                ]

                # Scale table height based on number of rows
                n_out = len(out_rows)
                tbl_height = min(0.55, 0.06 + n_out * 0.045)

                out_tbl = ax_table.table(
                    cellText=out_rows,
                    colLabels=[
                        "Carrier", group_column, "Loads",
                        "Avg MAD", "Max MAD", "Total Cost",
                        "Distance km", "Over MAD",
                        "CPKM", "CPKM over MAD",
                    ],
                    cellLoc="center",
                    bbox=[0.00, max(0.0, 0.55 - tbl_height), 1.00, tbl_height],
                )
                _style_table(out_tbl, fontsize=9)
                out_tbl.scale(1.1, 1.3)

        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # --------------------------------------------------------------------------
    # Carrier concentration risk (Herfindahl index)
    # --------------------------------------------------------------------------
    def _create_carrier_concentration(self, pdf, df, year, month, top_n=30):
        """HHI per route with scatter (outlier-trimmed) + two tables (by cost, by volume)."""
        logger.info("Creating carrier concentration analysis...")

        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        # Compute HHI per route
        route_carrier = (
            df.groupby(["route", "vehicle_carrier"])["executed_load"]
            .sum()
            .reset_index()
        )
        route_totals = (
            route_carrier.groupby("route")["executed_load"]
            .sum()
            .rename("route_total")
        )
        route_carrier = route_carrier.merge(route_totals, on="route")
        route_carrier["share"] = (
            route_carrier["executed_load"] / route_carrier["route_total"]
        )
        route_carrier["share_sq"] = route_carrier["share"] ** 2

        hhi = (
            route_carrier.groupby("route")["share_sq"]
            .sum()
            .rename("hhi")
            .reset_index()
        )

        # Route-level stats
        route_stats = (
            df.groupby("route")
            .agg(
                total_cost=("total_cost", "sum"),
                total_distance=("total_distance", "sum"),
                total_loads=("executed_load", "sum"),
                total_cubes=("cubes", "sum"),
                median_cpkm=("cpkm", "median"),
                mean_cpkm=("cpkm", "mean"),
                n_carriers=("vehicle_carrier", "nunique"),
            )
            .reset_index()
        )
        route_stats["lcr"] = route_stats["total_loads"] / route_stats["n_carriers"]

        # Supply type breakdown per route
        route_supply = (
            df.groupby(["route", "supply_type"])
            .agg(
                st_cost=("total_cost", "sum"),
                st_cubes=("cubes", "sum"),
                st_loads=("executed_load", "sum"),
                st_carriers=("vehicle_carrier", "nunique"),
                st_mean_cpkm=("cpkm", "mean"),
                st_median_cpkm=("cpkm", "median"),
            )
            .reset_index()
        )

        hhi = hhi.merge(route_stats, on="route")
        hhi = hhi[hhi["total_loads"] >= 5].copy()

        # ---- Scatter plot: trim CPKM outliers for readable axes ----
        cpkm_lo = hhi["median_cpkm"].quantile(0.02)
        cpkm_hi = hhi["median_cpkm"].quantile(0.98)
        hhi_plot = hhi[
            (hhi["median_cpkm"] >= cpkm_lo) & (hhi["median_cpkm"] <= cpkm_hi)
        ]

        fig, ax = plt.subplots(figsize=(22, 12))

        scatter = ax.scatter(
            hhi_plot["hhi"],
            hhi_plot["median_cpkm"],
            s=hhi_plot["total_cost"] / hhi_plot["total_cost"].max() * 800 + 20,
            c=hhi_plot["hhi"],
            cmap="RdYlGn_r",
            alpha=0.65,
            edgecolors="black",
            linewidth=0.5,
        )

        ax.set_title(
            f"Carrier Concentration Risk  -  {year} {month}\n"
            f"Bubble size = Total cost | Color = HHI (red = concentrated)\n"
            f"CPKM trimmed to P2-P98 for readability",
            fontsize=18, fontweight="bold",
        )
        ax.set_xlabel("Herfindahl-Hirschman Index (HHI)", fontsize=14)
        ax.set_ylabel("Median CPKM", fontsize=14)

        ax.axvline(0.25, color="orange", linestyle="--", alpha=0.7, label="Moderate concentration")
        ax.axvline(0.50, color="red", linestyle="--", alpha=0.7, label="High concentration")
        ax.legend(fontsize=11)

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label("HHI", fontsize=12)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # ---- Helper to build detailed table ----
        def _build_concentration_table(hhi_subset, sort_col, title):
            top = hhi_subset.sort_values(sort_col, ascending=False).head(top_n)

            tbl_data = []
            for _, r in top.iterrows():
                risk = "HIGH" if r.hhi >= 0.50 else ("MOD" if r.hhi >= 0.25 else "LOW")

                # Supply type breakdown
                rt_supply = route_supply[route_supply["route"] == r.route]
                st_parts = []
                for _, s in rt_supply.iterrows():
                    st_lcr = s.st_loads / s.st_carriers if s.st_carriers > 0 else 0
                    st_parts.append(
                        f"{s.supply_type}: "
                        f"C={s.st_cost:,.0f} V={s.st_cubes:,.0f} "
                        f"L={int(s.st_loads)} LCR={st_lcr:.1f} "
                        f"u={s.st_mean_cpkm:.3f} m={s.st_median_cpkm:.3f}"
                    )
                st_info = " | ".join(st_parts)

                tbl_data.append([
                    r.route,
                    f"{r.total_cost:,.0f}",
                    f"{r.total_cubes:,.0f}",
                    int(r.total_loads),
                    int(r.n_carriers),
                    f"{r.lcr:.1f}",
                    f"{r.mean_cpkm:.3f}",
                    f"{r.median_cpkm:.3f}",
                    f"{r.hhi:.3f}",
                    risk,
                    st_info,
                ])

            # Two-column layout: main table on left page, supply detail wraps
            fig, ax = plt.subplots(figsize=(28, 3 + len(tbl_data) * 0.6))
            ax.axis("off")
            ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

            table = ax.table(
                cellText=tbl_data,
                colLabels=[
                    "Route", "Total Cost", "Volume m3", "Loads", "Carriers",
                    "LCR", "Mean CPKM", "Median CPKM", "HHI", "Risk",
                    "Supply Type Breakdown",
                ],
                cellLoc="center",
                loc="center",
            )
            _style_table(table, fontsize=8)
            table.scale(1, 1.6)

            # Set supply type column wider
            for row_idx in range(len(tbl_data) + 1):
                table[row_idx, 10].set_text_props(fontsize=7, ha="left")

            # Color-code risk
            for row_idx in range(1, len(tbl_data) + 1):
                risk_cell = table[row_idx, 9]
                risk_val = tbl_data[row_idx - 1][9]
                if risk_val == "HIGH":
                    risk_cell.set_facecolor("#FF6B6B")
                    risk_cell.set_text_props(color="white", fontweight="bold")
                elif risk_val == "MOD":
                    risk_cell.set_facecolor("#FFD93D")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        # ---- Table 1: Top by cost ----
        _build_concentration_table(
            hhi, "total_cost",
            f"Carrier Concentration - Top {top_n} Routes by Cost  -  {year} {month}",
        )

        # ---- Table 2: Top by volume ----
        _build_concentration_table(
            hhi, "total_cubes",
            f"Carrier Concentration - Top {top_n} Routes by Volume  -  {year} {month}",
        )

        logger.info("Carrier concentration analysis created")

    # --------------------------------------------------------------------------
    def generate_report(
        self,
        enable_top_routes_by_cost=True,
        enable_top_routes_by_volume=True,
        enable_mad_analysis=True,
        enable_carrier_concentration=True,
    ):
        logger.info("=" * 80)
        logger.info("STARTING REPORT GENERATION")
        logger.info(f"  top_routes_by_cost={enable_top_routes_by_cost}")
        logger.info(f"  top_routes_by_volume={enable_top_routes_by_volume}")
        logger.info(f"  mad_analysis={enable_mad_analysis}")
        logger.info(f"  carrier_concentration={enable_carrier_concentration}")
        logger.info("=" * 80)

        year, month = self.get_previous_month()
        df_routes = self.load_data(year, month)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"monthly_report_{year}_{month}_{timestamp}.pdf"
        local_pdf_path = f"/tmp/{pdf_filename}"

        with PdfPages(local_pdf_path) as pdf:
            # ---- Title page ----
            sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)
            fig = plt.figure(figsize=(22, 14))
            fig.text(
                0.5, 0.55,
                f"Monthly CPKM Report\n{year} {month}",
                ha="center", va="center",
                fontsize=32, fontweight="bold",
            )
            fig.text(
                0.5, 0.40,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ha="center", va="center",
                fontsize=14, color="grey",
            )
            pdf.savefig(fig)
            plt.close()

            # ---- Supply type overview table ----
            self._create_supply_type_summary(pdf, df_routes, year, month)

            # ---- Top 15 routes by cost (histograms + box + stats) ----
            if enable_top_routes_by_cost:
                self._create_top_routes_overview(
                    pdf, df_routes, year, month,
                    top_n=15, top_filter="total_cost", title_label="Cost",
                )

            # ---- Top 15 routes by volume (histograms + box + stats) ----
            if enable_top_routes_by_volume:
                self._create_top_routes_overview(
                    pdf, df_routes, year, month,
                    top_n=15, top_filter="cubes", title_label="Volume",
                )

            # ---- Per supply type MAD analysis ----
            if enable_mad_analysis:
                for supply_type in ["1.AZNG", "2.RLB", "3.3P"]:
                    self.plot_supply_type_analysis(pdf, df_routes, supply_type)

            # ---- Carrier concentration risk ----
            if enable_carrier_concentration:
                self._create_carrier_concentration(pdf, df_routes, year, month)

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

    generator.generate_report(
        enable_top_routes_by_cost=True,
        enable_top_routes_by_volume=True,
        enable_mad_analysis=True,
        enable_carrier_concentration=True,
    )

    logger.info("=" * 80)
    logger.info("ECS TASK COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
