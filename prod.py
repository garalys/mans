import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import boto3
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
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
    def _get_historical_periods(self, year, month, lookback_months=12):
        """Build list of (Ryear, Mmonth) tuples for the last N months."""
        y = int(year.replace("R", ""))
        m = int(month.replace("M", ""))
        target = datetime(y, m, 1)

        periods = []
        for i in range(lookback_months):
            dt = target - relativedelta(months=i)
            periods.append((f"R{dt.year}", f"M{dt.month:02d}"))
        return periods

    # --------------------------------------------------------------------------
    def load_data(self, year, month, load_history=False, lookback_months=12):
        """Load and process data from S3."""
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
            logger.info(f"Loaded {len(df_raw)} rows from S3")
            conn = duckdb.connect()

            if load_history:
                periods = self._get_historical_periods(year, month, lookback_months)
                period_filter = " OR ".join(
                    f"(report_year = '{y}' AND report_month = '{m}')"
                    for y, m in periods
                )
                where_clause = f"({period_filter})"
                logger.info(
                    f"Loading historical data: {len(periods)} periods "
                    f"({periods[-1][0]} {periods[-1][1]} to {periods[0][0]} {periods[0][1]})"
                )
            else:
                where_clause = (
                    f"report_year = '{year}' AND report_month = '{month}'"
                )

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
                WHERE {where_clause}
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
    def _create_supply_type_summary(self, pdf, df, year, month):
        """Create a descriptive stats summary page per supply type."""
        logger.info("Creating supply type summary page...")

        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        supply_types = sorted(df["supply_type"].dropna().unique())

        rows = []
        for st in supply_types:
            vals = df.loc[df["supply_type"] == st, "cpkm"].dropna()
            sub = df[df["supply_type"] == st]
            rows.append({
                "Supply Type": st,
                "Loads": int(sub["executed_load"].sum()),
                "Total Cost": f"{sub['total_cost'].sum():,.0f}",
                "Total Volume m3": f"{sub['cubes'].sum():,.0f}" if "cubes" in sub.columns else "N/A",
                "Routes": sub["route"].nunique(),
                "Carriers": sub["vehicle_carrier"].nunique(),
                "Mean CPKM": f"{vals.mean():.4f}",
                "Median CPKM": f"{vals.median():.4f}",
                "Min CPKM": f"{vals.min():.4f}",
                "Max CPKM": f"{vals.max():.4f}",
                "P25": f"{np.percentile(vals, 25):.4f}",
                "P75": f"{np.percentile(vals, 75):.4f}",
                "Std CPKM": f"{vals.std():.4f}",
            })

        summary_df = pd.DataFrame(rows)

        # --- Page 1: Summary table ---
        fig, ax = plt.subplots(figsize=(22, 4 + len(supply_types) * 0.8))
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
        _style_table(table, fontsize=11)
        table.scale(1, 2.0)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # --- Page 2: Box + violin plots ---
        fig, axes = plt.subplots(1, 2, figsize=(22, 8))
        fig.suptitle(
            f"CPKM Distribution by Supply Type  -  {year} {month}",
            fontsize=20, fontweight="bold",
        )

        sns.boxplot(
            data=df, x="supply_type", y="cpkm",
            palette=SEABORN_PALETTE, ax=axes[0],
            showfliers=False,
        )
        axes[0].set_title("Box Plot (outliers hidden for clarity)", fontsize=14)
        axes[0].set_xlabel("Supply Type", fontsize=12)
        axes[0].set_ylabel("CPKM", fontsize=12)

        sns.violinplot(
            data=df, x="supply_type", y="cpkm",
            palette=SEABORN_PALETTE, ax=axes[1],
            inner="quartile", cut=0,
        )
        axes[1].set_title("Violin Plot (density shape + quartiles)", fontsize=14)
        axes[1].set_xlabel("Supply Type", fontsize=12)
        axes[1].set_ylabel("CPKM", fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        logger.info("Supply type summary pages created")

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

        # Plot 1: by carrier (outliers only get names, rest = "OK")
        self._create_mad_plots(
            pdf, df_supply, routes_top_mad, supply_type, cpkm_column,
            mad_cutoff, hue_column="vehicle_carrier",
            title_suffix=f"MAD Outlier Analysis - {supply_type}",
        )

        # Plot 2: by is_set
        self._create_mad_plots(
            pdf, df_supply, routes_top_mad, supply_type, cpkm_column,
            mad_cutoff, hue_column="is_set",
            title_suffix=f"MAD Outlier Analysis - {supply_type} - Split by SET",
        )

    # --------------------------------------------------------------------------
    def _create_mad_plots(
        self, pdf, df, routes, supply_type, cpkm_column, mad_cutoff,
        hue_column="vehicle_carrier", title_suffix="", max_outlier_rows=6,
    ):
        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        n_cols = 2
        n_rows = int(np.ceil(len(routes) / n_cols))

        fig = plt.figure(figsize=(24, 12 * n_rows))
        gs = fig.add_gridspec(n_rows * 2, n_cols, height_ratios=[3, 2] * n_rows)

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

            # ---- MAD cutoff line ----
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

            # ---- Summary table ----
            summary_data = [
                ["Loads", f"{int(df_r.executed_load.sum())}"],
                ["Total cost", f"{df_r.total_cost.sum():,.0f}"],
                ["Cost over MAD", f"{df_r.cost_over_threshold.sum():,.0f}"],
                ["Median CPKM", f"{median:.3f}"],
                ["MAD", f"{mad:.3f}"],
                ["MAD Threshold", f"{mad_threshold:.3f}"],
            ]

            summary_tbl = ax_table.table(
                cellText=summary_data,
                colLabels=["Metric", "Value"],
                cellLoc="center",
                bbox=[0.00, 0.55, 1.00, 0.40],
            )
            _style_table(summary_tbl, fontsize=11)
            summary_tbl.scale(1.1, 1.6)

            # ---- Outlier detail table ----
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

                extra = max(0, len(out_tbl_df) - max_outlier_rows)
                out_tbl_df = out_tbl_df.head(max_outlier_rows)

                out_rows = [
                    [
                        carrier,
                        st,
                        int(r.Loads),
                        f"{r.Avg_MAD:.2f}",
                        f"{r.Max_MAD:.2f}",
                        f"{r.Cost:,.0f}",
                        f"{r.Distance:,.0f}",
                        f"{r.Cost_Over_Threshold:,.0f}",
                        f"{r.Outlier_CPKM:,.2f}",
                        f"{r.CPKM_Over_MAD:,.2f}",
                    ]
                    for (carrier, st), r in out_tbl_df.iterrows()
                ]

                if extra > 0:
                    out_rows.append([
                        f"+ {extra} more", "", "", "", "", "", "", "", "", "",
                    ])

                out_tbl = ax_table.table(
                    cellText=out_rows,
                    colLabels=[
                        "Carrier", group_column, "Loads",
                        "Avg MAD", "Max MAD", "Total Cost",
                        "Distance km", "Over MAD",
                        "CPKM", "CPKM over MAD",
                    ],
                    cellLoc="center",
                    bbox=[0.00, 0.00, 1.00, 0.50],
                )
                _style_table(out_tbl, fontsize=10)
                out_tbl.scale(1.1, 1.5)

        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # --------------------------------------------------------------------------
    # ANALYTICS: Route-level supply type benchmark
    # --------------------------------------------------------------------------
    def _create_route_supply_benchmark(self, pdf, df, year, month, top_n=15):
        """Side-by-side box plots of CPKM by supply type for top routes."""
        logger.info("Creating route-level supply type benchmark...")

        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        top_routes = (
            df.groupby("route")["total_cost"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

        df_top = df[df["route"].isin(top_routes)].copy()

        # Order routes by total cost descending
        route_order = (
            df_top.groupby("route")["total_cost"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

        n_cols = 3
        n_rows = int(np.ceil(len(route_order) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 5 * n_rows))
        fig.suptitle(
            f"Route-Level Supply Type Benchmark  -  {year} {month}\n"
            f"CPKM by Supply Type for Top {top_n} Routes by Cost",
            fontsize=20, fontweight="bold",
        )
        axes = np.array(axes).flatten()

        for idx, route in enumerate(route_order):
            ax = axes[idx]
            df_r = df_top[df_top["route"] == route]

            sns.boxplot(
                data=df_r, x="supply_type", y="cpkm",
                palette=SEABORN_PALETTE, ax=ax,
                showfliers=True, fliersize=3,
            )

            origin = df_r["orig_eu5_country"].iloc[0]
            dest = df_r["dest_eu5_country"].iloc[0]
            total_cost = df_r["total_cost"].sum()

            ax.set_title(
                f"{route}\n{origin} -> {dest} | Cost: {total_cost:,.0f}",
                fontsize=10, fontweight="bold",
            )
            ax.set_xlabel("")
            ax.set_ylabel("CPKM", fontsize=9)
            ax.tick_params(axis="x", labelsize=8)

        for j in range(len(route_order), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        logger.info("Route-level supply type benchmark created")

    # --------------------------------------------------------------------------
    # ANALYTICS: Carrier concentration risk (Herfindahl index)
    # --------------------------------------------------------------------------
    def _create_carrier_concentration(self, pdf, df, year, month, top_n=30):
        """Herfindahl index per route. High HHI + high CPKM = negotiation leverage."""
        logger.info("Creating carrier concentration analysis...")

        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        # Compute HHI per route: sum of squared market shares by carrier
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

        # Attach route-level CPKM and cost
        route_stats = (
            df.groupby("route")
            .agg(
                total_cost=("total_cost", "sum"),
                total_distance=("total_distance", "sum"),
                total_loads=("executed_load", "sum"),
                median_cpkm=("cpkm", "median"),
            )
            .reset_index()
        )

        hhi = hhi.merge(route_stats, on="route")

        # Filter to routes with meaningful volume
        hhi = hhi[hhi["total_loads"] >= 5].copy()

        # Top routes by cost for the table
        hhi_top = hhi.sort_values("total_cost", ascending=False).head(top_n)

        # --- Page 1: Scatter plot HHI vs CPKM ---
        fig, ax = plt.subplots(figsize=(22, 12))

        scatter = ax.scatter(
            hhi["hhi"],
            hhi["median_cpkm"],
            s=hhi["total_cost"] / hhi["total_cost"].max() * 800 + 20,
            c=hhi["hhi"],
            cmap="RdYlGn_r",
            alpha=0.65,
            edgecolors="black",
            linewidth=0.5,
        )

        ax.set_title(
            f"Carrier Concentration Risk  -  {year} {month}\n"
            f"Bubble size = Total cost | Color = HHI (red = concentrated)",
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

        # --- Page 2: Table of top routes by cost with HHI ---
        fig, ax = plt.subplots(figsize=(22, 2 + len(hhi_top) * 0.5))
        ax.axis("off")
        ax.set_title(
            f"Carrier Concentration - Top {top_n} Routes by Cost  -  {year} {month}",
            fontsize=18, fontweight="bold", pad=20,
        )

        tbl_data = []
        for _, r in hhi_top.iterrows():
            risk = "HIGH" if r.hhi >= 0.50 else ("MODERATE" if r.hhi >= 0.25 else "LOW")
            tbl_data.append([
                r.route,
                f"{r.total_cost:,.0f}",
                int(r.total_loads),
                f"{r.median_cpkm:.3f}",
                f"{r.hhi:.3f}",
                risk,
            ])

        table = ax.table(
            cellText=tbl_data,
            colLabels=["Route", "Total Cost", "Loads", "Median CPKM", "HHI", "Risk"],
            cellLoc="center",
            loc="center",
        )
        _style_table(table, fontsize=10)
        table.scale(1, 1.8)

        # Color-code the risk column
        for row_idx in range(1, len(tbl_data) + 1):
            risk_cell = table[row_idx, 5]
            risk_val = tbl_data[row_idx - 1][5]
            if risk_val == "HIGH":
                risk_cell.set_facecolor("#FF6B6B")
                risk_cell.set_text_props(color="white", fontweight="bold")
            elif risk_val == "MODERATE":
                risk_cell.set_facecolor("#FFD93D")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        logger.info("Carrier concentration analysis created")

    # --------------------------------------------------------------------------
    # ANALYTICS: Month-over-month trend
    # --------------------------------------------------------------------------
    def _create_mom_trend(self, pdf, df_historical, year, month):
        """Small-multiples line chart of CPKM trend per supply type."""
        logger.info("Creating month-over-month trend analysis...")

        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        # Build a period sort key
        df_historical = df_historical.copy()
        df_historical["period"] = (
            df_historical["report_year"] + " " + df_historical["report_month"]
        )
        df_historical["period_sort"] = (
            df_historical["report_year"].str.replace("R", "").astype(int) * 100
            + df_historical["report_month"].str.replace("M", "").astype(int)
        )

        # Aggregate weighted CPKM per period + supply type
        agg = (
            df_historical.groupby(["period", "period_sort", "supply_type"])
            .agg(
                total_cost=("total_cost", "sum"),
                total_distance=("total_distance", "sum"),
                total_loads=("executed_load", "sum"),
                median_cpkm=("cpkm", "median"),
                mean_cpkm=("cpkm", "mean"),
            )
            .reset_index()
        )
        agg["weighted_cpkm"] = np.where(
            agg["total_distance"] > 0,
            agg["total_cost"] / agg["total_distance"],
            np.nan,
        )
        agg = agg.sort_values("period_sort")

        supply_types = sorted(agg["supply_type"].dropna().unique())

        # --- Page 1: Weighted CPKM trend ---
        fig, axes = plt.subplots(1, 2, figsize=(24, 8))
        fig.suptitle(
            f"Month-over-Month CPKM Trend  -  up to {year} {month}",
            fontsize=20, fontweight="bold",
        )

        for st in supply_types:
            sub = agg[agg["supply_type"] == st].sort_values("period_sort")
            axes[0].plot(
                sub["period"], sub["weighted_cpkm"],
                marker="o", linewidth=2, markersize=5, label=st,
            )
            axes[1].plot(
                sub["period"], sub["total_loads"],
                marker="s", linewidth=2, markersize=5, label=st,
            )

        axes[0].set_title("Weighted CPKM (cost / distance)", fontsize=14)
        axes[0].set_ylabel("CPKM", fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].tick_params(axis="x", rotation=45, labelsize=9)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Total Loads", fontsize=14)
        axes[1].set_ylabel("Loads", fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].tick_params(axis="x", rotation=45, labelsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # --- Page 2: Small multiples per supply type (median + P25/P75 band) ---
        n_st = len(supply_types)
        fig, axes = plt.subplots(1, n_st, figsize=(8 * n_st, 7), sharey=True)
        if n_st == 1:
            axes = [axes]

        fig.suptitle(
            f"CPKM Trend by Supply Type (Median + P25/P75)  -  up to {year} {month}",
            fontsize=18, fontweight="bold",
        )

        # Need percentiles from raw data
        pct_agg = (
            df_historical.groupby(["period", "period_sort", "supply_type"])["cpkm"]
            .agg(["median", lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)])
            .reset_index()
        )
        pct_agg.columns = ["period", "period_sort", "supply_type", "median", "p25", "p75"]
        pct_agg = pct_agg.sort_values("period_sort")

        palette = sns.color_palette(SEABORN_PALETTE, n_st)

        for i, st in enumerate(supply_types):
            ax = axes[i]
            sub = pct_agg[pct_agg["supply_type"] == st]

            ax.fill_between(
                sub["period"], sub["p25"], sub["p75"],
                alpha=0.2, color=palette[i],
            )
            ax.plot(
                sub["period"], sub["median"],
                marker="o", linewidth=2, color=palette[i], label="Median",
            )

            ax.set_title(st, fontsize=14, fontweight="bold")
            ax.set_ylabel("CPKM" if i == 0 else "", fontsize=12)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        logger.info("Month-over-month trend created")

    # --------------------------------------------------------------------------
    # ANALYTICS: Volume-weighted CPKM scatter (bubble chart)
    # --------------------------------------------------------------------------
    def _create_volume_cpkm_scatter(self, pdf, df, year, month):
        """Bubble chart: x=distance, y=CPKM, size=volume, color=supply type."""
        logger.info("Creating volume-weighted CPKM scatter...")

        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        route_agg = (
            df.groupby(["route", "supply_type"])
            .agg(
                total_cost=("total_cost", "sum"),
                total_distance=("total_distance", "sum"),
                total_loads=("executed_load", "sum"),
                total_cubes=("cubes", "sum"),
            )
            .reset_index()
        )
        route_agg["cpkm"] = np.where(
            route_agg["total_distance"] > 0,
            route_agg["total_cost"] / route_agg["total_distance"],
            np.nan,
        )
        route_agg = route_agg.dropna(subset=["cpkm"])

        # Bubble size: scale cubes to reasonable point sizes
        max_cubes = route_agg["total_cubes"].max()
        route_agg["bubble_size"] = np.where(
            max_cubes > 0,
            route_agg["total_cubes"] / max_cubes * 600 + 15,
            50,
        )

        supply_types = sorted(route_agg["supply_type"].dropna().unique())
        palette = sns.color_palette(SEABORN_PALETTE, len(supply_types))
        color_map = {st: palette[i] for i, st in enumerate(supply_types)}

        fig, ax = plt.subplots(figsize=(24, 12))

        for st in supply_types:
            sub = route_agg[route_agg["supply_type"] == st]
            ax.scatter(
                sub["total_distance"],
                sub["cpkm"],
                s=sub["bubble_size"],
                c=[color_map[st]],
                alpha=0.55,
                edgecolors="black",
                linewidth=0.4,
                label=f"{st} ({len(sub)} routes)",
            )

        ax.set_title(
            f"Volume-Weighted CPKM Scatter  -  {year} {month}\n"
            f"Bubble size = Volume (m3) | Each dot = one route-supply combo",
            fontsize=18, fontweight="bold",
        )
        ax.set_xlabel("Total Distance (km)", fontsize=14)
        ax.set_ylabel("CPKM (cost per km)", fontsize=14)
        ax.legend(fontsize=12, markerscale=0.5)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        logger.info("Volume-weighted CPKM scatter created")

    # --------------------------------------------------------------------------
    # ANALYTICS: MAD score distribution per supply type
    # --------------------------------------------------------------------------
    def _create_mad_score_distribution(self, pdf, df, year, month, mad_cutoff=3.0):
        """Violin plot of MAD scores by supply type to show pricing volatility."""
        logger.info("Creating MAD score distribution per supply type...")

        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        df = df.copy()

        # Compute MAD score per row (within each route)
        g = df.groupby("route")["cpkm"]
        median = g.transform("median")
        mad = (df["cpkm"] - median).abs().groupby(df["route"]).transform("median")

        df["mad_score"] = np.where(
            mad > 0,
            0.6745 * (df["cpkm"] - median) / mad,
            np.nan,
        )

        df_valid = df.dropna(subset=["mad_score"])

        if df_valid.empty:
            logger.warning("No valid MAD scores to plot")
            return

        # --- Page 1: Violin + box overlay ---
        fig, axes = plt.subplots(1, 2, figsize=(24, 9))
        fig.suptitle(
            f"MAD Score Distribution by Supply Type  -  {year} {month}\n"
            f"Shows pricing volatility per supply type (higher = more variable)",
            fontsize=18, fontweight="bold",
        )

        sns.violinplot(
            data=df_valid, x="supply_type", y="mad_score",
            palette=SEABORN_PALETTE, ax=axes[0],
            inner="quartile", cut=0,
        )
        axes[0].axhline(
            mad_cutoff, color="red", linestyle="--", linewidth=2,
            label=f"MAD cutoff ({mad_cutoff})",
        )
        axes[0].axhline(
            -mad_cutoff, color="red", linestyle="--", linewidth=2,
        )
        axes[0].set_title("Full MAD Score Distribution", fontsize=14)
        axes[0].set_xlabel("Supply Type", fontsize=12)
        axes[0].set_ylabel("MAD Score", fontsize=12)
        axes[0].legend(fontsize=11)

        # Outlier counts as bar chart
        outlier_counts = (
            df_valid[df_valid["mad_score"].abs() > mad_cutoff]
            .groupby("supply_type")
            .size()
            .reindex(sorted(df_valid["supply_type"].unique()), fill_value=0)
        )
        total_counts = (
            df_valid.groupby("supply_type")
            .size()
            .reindex(sorted(df_valid["supply_type"].unique()), fill_value=0)
        )
        outlier_pct = (outlier_counts / total_counts * 100).fillna(0)

        bars = axes[1].bar(
            outlier_pct.index, outlier_pct.values,
            color=sns.color_palette(SEABORN_PALETTE, len(outlier_pct)),
            edgecolor="black", linewidth=0.5,
        )
        for bar, pct, cnt in zip(bars, outlier_pct.values, outlier_counts.values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{pct:.1f}%\n({cnt})",
                ha="center", fontsize=11, fontweight="bold",
            )

        axes[1].set_title(f"% of Loads Exceeding MAD Cutoff ({mad_cutoff})", fontsize=14)
        axes[1].set_xlabel("Supply Type", fontsize=12)
        axes[1].set_ylabel("Outlier %", fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.91])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        logger.info("MAD score distribution created")

    # --------------------------------------------------------------------------
    def generate_report(
        self,
        enable_route_benchmark=True,
        enable_carrier_concentration=True,
        enable_mom_trend=True,
        enable_volume_scatter=True,
        enable_mad_distribution=True,
        lookback_months=12,
    ):
        logger.info("=" * 80)
        logger.info("STARTING REPORT GENERATION")
        logger.info(f"  route_benchmark={enable_route_benchmark}")
        logger.info(f"  carrier_concentration={enable_carrier_concentration}")
        logger.info(f"  mom_trend={enable_mom_trend}")
        logger.info(f"  volume_scatter={enable_volume_scatter}")
        logger.info(f"  mad_distribution={enable_mad_distribution}")
        logger.info("=" * 80)

        year, month = self.get_previous_month()

        # Load current month data (always needed)
        df_routes = self.load_data(year, month)

        # Load historical data only if MoM trend is enabled
        df_historical = None
        if enable_mom_trend:
            df_historical = self.load_data(
                year, month, load_history=True, lookback_months=lookback_months
            )

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

            # ---- Supply type summary (descriptive stats + box/violin) ----
            self._create_supply_type_summary(pdf, df_routes, year, month)

            # ---- Per supply type MAD analysis ----
            for supply_type in ["1.AZNG", "2.RLB", "3.3P"]:
                self.plot_supply_type_analysis(pdf, df_routes, supply_type)

            # ---- ANALYTICS: Route-level supply type benchmark ----
            if enable_route_benchmark:
                self._create_route_supply_benchmark(pdf, df_routes, year, month)

            # ---- ANALYTICS: Carrier concentration risk ----
            if enable_carrier_concentration:
                self._create_carrier_concentration(pdf, df_routes, year, month)

            # ---- ANALYTICS: Month-over-month trend ----
            if enable_mom_trend and df_historical is not None:
                self._create_mom_trend(pdf, df_historical, year, month)

            # ---- ANALYTICS: Volume-weighted CPKM scatter ----
            if enable_volume_scatter:
                self._create_volume_cpkm_scatter(pdf, df_routes, year, month)

            # ---- ANALYTICS: MAD score distribution per supply type ----
            if enable_mad_distribution:
                self._create_mad_score_distribution(pdf, df_routes, year, month)

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

    # Feature flags - set to False to skip any analytics section
    generator.generate_report(
        enable_route_benchmark=True,
        enable_carrier_concentration=True,
        enable_mom_trend=True,
        enable_volume_scatter=True,
        enable_mad_distribution=True,
        lookback_months=12,
    )

    logger.info("=" * 80)
    logger.info("ECS TASK COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
