import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

import boto3
from datetime import datetime, timedelta
import os
import logging
import duckdb
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SEABORN_PALETTE = "muted"
TABLE_HEADER_COLOR = "#4472C4"
TABLE_HEADER_FONT_COLOR = "#FFFFFF"
TABLE_ROW_EVEN = "#F2F2F2"
TABLE_ROW_ODD = "#FFFFFF"

SUPPLY_TYPES_ORDERED = ["1.AZNG", "2.RLB", "3.3P"]


def _style_table(table, fontsize=10):
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


def _render_table_page(pdf, tbl_data, col_labels, title, fontsize=8, figwidth=28, row_height=0.55):
    fig, ax = plt.subplots(figsize=(figwidth, 3 + len(tbl_data) * row_height))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    table = ax.table(
        cellText=tbl_data, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    _style_table(table, fontsize=fontsize)
    table.scale(1, 1.6)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    return table


def _compute_hhi(df):
    """Compute HHI per route from carrier load shares."""
    rc = df.groupby(["route", "vehicle_carrier"])["executed_load"].sum().reset_index()
    rt = rc.groupby("route")["executed_load"].sum().rename("route_total")
    rc = rc.merge(rt, on="route")
    rc["share_sq"] = (rc["executed_load"] / rc["route_total"]) ** 2
    return rc.groupby("route")["share_sq"].sum().rename("hhi").reset_index()


def _hhi_risk(hhi_val):
    if hhi_val >= 0.50:
        return "HIGH"
    elif hhi_val >= 0.25:
        return "MOD"
    return "LOW"


class MonthlyReportGenerator:
    def __init__(self, s3_bucket, output_bucket, sns_topic_arn, volume_bucket):
        self.s3_bucket = s3_bucket
        self.output_bucket = output_bucket
        self.volume_bucket = volume_bucket
        self.sns_topic_arn = sns_topic_arn
        self.s3_client = boto3.client("s3")
        self.sns_client = boto3.client("sns")
        logger.info("MonthlyReportGenerator initialized")

    def get_previous_month(self):
        today = datetime.now()
        first = today.replace(day=1)
        prev = first - timedelta(days=1)
        year = f"R{prev.year}"
        month = f"M{prev.month:02d}"
        logger.info(f"Target period: {year} {month}")
        return year, month

    def load_data(self, year, month):
        logger.info("STARTING DATA LOAD")
        try:
            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key="input-data/data.csv")
            df_raw = pd.read_csv(io.BytesIO(obj["Body"].read()))
            logger.info(f"Loaded {len(df_raw)} rows")
            conn = duckdb.connect()
            query = f"""
                SELECT report_year, report_month, route, vrid, vehicle_carrier,
                       is_set, orig_eu5_country, dest_eu5_country, supply_type,
                       distance_band,
                       SUM(cost_for_cpkm_eur) * 1.17459 AS total_cost,
                       SUM(distance_for_cpkm) AS total_distance,
                       SUM(executed_loads) AS executed_load
                FROM df_raw
                WHERE report_year = '{year}' AND report_month = '{month}'
                GROUP BY ALL
            """
            df = conn.execute(query).df()
            conn.close()
            logger.info(f"Route data: {len(df)} records")

            vol_path = f"s3://{self.volume_bucket}/volume/volume_by_vrid.csv000"
            df_cubes = pd.read_csv(vol_path)

            df["cpkm"] = np.where(df["total_distance"] > 0, df["total_cost"] / df["total_distance"], 0)
            df["cpkm_route_z"] = (
                df.groupby(["report_year", "report_month", "route"])["cpkm"]
                .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            )
            df = df.merge(df_cubes, on="vrid", how="left").reset_index(drop=True)
            logger.info(f"Processing complete: {len(df)} records")
            return df
        except Exception:
            logger.error("Error loading data", exc_info=True)
            raise

    # --------------------------------------------------------------------------
    # Enrich data with MAD outlier flags
    # --------------------------------------------------------------------------
    def _enrich_outliers(self, df, mad_cutoff=3.0):
        df = df.copy()
        g = df.groupby("route")["cpkm"]
        med = g.transform("median")
        mad = (df["cpkm"] - med).abs().groupby(df["route"]).transform("median")
        mad_thr = med + mad_cutoff * mad / 0.6745
        df["mad_score"] = np.where(mad > 0, 0.6745 * (df["cpkm"] - med) / mad, np.nan)
        df["cost_over_mad"] = np.where(
            df["cpkm"] > mad_thr,
            (df["cpkm"] - mad_thr) * df["total_distance"] * df["executed_load"], 0.0,
        )
        df["is_outlier"] = df["mad_score"] > mad_cutoff
        logger.info(f"Outliers: {df['is_outlier'].sum()} of {len(df)} rows")
        return df

    # --------------------------------------------------------------------------
    # Supply Type Overview
    # --------------------------------------------------------------------------
    def _create_supply_type_summary(self, pdf, df, year, month):
        logger.info("Creating supply type summary...")
        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)
        supply_types = sorted(df["supply_type"].dropna().unique())
        total_loads_all = df["executed_load"].sum()
        hhi_df = _compute_hhi(df)
        rows = []
        for st in supply_types:
            vals = df.loc[df["supply_type"] == st, "cpkm"].dropna()
            sub = df[df["supply_type"] == st]
            nc = sub["vehicle_carrier"].nunique()
            tl = int(sub["executed_load"].sum())
            load_share = tl / total_loads_all * 100 if total_loads_all > 0 else 0
            st_routes = sub["route"].unique()
            st_hhi = hhi_df[hhi_df["route"].isin(st_routes)]["hhi"]
            avg_hhi = st_hhi.mean() if len(st_hhi) > 0 else 0
            rows.append([
                st, tl, f"{load_share:.1f}%", f"{sub['total_cost'].sum():,.0f}",
                f"{sub['cubes'].sum():,.0f}", sub["route"].nunique(), nc,
                f"{tl/nc:.1f}" if nc else "-",
                f"{vals.mean():.4f}", f"{vals.median():.4f}",
                f"{np.percentile(vals,2.5):.4f}", f"{np.percentile(vals,97.5):.4f}",
                f"{np.percentile(vals,25):.4f}", f"{np.percentile(vals,75):.4f}",
                f"{vals.std():.4f}", f"{avg_hhi:.3f}",
            ])
        _render_table_page(
            pdf, rows,
            ["Supply Type","Loads","Load Share","Total Cost","Volume m3","Routes","Carriers",
             "LCR","Mean CPKM","Median CPKM","P2.5","P97.5","P25","P75","Std","Avg HHI"],
            f"Supply Type Overview  -  {year} {month}",
            fontsize=9, figwidth=28,
        )

    # --------------------------------------------------------------------------
    # Overall outlier summary (all countries aggregated)
    # --------------------------------------------------------------------------
    def _create_overall_outlier_table(self, pdf, df, year, month, top_n=30):
        logger.info("Creating overall outlier summary...")
        out = df[df["is_outlier"]].copy()
        if out.empty:
            logger.warning("No outliers found")
            return

        route_agg = (
            df.groupby("route")
            .agg(
                orig=("orig_eu5_country", "first"), dest=("dest_eu5_country", "first"),
                total_cost=("total_cost", "sum"), total_cubes=("cubes", "sum"),
                total_loads=("executed_load", "sum"), n_carriers=("vehicle_carrier", "nunique"),
                mean_cpkm=("cpkm", "mean"), median_cpkm=("cpkm", "median"),
                cost_over_mad=("cost_over_mad", "sum"),
            ).reset_index()
        )
        route_agg["lcr"] = route_agg["total_loads"] / route_agg["n_carriers"]

        # Outlier counts per route per supply type
        out_by_st = (
            out.groupby(["route", "supply_type"])["executed_load"].sum()
            .unstack(fill_value=0).reset_index()
        )
        for st in SUPPLY_TYPES_ORDERED:
            if st not in out_by_st.columns:
                out_by_st[st] = 0
        out_totals = out.groupby("route").agg(
            out_loads=("executed_load", "sum"),
            out_cubes=("cubes", "sum"),
            out_cost_over_mad=("cost_over_mad", "sum"),
        ).reset_index()

        hhi = _compute_hhi(df)
        route_agg = route_agg.merge(hhi, on="route", how="left")
        route_agg = route_agg.merge(out_totals, on="route", how="left")
        route_agg = route_agg.merge(out_by_st[["route"] + SUPPLY_TYPES_ORDERED], on="route", how="left")
        route_agg = route_agg.fillna(0)

        # Only show routes with actual MAD cost > 0
        route_agg = route_agg[route_agg["out_cost_over_mad"] > 0]
        if route_agg.empty:
            return

        top = route_agg.sort_values("out_cost_over_mad", ascending=False).head(top_n)
        tbl_data = []
        for _, r in top.iterrows():
            tbl_data.append([
                r.route, f"{r.orig} -> {r.dest}",
                f"{r.total_cost:,.0f}", f"{r.total_cubes:,.0f}",
                int(r.total_loads), int(r.n_carriers), f"{r.lcr:.1f}",
                f"{r.mean_cpkm:.3f}", f"{r.median_cpkm:.3f}",
                int(r.out_loads), f"{r.out_cubes:,.0f}", f"{r.out_cost_over_mad:,.0f}",
                int(r.get("1.AZNG", 0)), int(r.get("2.RLB", 0)), int(r.get("3.3P", 0)),
                f"{r.hhi:.3f}", _hhi_risk(r.hhi),
            ])
        _render_table_page(
            pdf, tbl_data,
            ["Route", "Lane", "Cost", "Volume", "Loads", "Carriers", "LCR",
             "Mean", "Median", "Out Loads", "Out Vol", "MAD Cost",
             "AZNG Out", "RLB Out", "3P Out", "HHI", "Risk"],
            f"Overall Outlier Summary - Top {top_n} by MAD Over Cost  -  {year} {month}",
            fontsize=7, figwidth=30,
        )

    # --------------------------------------------------------------------------
    # Pivot tables: orig x dest x distance bands
    # --------------------------------------------------------------------------
    def _create_outlier_pivots(self, pdf, df, year, month):
        logger.info("Creating outlier pivot tables...")
        out = df[df["is_outlier"]].copy()
        if out.empty:
            return

        for value_col, agg_func, label in [
            ("executed_load", "sum", "Number of Outlier Loads"),
            ("cubes", "sum", "Volume of Outliers (m3)"),
            ("cost_over_mad", "sum", "MAD Cost of Outliers"),
        ]:
            out["lane"] = out["orig_eu5_country"] + " -> " + out["dest_eu5_country"]
            pivot = (
                out.groupby(["lane", "distance_band"])[value_col]
                .agg(agg_func)
                .unstack(fill_value=0)
            )
            band_order = sorted(pivot.columns)
            pivot = pivot.reindex(columns=band_order, fill_value=0)
            pivot["TOTAL"] = pivot.sum(axis=1)
            pivot = pivot.sort_values("TOTAL", ascending=False)

            fmt = lambda x: f"{x:,.0f}" if abs(x) >= 1 else f"{x:.2f}"
            tbl_data = []
            for lane, row in pivot.iterrows():
                tbl_data.append([lane] + [fmt(row[c]) for c in pivot.columns])

            _render_table_page(
                pdf, tbl_data, ["Lane"] + list(pivot.columns),
                f"{label} by Lane x Distance Band  -  {year} {month}",
                fontsize=8, figwidth=24,
            )

    # --------------------------------------------------------------------------
    # Per-country tables
    # --------------------------------------------------------------------------
    def _create_per_country_tables(self, pdf, df, year, month, top_n=20):
        logger.info("Creating per-country tables...")
        countries = sorted(df["orig_eu5_country"].dropna().unique())
        hhi = _compute_hhi(df)

        for country in countries:
            df_c = df[
                (df["orig_eu5_country"] == country) | (df["dest_eu5_country"] == country)
            ]
            if df_c.empty:
                continue

            route_agg = (
                df_c.groupby("route").agg(
                    orig=("orig_eu5_country", "first"), dest=("dest_eu5_country", "first"),
                    total_cost=("total_cost", "sum"), total_cubes=("cubes", "sum"),
                    total_loads=("executed_load", "sum"), n_carriers=("vehicle_carrier", "nunique"),
                    mean_cpkm=("cpkm", "mean"), median_cpkm=("cpkm", "median"),
                    cost_over_mad=("cost_over_mad", "sum"),
                ).reset_index()
            )
            route_agg["lcr"] = route_agg["total_loads"] / route_agg["n_carriers"]
            route_agg = route_agg.merge(hhi, on="route", how="left").fillna(0)

            # Outlier counts per supply type
            out_c = df_c[df_c["is_outlier"]]
            out_by_st = (
                out_c.groupby(["route", "supply_type"])["executed_load"].sum()
                .unstack(fill_value=0).reset_index()
            )
            for st in SUPPLY_TYPES_ORDERED:
                if st not in out_by_st.columns:
                    out_by_st[st] = 0
            out_totals = out_c.groupby("route").agg(
                out_loads=("executed_load", "sum"),
                out_cost_over_mad=("cost_over_mad", "sum"),
            ).reset_index()
            route_agg = route_agg.merge(out_totals, on="route", how="left")
            route_agg = route_agg.merge(out_by_st[["route"] + SUPPLY_TYPES_ORDERED], on="route", how="left")
            route_agg = route_agg.fillna(0)

            for sort_col, label in [
                ("total_cubes", "Volume"), ("total_cost", "Cost"), ("cost_over_mad", "MAD Cost"),
            ]:
                top = route_agg.sort_values(sort_col, ascending=False).head(top_n)
                if top.empty:
                    continue
                tbl_data = []
                for _, r in top.iterrows():
                    tbl_data.append([
                        r.route, f"{r.orig} -> {r.dest}",
                        f"{r.total_cost:,.0f}", f"{r.total_cubes:,.0f}",
                        int(r.total_loads), int(r.n_carriers), f"{r.lcr:.1f}",
                        f"{r.mean_cpkm:.3f}", f"{r.median_cpkm:.3f}",
                        int(r.out_loads), f"{r.out_cost_over_mad:,.0f}",
                        int(r.get("1.AZNG", 0)), int(r.get("2.RLB", 0)), int(r.get("3.3P", 0)),
                        f"{r.hhi:.3f}", _hhi_risk(r.hhi),
                    ])
                _render_table_page(
                    pdf, tbl_data,
                    ["Route", "Lane", "Cost", "Volume", "Loads", "Carriers", "LCR",
                     "Mean", "Median", "Out Loads", "MAD Cost",
                     "AZNG Out", "RLB Out", "3P Out", "HHI", "Risk"],
                    f"{country} - Top {top_n} Routes by {label}  -  {year} {month}",
                    fontsize=7, figwidth=28,
                )

    # --------------------------------------------------------------------------
    # HHI scatter + tables (at beginning)
    # --------------------------------------------------------------------------
    def _create_carrier_concentration(self, pdf, df, year, month, top_n=30):
        logger.info("Creating carrier concentration analysis...")
        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        hhi = _compute_hhi(df)
        route_stats = (
            df.groupby("route").agg(
                total_cost=("total_cost", "sum"), total_cubes=("cubes", "sum"),
                total_loads=("executed_load", "sum"),
                median_cpkm=("cpkm", "median"), mean_cpkm=("cpkm", "mean"),
                n_carriers=("vehicle_carrier", "nunique"),
            ).reset_index()
        )
        route_stats["lcr"] = route_stats["total_loads"] / route_stats["n_carriers"]
        route_supply = (
            df.groupby(["route", "supply_type"]).agg(
                st_cost=("total_cost", "sum"), st_cubes=("cubes", "sum"),
                st_loads=("executed_load", "sum"), st_carriers=("vehicle_carrier", "nunique"),
                st_mean=("cpkm", "mean"), st_median=("cpkm", "median"),
            ).reset_index()
        )
        hhi = hhi.merge(route_stats, on="route")
        hhi = hhi[hhi["total_loads"] >= 5].copy()

        # Tables
        def _build_table(sort_col, label):
            top = hhi.sort_values(sort_col, ascending=False).head(top_n)
            tbl_data = []
            for _, r in top.iterrows():
                rs = route_supply[route_supply["route"] == r.route]
                parts = []
                for _, s in rs.iterrows():
                    slcr = s.st_loads / s.st_carriers if s.st_carriers > 0 else 0
                    parts.append(
                        f"{s.supply_type}: C={s.st_cost:,.0f} V={s.st_cubes:,.0f} "
                        f"L={int(s.st_loads)} LCR={slcr:.1f} u={s.st_mean:.3f} m={s.st_median:.3f}"
                    )
                tbl_data.append([
                    r.route, f"{r.total_cost:,.0f}", f"{r.total_cubes:,.0f}",
                    int(r.total_loads), int(r.n_carriers), f"{r.lcr:.1f}",
                    f"{r.mean_cpkm:.3f}", f"{r.median_cpkm:.3f}",
                    f"{r.hhi:.3f}", _hhi_risk(r.hhi), " | ".join(parts),
                ])
            fig, ax = plt.subplots(figsize=(28, 3 + len(tbl_data) * 0.6))
            ax.axis("off")
            ax.set_title(f"HHI - Top {top_n} by {label}  -  {year} {month}",
                         fontsize=16, fontweight="bold", pad=20)
            table = ax.table(
                cellText=tbl_data,
                colLabels=["Route","Cost","Volume","Loads","Carriers","LCR",
                           "Mean","Median","HHI","Risk","Supply Type Breakdown"],
                cellLoc="center", loc="center",
            )
            _style_table(table, fontsize=8)
            table.scale(1, 1.6)
            for ri in range(len(tbl_data) + 1):
                table[ri, 10].set_text_props(fontsize=7, ha="left")
            for ri in range(1, len(tbl_data) + 1):
                rv = tbl_data[ri - 1][9]
                cell = table[ri, 9]
                if rv == "HIGH":
                    cell.set_facecolor("#FF6B6B")
                    cell.set_text_props(color="white", fontweight="bold")
                elif rv == "MOD":
                    cell.set_facecolor("#FFD93D")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        _build_table("total_cost", "Cost")
        _build_table("total_cubes", "Volume")

    # --------------------------------------------------------------------------
    # Top routes overview (histograms + box + stats)
    # --------------------------------------------------------------------------
    def _create_top_routes_overview(self, pdf, df, year, month, top_n=15,
                                    top_filter="total_cost", title_label="Cost"):
        logger.info(f"Creating top {top_n} routes by {title_label}...")
        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)

        top_routes = (
            df.groupby("route", as_index=False)[top_filter].sum()
            .sort_values(top_filter, ascending=False)
            .head(top_n)["route"].tolist()
        )
        df_top = df[df["route"].isin(top_routes)].copy()
        supply_types = sorted(df["supply_type"].dropna().unique())
        palette = sns.color_palette(SEABORN_PALETTE, len(supply_types))
        color_map = {st: palette[i] for i, st in enumerate(supply_types)}

        hhi_df = _compute_hhi(df)

        for route in top_routes:
            df_r = df_top[df_top["route"] == route]
            vals = df_r["cpkm"].dropna()
            if vals.empty:
                continue

            origin = df_r["orig_eu5_country"].iloc[0]
            dest = df_r["dest_eu5_country"].iloc[0]
            route_hhi_row = hhi_df[hhi_df["route"] == route]
            route_hhi = route_hhi_row["hhi"].iloc[0] if len(route_hhi_row) else 0
            risk = _hhi_risk(route_hhi)

            fig = plt.figure(figsize=(24, 16))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
            ax_hist = fig.add_subplot(gs[0, 0])
            ax_box = fig.add_subplot(gs[0, 1])
            ax_table = fig.add_subplot(gs[1, :])
            ax_table.axis("off")

            hhi_color = "#FF6B6B" if risk == "HIGH" else "#FFD93D" if risk == "MOD" else "#6BCB77"
            fig.suptitle(
                f"Top Routes by {title_label}  -  {year} {month}\n"
                f"{route} | {origin} -> {dest}  |  HHI: {route_hhi:.3f} ({risk})",
                fontsize=18, fontweight="bold")

            bin_edges = np.histogram_bin_edges(vals, bins=30)
            route_total_loads = df_r["executed_load"].sum()
            for st in df_r["supply_type"].dropna().unique():
                sv = df_r.loc[df_r["supply_type"] == st, "cpkm"].dropna()
                st_loads = df_r.loc[df_r["supply_type"] == st, "executed_load"].sum()
                st_share = st_loads / route_total_loads * 100 if route_total_loads > 0 else 0
                if sv.empty:
                    continue
                ax_hist.hist(sv, bins=bin_edges, alpha=0.6,
                             label=f"{st} ({st_share:.0f}% | u={sv.mean():.2f}, m={sv.median():.2f})",
                             edgecolor="black", color=color_map.get(st))
            ax_hist.set_title("CPKM by Supply Type", fontsize=13, fontweight="bold")
            ax_hist.set_xlabel("CPKM"); ax_hist.set_ylabel("Loads")
            ax_hist.legend(fontsize=9); ax_hist.grid(True, alpha=0.3)

            st_present = sorted(df_r["supply_type"].dropna().unique())
            if st_present:
                sns.boxplot(data=df_r, x="supply_type", y="cpkm", hue="supply_type",
                            palette=color_map, ax=ax_box, showfliers=True, fliersize=3,
                            order=st_present, legend=False)
            ax_box.set_title("CPKM Box Plot", fontsize=13, fontweight="bold")
            ax_box.set_xlabel(""); ax_box.set_ylabel("CPKM")

            table_rows = []
            for st in st_present:
                sub = df_r[df_r["supply_type"] == st]
                sv = sub["cpkm"].dropna()
                nc = sub["vehicle_carrier"].nunique()
                sl = int(sub["executed_load"].sum())
                st_share = sl / route_total_loads * 100 if route_total_loads > 0 else 0
                table_rows.append([
                    st, f"{sub['total_cost'].sum():,.0f}",
                    f"{sub['cubes'].sum():,.0f}", sl, f"{st_share:.1f}%", nc,
                    f"{sl/nc:.1f}" if nc else "-",
                    f"{sv.mean():.3f}" if len(sv) else "-",
                    f"{sv.median():.3f}" if len(sv) else "-",
                    f"{np.percentile(sv,25):.3f}" if len(sv) else "-",
                    f"{np.percentile(sv,75):.3f}" if len(sv) else "-",
                ])
            nc_all = df_r["vehicle_carrier"].nunique()
            tl_all = int(df_r["executed_load"].sum())
            table_rows.append([
                "OVERALL", f"{df_r['total_cost'].sum():,.0f}",
                f"{df_r['cubes'].sum():,.0f}", tl_all, "100%", nc_all,
                f"{tl_all/nc_all:.1f}" if nc_all else "-",
                f"{vals.mean():.3f}", f"{vals.median():.3f}",
                f"{np.percentile(vals,25):.3f}", f"{np.percentile(vals,75):.3f}",
            ])
            n_cols = 11
            tbl = ax_table.table(
                cellText=table_rows,
                colLabels=["Supply Type","Cost","Volume m3","Loads","Load Share","Carriers",
                           "LCR","Mean","Median","P25","P75"],
                cellLoc="center", loc="center",
            )
            _style_table(tbl, fontsize=10)
            tbl.scale(1, 1.8)
            nr = len(table_rows)
            for ci in range(n_cols):
                tbl[nr, ci].set_text_props(fontweight="bold")
                tbl[nr, ci].set_facecolor("#D6E4F0")

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    # --------------------------------------------------------------------------
    # MAD analysis per supply type
    # --------------------------------------------------------------------------
    def _filter_mad_routes(self, df, routes, cpkm_col="cpkm", min_obs=3):
        valid = []
        for route in routes:
            vals = df[df["route"] == route][cpkm_col].dropna()
            if len(vals) < min_obs:
                continue
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            if mad > 0 and not np.isnan(mad):
                valid.append(route)
        return valid

    def plot_supply_type_analysis(self, pdf, df_full, supply_type, cpkm_column="cpkm",
                                  mad_cutoff=3, top_n=8):
        logger.info(f"MAD analysis: {supply_type}")
        df_supply = df_full[df_full["supply_type"] == supply_type].copy()
        if df_supply.empty:
            return

        df_supply = self._enrich_outliers(df_supply, mad_cutoff)

        valid = self._filter_mad_routes(df_supply, df_supply["route"].unique(), min_obs=5)
        if not valid:
            return
        routes_top = (
            df_supply[df_supply["route"].isin(valid)]
            .groupby("route")["cost_over_mad"].sum()
            .sort_values(ascending=False).head(top_n).index.tolist()
        )
        routes_filtered = self._filter_mad_routes(df_supply, routes_top, min_obs=3)
        if not routes_filtered:
            return

        self._create_mad_plots(
            pdf, df_full, df_supply, routes_filtered, supply_type, cpkm_column,
            mad_cutoff, hue_column="vehicle_carrier",
            title_suffix=f"MAD Outlier Analysis - {supply_type}",
        )
        self._create_mad_plots(
            pdf, df_full, df_supply, routes_filtered, supply_type, cpkm_column,
            mad_cutoff, hue_column="is_set",
            title_suffix=f"MAD Outlier Analysis - {supply_type} - Split by SET",
        )

    def _create_mad_plots(self, pdf, df_full, df_supply, routes, supply_type,
                           cpkm_column, mad_cutoff, hue_column="vehicle_carrier",
                           title_suffix=""):
        sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)
        hhi_df = _compute_hhi(df_full)

        for route in routes:
            df_r = df_supply[df_supply["route"] == route].copy()
            vals = df_r[cpkm_column].dropna()
            if vals.empty:
                continue

            median = np.median(vals)
            mad = np.median(np.abs(vals - median))
            if mad == 0 or np.isnan(mad):
                continue

            mad_threshold = median + mad_cutoff * mad / 0.6745
            df_r["mad_score"] = 0.6745 * (df_r[cpkm_column] - median) / mad
            df_r["cost_over_threshold"] = np.where(
                df_r[cpkm_column] > mad_threshold,
                (df_r[cpkm_column] - mad_threshold) * df_r["total_distance"] * df_r["executed_load"],
                0.0,
            )
            outliers = df_r[df_r["mad_score"] > mad_cutoff]

            # HHI for this route
            route_hhi_row = hhi_df[hhi_df["route"] == route]
            route_hhi = route_hhi_row["hhi"].iloc[0] if len(route_hhi_row) else 0
            risk = _hhi_risk(route_hhi)
            hhi_color = "#FF6B6B" if risk == "HIGH" else "#FFD93D" if risk == "MOD" else "#6BCB77"

            fig = plt.figure(figsize=(24, 14))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 2])

            ax_plot = fig.add_subplot(gs[0])
            ax_tables = fig.add_subplot(gs[1])
            ax_tables.axis("off")

            fig.suptitle(
                f"{title_suffix}\n{route} | "
                f"{df_r.orig_eu5_country.iloc[0]} -> {df_r.dest_eu5_country.iloc[0]}"
                f"  |  HHI: {route_hhi:.3f} ({risk})",
                fontsize=16, fontweight="bold", y=0.98,
            )
            # Color the HHI part of the title
            fig.patches.append(plt.Rectangle(
                (0.82, 0.96), 0.16, 0.035, transform=fig.transFigure,
                facecolor=hhi_color, alpha=0.3, edgecolor="none",
            ))

            # ---- Histogram ----
            plot_hue = hue_column
            if hue_column == "vehicle_carrier":
                oc = set(outliers["vehicle_carrier"].unique())
                df_r["plot_carrier"] = np.where(
                    df_r["vehicle_carrier"].isin(oc), df_r["vehicle_carrier"], "OK")
                plot_hue = "plot_carrier"

            sns.histplot(data=df_r, x=cpkm_column, hue=plot_hue, bins=50, kde=True,
                         stat="count", element="step", alpha=0.55, ax=ax_plot)
            ax_plot.axvline(mad_threshold, color="red", ls="--", lw=2.5,
                            label=f"MAD cutoff ({mad_cutoff})")
            ax_plot.set_xlabel("CPKM"); ax_plot.set_ylabel("Loads")

            # ---- Summary table ----
            nc = df_r["vehicle_carrier"].nunique()
            tl = int(df_r["executed_load"].sum())
            lcr = tl / nc if nc else 0
            tc = df_r["cubes"].sum() if "cubes" in df_r.columns else 0

            summary = [
                ["Loads", f"{tl}"], ["Carriers", f"{nc}"], ["LCR", f"{lcr:.1f}"],
                ["Cost", f"{df_r.total_cost.sum():,.0f}"], ["Volume m3", f"{tc:,.0f}"],
                ["Cost over MAD", f"{df_r.cost_over_threshold.sum():,.0f}"],
                ["Mean CPKM", f"{vals.mean():.3f}"], ["Median", f"{median:.3f}"],
                ["MAD", f"{mad:.3f}"], ["Threshold", f"{mad_threshold:.3f}"],
                ["HHI", f"{route_hhi:.3f} ({risk})"],
            ]
            s_tbl = ax_tables.table(
                cellText=summary, colLabels=["Metric", "Value"],
                cellLoc="center", loc="center",
            )
            _style_table(s_tbl, fontsize=10)
            s_tbl.scale(1, 1.5)

            # Color HHI row
            hhi_row_idx = len(summary)
            for ci in range(2):
                s_tbl[hhi_row_idx, ci].set_facecolor(hhi_color)
                s_tbl[hhi_row_idx, ci].set_text_props(fontweight="bold")
                if risk == "HIGH":
                    s_tbl[hhi_row_idx, ci].set_text_props(color="white", fontweight="bold")

            plt.tight_layout(rect=[0, 0.01, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    # --------------------------------------------------------------------------
    def generate_report(
        self,
        enable_top_routes_by_cost=True,
        enable_top_routes_by_volume=True,
        enable_mad_analysis=True,
        enable_carrier_concentration=True,
        enable_outlier_summary=True,
        enable_outlier_pivots=True,
        enable_per_country=True,
    ):
        logger.info("=" * 80)
        logger.info("STARTING REPORT GENERATION")
        logger.info("=" * 80)

        year, month = self.get_previous_month()
        df = self.load_data(year, month)

        # Enrich with outlier flags globally
        df = self._enrich_outliers(df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"monthly_report_{year}_{month}_{timestamp}.pdf"
        local_pdf_path = f"/tmp/{pdf_filename}"

        with PdfPages(local_pdf_path) as pdf:
            # Title
            sns.set_theme(style="whitegrid", palette=SEABORN_PALETTE)
            fig = plt.figure(figsize=(22, 14))
            fig.text(0.5, 0.55, f"Monthly CPKM Report\n{year} {month}",
                     ha="center", va="center", fontsize=32, fontweight="bold")
            fig.text(0.5, 0.40, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                     ha="center", va="center", fontsize=14, color="grey")
            pdf.savefig(fig); plt.close()

            # 1. Supply type overview
            self._create_supply_type_summary(pdf, df, year, month)

            # 2. Overall outlier summary (aggregated)
            if enable_outlier_summary:
                self._create_overall_outlier_table(pdf, df, year, month)

            # 3. Pivot tables (orig x dest x distance bands)
            if enable_outlier_pivots:
                self._create_outlier_pivots(pdf, df, year, month)

            # 4. HHI scatter + tables (at beginning)
            if enable_carrier_concentration:
                self._create_carrier_concentration(pdf, df, year, month)

            # 5. Per-country tables
            if enable_per_country:
                self._create_per_country_tables(pdf, df, year, month)

            # 6. Top routes by cost
            if enable_top_routes_by_cost:
                self._create_top_routes_overview(pdf, df, year, month,
                                                  top_n=15, top_filter="total_cost", title_label="Cost")

            # 7. Top routes by volume
            if enable_top_routes_by_volume:
                self._create_top_routes_overview(pdf, df, year, month,
                                                  top_n=15, top_filter="cubes", title_label="Volume")

            # 8. MAD analysis per supply type
            if enable_mad_analysis:
                for st in SUPPLY_TYPES_ORDERED:
                    self.plot_supply_type_analysis(pdf, df, st)

            info = pdf.infodict()
            info["Title"] = f"Monthly CPKM Report {year} {month}"
            info["Author"] = "Automated Report System"
            info["CreationDate"] = datetime.now()

        s3_key = f"reports/{year}/{month}/{pdf_filename}"
        self.s3_client.upload_file(local_pdf_path, self.output_bucket, s3_key)
        pdf_url = self.s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": self.output_bucket, "Key": s3_key},
            ExpiresIn=604800,
        )
        self.send_notification(year, month, pdf_url, s3_key)
        return pdf_url

    def send_notification(self, year, month, pdf_url, s3_key):
        self.sns_client.publish(
            TopicArn=self.sns_topic_arn,
            Subject=f"Monthly CPKM Report - {year} {month}",
            Message=f"Monthly CPKM Report Generated\n\nPeriod: {year} {month}\n\n"
                    f"Download (7 days): {pdf_url}\n\n"
                    f"S3: s3://{self.output_bucket}/{s3_key}",
        )


def main():
    logger.info("ECS TASK STARTED")
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
        enable_outlier_summary=True,
        enable_outlier_pivots=True,
        enable_per_country=True,
    )
    logger.info("ECS TASK COMPLETED")


if __name__ == "__main__":
    main()
