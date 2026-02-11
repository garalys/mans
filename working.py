import matplotlib.pyplot as plt
import numpy as np

def plot_cpkm_histograms_top_routes(
    df,
    top_n=10,
    bins=12,
    title_suffix='',
    top_filter='total_cost',
    cpkm_column='cpkm',
    hue_column='supply_type'
):
    """
    Plots CPKM histograms with summary tables for top routes by executed loads,
    colored by supply type.
    """

    # ---- Identify top routes by load volume ----
    top_routes = (
        df.groupby('route', as_index=False)[top_filter]
          .sum()
          .sort_values(top_filter, ascending=False)
          .head(top_n)['route']
          .tolist()
    )

    df_top = df[df['route'].isin(top_routes)]

    # ---- Colors for supply types (auto) ----
    supply_types = df[hue_column].dropna().unique()
    cmap = plt.get_cmap('tab10')
    color_map = {st: cmap(i) for i, st in enumerate(supply_types)}

    # ---- Layout ----
    n_cols = 5
    n_rows = int(np.ceil(top_n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 6 * n_rows))
    fig.suptitle(title_suffix, fontsize=16, fontweight='bold')

    axes = np.array(axes).flatten()

    # ---- Plot per route ----
    for idx, route in enumerate(top_routes):
        ax = axes[idx]
        df_r = df_top[df_top['route'] == route]

        if df_r[cpkm_column].dropna().empty:
            ax.set_title(f'{route}\n(No variance)', fontsize=10)
            ax.axis('off')
            continue

        loads = df_r['executed_load']
        total_cost = df_r['total_cost'].sum()
        total_volume = df_r['cubes'].sum()

        carrier = df_r['vehicle_carrier'].iloc[0]
        origin = df_r['orig_eu5_country'].iloc[0]
        dest = df_r['dest_eu5_country'].iloc[0]

        # ---- Shared bins per route ----
        all_vals = df_r[cpkm_column].dropna()
        bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

        # ---- Histogram per supply type ----
        for st in df_r[hue_column].dropna().unique():
            vals = df_r.loc[df_r[hue_column] == st, cpkm_column].dropna()
            if vals.empty:
                continue

            mean_cpkm = vals.mean()
            median_cpkm = vals.median()

            label = f"{st} (μ={mean_cpkm:.2f}, m={median_cpkm:.2f})"

            ax.hist(
                vals,
                bins=bin_edges,
                alpha=0.6,
                label=label,
                edgecolor='black',
                color=color_map.get(st)
            )


        ax.set_title(
            f'{route} | {origin} → {dest}\n{carrier}',
            fontsize=10,
            fontweight='bold'
        )

        ax.set_xlabel('CPKM', fontsize=9)
        ax.set_ylabel('Number of Loads', fontsize=9)
        ax.grid(True, alpha=0.3)

        ax.legend(fontsize=8)

        # ---- Summary table ----
        table_data = [
            ['Total Loads', f'{int(loads.sum())}'],
            ['Total Cost USD', f'{total_cost:,.0f}'],
            ['Total Volume m3', f'{total_volume:,.0f}'],
            ['Mean CPKM', f'{all_vals.mean():.3f}'],
            ['Median CPKM', f'{all_vals.median():.3f}'],
            ['P25', f'{np.percentile(all_vals, 25):.3f}'],
            ['P75', f'{np.percentile(all_vals, 75):.3f}'],
        ]

        table = ax.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            bbox=[0, -0.55, 1, 0.45]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)

    # ---- Remove unused axes ----
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
  
plot_cpkm_histograms_top_routes(
    df_routes,
    top_n=20,
    bins=30,
    title_suffix='R2025 M10 - CPKM Distribution of Top Routes by Cost',
    cpkm_column='cpkm',
    top_filter='total_cost',
)

plot_cpkm_histograms_top_routes(
    df_routes,
    top_n=20,
    bins=30,
    title_suffix='R2025 M10 - CPKM Distribution of Top Routes by Volume',
    cpkm_column='cpkm',
    top_filter='cubes',
)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_route_metrics(df):
    return (
        df.groupby(['route', 'supply_type'])
        .agg(
            total_loads=('executed_load', 'sum'),
            total_cost=('total_cost', 'sum'),
            cpkm_var=('cpkm', 'var'),
            cpkm_std=('cpkm', 'std'),
            obs=('cpkm', 'count')
        )
        .reset_index()
    )
    
def routes_with_variance(route_metrics, min_obs=3):
    """
    Returns routes that have real variance
    (enough observations AND non-zero std).
    """
    return (
        route_metrics
        .groupby('route')
        .agg(
            obs=('obs', 'sum'),
            std=('cpkm_std', 'mean')
        )
        .query('obs >= @min_obs and std > 0')
        .index
    )


def top_routes_by_revenue(route_metrics, top_n=10, min_obs=3):
    valid_routes = routes_with_variance(route_metrics, min_obs)

    return (
        route_metrics
        .query('route in @valid_routes')
        .groupby('route')['total_cost']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )


def top_routes_by_variance(route_metrics, top_n=10, min_obs=3):
    valid_routes = routes_with_variance(route_metrics, min_obs)

    return (
        route_metrics
        .query('route in @valid_routes')
        .groupby('route')['cpkm_var']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )


def deep_dive_supply_type(df, supply_type):
    return df[df['supply_type'] == supply_type]

def compute_mad_outliers(df, cpkm_col='cpkm'):
    df = df.copy()

    g = df.groupby('route')[cpkm_col]
    med = g.transform('median')
    mad = (df[cpkm_col] - med).abs().groupby(df['route']).transform('median')

    df['mad_score'] = np.where(
        mad > 0,
        0.6745 * (df[cpkm_col] - med) / mad,
        np.nan
    )

    return df[df['mad_score'] > 3.5]


def add_cost_over_mad(df, cpkm_col='cpkm', mad_cutoff=3.0):
    df = df.copy()

    g = df.groupby('route')[cpkm_col]
    median = g.transform('median')
    mad = (df[cpkm_col] - median).abs().groupby(df['route']).transform('median')

    mad_threshold = median + mad_cutoff * mad / 0.6745

    df['cost_over_mad'] = np.where(
        df[cpkm_col] > mad_threshold,
        (df[cpkm_col] - mad_threshold)
        * df['total_distance']
        * df['executed_load'],
        0.0
    )

    return df
    
def top_routes_by_cost_over_mad(df, route_metrics, top_n=10, min_obs=3):

    valid_routes = routes_with_variance(route_metrics, min_obs)
    
    return (
        df[df['route'].isin(valid_routes)]
        .groupby('route')['cost_over_mad']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )




def plot_cpkm_histograms_top_routes_mad(
    df,
    top_routes,
    top_n=6,
    bins=20,
    title_suffix='',
    cpkm_column='cpkm',
    hue_column='is_set',
    mad_cutoff=3.0,
    max_outlier_rows=6
):
    """
    Clean, readable MAD-based distribution analysis using seaborn.
    Special behavior:
    - if hue_column == 'vehicle_carrier', only outlier carriers get colors,
      all others are grouped as 'OK'.
    """

    df_top=df.copy()

    sns.set_theme(style="whitegrid")

    # ---- Layout ----
    n_cols = 2
    n_rows = int(np.ceil(top_n / n_cols))

    fig = plt.figure(figsize=(24, 12 * n_rows))
    gs = fig.add_gridspec(n_rows * 2, n_cols, height_ratios=[3, 2] * n_rows)

    fig.suptitle(title_suffix, fontsize=22, fontweight='bold', y=0.995)

    for idx, route in enumerate(top_routes):
        row = (idx // n_cols) * 2
        col = idx % n_cols

        ax_plot = fig.add_subplot(gs[row, col])
        ax_table = fig.add_subplot(gs[row + 1, col])
        ax_table.axis('off')

        df_r = df_top[df_top['route'] == route].copy()
        vals = df_r[cpkm_column].dropna()

        if vals.empty:
            ax_plot.set_title(f'{route} (no variance)')
            ax_plot.axis('off')
            continue

        # ---- MAD stats ----
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))

        if mad == 0 or np.isnan(mad):
            ax_plot.set_title(f'{route} (no MAD variance)')
            ax_plot.axis('off')
            continue

        mad_threshold = median + mad_cutoff * mad / 0.6745
        df_r['mad_score'] = 0.6745 * (df_r[cpkm_column] - median) / mad
        df_r['cost_over_threshold'] = np.where(
            df_r[cpkm_column] > mad_threshold,
            (df_r[cpkm_column] - mad_threshold)
            * df_r['total_distance']
            * df_r['executed_load'],
            0.0
        )


        outliers = df_r[df_r['mad_score'] > mad_cutoff]

        # =========================================================
        # 🔥 NEW LOGIC: carrier hue = only outliers, rest = "OK"
        # =========================================================
        plot_hue = hue_column

        if hue_column == 'vehicle_carrier':
            outlier_carriers = set(outliers['vehicle_carrier'].unique())

            df_r['plot_carrier'] = np.where(
                df_r['vehicle_carrier'].isin(outlier_carriers),
                df_r['vehicle_carrier'],
                'OK'
            )

            plot_hue = 'plot_carrier'

        # ---- DISTRIBUTION (seaborn) ----
        sns.histplot(
            data=df_r,
            x=cpkm_column,
            hue=plot_hue,
            bins=bins,
            kde=True,
            stat='count',
            element='step',
            alpha=0.55,
            ax=ax_plot
        )

        # ---- MAD cutoff line ----
        ax_plot.axvline(
            mad_threshold,
            color='red',
            linestyle='--',
            linewidth=3,
            label=f'MAD cutoff ({mad_cutoff})'
        )

        ax_plot.set_title(
            f'{route} | {df_r.orig_eu5_country.iloc[0]} → {df_r.dest_eu5_country.iloc[0]}',
            fontsize=15,
            fontweight='bold'
        )

        ax_plot.set_xlabel('CPKM', fontsize=12)
        ax_plot.set_ylabel('Number of loads', fontsize=12)

        # ---- SUMMARY TABLE ----
        summary_data = [
            ['Loads', int(df_r.executed_load.sum())],
            ['Total cost €', f'€{df_r.total_cost.sum():,.0f}'],
            ['Cost over MAD €', f'€{df_r.cost_over_threshold.sum():,.0f}'],
            ['Median CPKM', f'{median:.3f}'],
            ['MAD', f'{mad:.3f}'],
            ['MAD cutoff', f'{mad_threshold:.3f}'],
        ]

        

        summary_tbl = ax_table.table(
            cellText=summary_data,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            bbox=[0.00, 0.55, 1.00, 0.40]
        )

        summary_tbl.auto_set_font_size(False)
        summary_tbl.set_fontsize(12)
        summary_tbl.scale(1.1, 1.6)

        # ---- OUTLIER TABLE (unchanged) ----
        if hue_column == 'vehicle_carrier':
            group_collumn = 'is_set'
        else: 
            group_collumn = hue_column
        
        if not outliers.empty:
            out_tbl_df = (
                outliers
                .groupby(['vehicle_carrier', group_collumn])
                .agg(
                    Loads=('executed_load', 'sum'),
                    Avg_MAD=('mad_score', 'mean'),
                    Max_MAD=('mad_score', 'max'),
                    Cost=('total_cost', 'sum'),
                    Distance=('total_distance', 'sum'),
                    Cost_Over_Threshold=('cost_over_threshold', 'sum')
                )
                .sort_values('Cost_Over_Threshold', ascending=False)
            )
            out_tbl_df['Outlier_CPKM'] = np.where(
                out_tbl_df['Distance'] > 0,
                out_tbl_df['Cost'] / out_tbl_df['Distance'],
                np.nan
            )

            out_tbl_df['Outlier_Cost_Over_Threshold_Cents'] = np.where(
                out_tbl_df['Distance'] > 0,
                out_tbl_df['Cost_Over_Threshold'] / out_tbl_df['Distance'],
                np.nan
            )


            extra = max(0, len(out_tbl_df) - max_outlier_rows)
            out_tbl_df = out_tbl_df.head(max_outlier_rows)

            out_rows = [
                [
                    carrier,
                    st,
                    int(r.Loads),
                    f'{r.Avg_MAD:.2f}',
                    f'{r.Max_MAD:.2f}',
                    f'€{r.Cost:,.0f}',
                    f'{r.Distance:,.0f}',
                    f'€{r.Cost_Over_Threshold:,.0f}',
                    f'€{r.Outlier_CPKM:,.2f}',
                    f'€{r.Outlier_Cost_Over_Threshold_Cents:,.2f}'
                ]
                for (carrier, st), r in out_tbl_df.iterrows()
            ]


            if extra > 0:
                out_rows.append([
                    '+ more',
                    '',
                    f'{extra} rows',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    ''
                ])


            out_tbl = ax_table.table(
                cellText=out_rows,
                colLabels=[
                    'Carrier',
                    group_collumn,
                    'Loads',
                    'Avg MAD',
                    'Max MAD',
                    'Total Cost €',
                    'Distance km',
                    'Over MAD €',
                    'CPKM',
                    'CPKM over MAD'
                ],
                cellLoc='center',
                bbox=[0.00, 0.00, 1.00, 0.50]
            )

            out_tbl.auto_set_font_size(False)
            out_tbl.set_fontsize(12)
            out_tbl.scale(1.1, 1.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.show()



def filter_routes_with_mad_variance(df, routes, cpkm_col='cpkm', min_obs=3):
    """
    Filter routes to only those with non-zero MAD.
    This is stricter than std-based filtering.
    """
    valid_routes = []

    for route in routes:
        vals = df[df['route'] == route][cpkm_col].dropna()

        if len(vals) < min_obs:
            continue

        median = np.median(vals)
        mad = np.median(np.abs(vals - median))

        if mad > 0 and not np.isnan(mad):
            valid_routes.append(route)

    return valid_routes



def supply_type_visualisation(df, supply_type, cpkm_column, mad_cutoff, top_n, hue='vehicle_carrier'):

    df_supply = deep_dive_supply_type(df, supply_type=supply_type)

    route_metrics = compute_route_metrics(df_supply)

    #routes_top_rev = top_routes_by_revenue(route_metrics, top_n)

    df_supply = add_cost_over_mad(df_supply, cpkm_col=cpkm_column, mad_cutoff=mad_cutoff)

    routes_top_mad = top_routes_by_cost_over_mad(df_supply, route_metrics, top_n=top_n, min_obs=5)

    routes_top_mad_filtered = filter_routes_with_mad_variance(
        df_supply, 
        routes_top_mad, 
        cpkm_col=cpkm_column, 
        min_obs=5
    )

    plot_cpkm_histograms_top_routes_mad(
        df_supply[df_supply['route'].isin(routes_top_mad_filtered)],
        top_routes = routes_top_mad_filtered,
        top_n=10,
        title_suffix=f'TOP ROUTES OVER MAD. MAD Outlier Analysis. Actual CPKM. SUPPLY: {supply_type}',
        cpkm_column=cpkm_column,
        hue_column=hue,
        bins=50
    )

    
    plot_cpkm_histograms_top_routes_mad(
        df_supply[df_supply['route'].isin(routes_top_mad_filtered)],
        top_routes = routes_top_mad_filtered,
        top_n=10,
        title_suffix=f'TOP ROUTES OVER MAD. MAD Outlier Analysis. Actual CPKM. SUPPLY: {supply_type}. Split by SET.',
        cpkm_column=cpkm_column,
        hue_column='is_set',
        bins=50
    )

for i in ['1.AZNG','2.RLB','3.3P']:
    supply_type_visualisation(df_routes, supply_type=i, cpkm_column='cpkm',mad_cutoff=3,top_n=8)



