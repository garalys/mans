import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import re
import time
import os
import io
import logging
import sys
 
# Configure logging for CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
 
 
# Configure logging
logger = logging.getLogger(__name__)
 
# Initialize S3 client
s3_client = boto3.client('s3')
 
 
############################ TO BE UPDATED ################################
# Constants
# Constants can be stored in environment variables
#VOLUME TO CARRIERS
COUNTRY_COEFFICIENTS_VOLUME = {
    'DE_SLOPE': float(os.environ.get('DE_SLOPE')),
    'ES_SLOPE': float(os.environ.get('ES_SLOPE')),
    'FR_SLOPE': float(os.environ.get('FR_SLOPE')),
    'IT_SLOPE': float(os.environ.get('IT_SLOPE')),
    'UK_SLOPE': float(os.environ.get('UK_SLOPE')),
    'EU_SLOPE': float(os.environ.get('EU_SLOPE')),
    'DE_INTERCEPT': float(os.environ.get('DE_INTERCEPT')),
    'ES_INTERCEPT': float(os.environ.get('ES_INTERCEPT')),
    'FR_INTERCEPT': float(os.environ.get('FR_INTERCEPT')),
    'IT_INTERCEPT': float(os.environ.get('IT_INTERCEPT')),
    'UK_INTERCEPT': float(os.environ.get('UK_INTERCEPT')),
    'EU_INTERCEPT': float(os.environ.get('EU_INTERCEPT'))
}
 
#LCR COEFFICIENTS
COUNTRY_COEFFICIENTS_LCR = {
    'DE_SLOPE_LCR': float(os.environ.get('DE_SLOPE_LCR')),
    'ES_SLOPE_LCR': float(os.environ.get('ES_SLOPE_LCR')),
    'FR_SLOPE_LCR': float(os.environ.get('FR_SLOPE_LCR')),
    'IT_SLOPE_LCR': float(os.environ.get('IT_SLOPE_LCR')),
    'UK_SLOPE_LCR': float(os.environ.get('UK_SLOPE_LCR')),
    'EU_SLOPE_LCR': float(os.environ.get('EU_SLOPE_LCR')),
    'DE_POLY_LCR': float(os.environ.get('DE_POLY_LCR')),
    'ES_POLY_LCR': float(os.environ.get('ES_POLY_LCR')),
    'FR_POLY_LCR': float(os.environ.get('FR_POLY_LCR')),
    'IT_POLY_LCR': float(os.environ.get('IT_POLY_LCR')),
    'UK_POLY_LCR': float(os.environ.get('UK_POLY_LCR')),
    'EU_POLY_LCR': float(os.environ.get('EU_POLY_LCR')),
    'DE_INTERCEPT_LCR': float(os.environ.get('DE_INTERCEPT_LCR')),
    'ES_INTERCEPT_LCR': float(os.environ.get('ES_INTERCEPT_LCR')),
    'FR_INTERCEPT_LCR': float(os.environ.get('FR_INTERCEPT_LCR')),
    'IT_INTERCEPT_LCR': float(os.environ.get('IT_INTERCEPT_LCR')),
    'UK_INTERCEPT_LCR': float(os.environ.get('UK_INTERCEPT_LCR')),
    'EU_INTERCEPT_LCR': float(os.environ.get('EU_INTERCEPT_LCR'))
}
### update as needed. need to create .env variables...
def get_tech_savings_rate(year, week):
    """
    Calculate tech savings rate based on year and week using environment variables.
   
    Dynamically reads all environment variables matching the pattern:
    TECH_SAVINGS_RATE_{YEAR}_W{START}_W{END}
   
    No code changes needed when adding new time periods - just add new env variables!
   
    Environment Variables Format:
        TECH_SAVINGS_RATE_{YEAR}_W{START}_W{END}={RATE}
       
    Examples:
        TECH_SAVINGS_RATE_2025_W27_W46=-0.0571
        TECH_SAVINGS_RATE_2025_W47_W52=-0.0257
        TECH_SAVINGS_RATE_2026_W1_W18=-0.0257
   
    Args:
        year (int): The year to check (e.g., 2025, 2026)
        week (int): The week number to check (1-52)
       
    Returns:
        float: The tech savings rate as a decimal (negative = savings)
               Returns 0 if no matching configuration is found
    """
    # Pattern to match: TECH_SAVINGS_RATE_2025_W27_W46
    pattern = re.compile(r'^TECH_SAVINGS_RATE_(\d{4})_W(\d+)_W(\d+)$')
   
    # Scan all environment variables
    for env_var, value in os.environ.items():
        match = pattern.match(env_var)
        if match:
            env_year = int(match.group(1))
            week_start = int(match.group(2))
            week_end = int(match.group(3))
           
            # Check if this config matches the requested year and week
            if env_year == year and week_start <= week <= week_end:
                try:
                    rate = float(value)
                    return rate
                except ValueError:
                    logger.warning(
                        f"Invalid value for {env_var}: '{value}'. Skipping."
                    )
                    continue
   
    # No matching configuration found
    logger.debug(f"No tech savings rate configured for year {year}, week {week}. Returning 0")
    return 0
 
def load_and_process_data(bucket, key, carrier_scaling_bucket, carrier_scaling_key, op2_bucket, op2_key):
    """
    Load and preprocess the input CSV file from S3
    """
    try:
        # Read file from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
       
        # Validate required columns
        required_columns = ['report_day', 'report_year', 'report_month', 'report_week',
                          'orig_country', 'business']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
           
        # Convert report_day to datetime immediately after loading
        try:
            df['report_day'] = pd.to_datetime(df['report_day'])
        except Exception as e:
            logger.error(f"Error converting report_day to datetime: {str(e)}")
            logger.info(f"Sample of report_day values: {df['report_day'].head()}")
            raise
           
        response_carrier = s3_client.get_object(Bucket=carrier_scaling_bucket, Key=carrier_scaling_key)
        df_carrier = pd.read_csv(io.BytesIO(response_carrier['Body'].read()))
 
        op2 = s3_client.get_object(Bucket=op2_bucket, Key=op2_key)
        df_op2 = pd.read_csv(io.BytesIO(op2['Body'].read()))
           
        return df, df_carrier, df_op2
   
    except Exception as e:
        logger.error(f"Error loading data from S3: {str(e)}")
        raise
 
 
def export_for_quicksight(bridge_df, destination_bucket, destination_key):
    """
    Exports the data to S3 in a format optimized for QuickSight visualization
    """
    try:
        # Convert DataFrame to CSV buffer
        csv_buffer = io.StringIO()
        bridge_df.to_csv(csv_buffer, index=False)
       
        # Upload to S3
        s3_client.put_object(
            Bucket=destination_bucket,
            Key=destination_key,
            Body=csv_buffer.getvalue()
        )
    except Exception as e:
        print(f"Error exporting to S3: {str(e)}")
        raise
###########################################################################
### fine
def calculate_market_rate_cost(base_data, compare_data, country='', return_total=False):
    """
    Calculate the market rate impact considering multiple factors
    """
    if country == 'EU':
        # For EU, first group by business if Total
        if 'Total' in base_data['business'].unique():
            # Group by dest_country, distance_band, is_set first
            base_grouped = base_data.groupby(['orig_country','dest_country', 'distance_band', 'is_set']).agg({
                'distance_for_cpkm': 'sum',
                'total_cost_usd': 'sum',
                'transporeon_contract_price_eur': 'first'
            }).reset_index()
           
            compare_grouped = compare_data.groupby(['orig_country','dest_country', 'distance_band', 'is_set']).agg({
                'distance_for_cpkm': 'sum',
                'transporeon_contract_price_eur': 'first'
            }).reset_index()
        else:
            # For EU by business, keep the original grouping
            base_grouped = base_data.groupby(['orig_country','dest_country', 'distance_band', 'is_set', 'business']).agg({
                'distance_for_cpkm': 'sum',
                'total_cost_usd': 'sum',
                'transporeon_contract_price_eur': 'mean'
            }).reset_index()
           
            compare_grouped = compare_data.groupby(['orig_country','dest_country', 'distance_band', 'is_set', 'business']).agg({
                'distance_for_cpkm': 'sum',
                'transporeon_contract_price_eur': 'mean'
            }).reset_index()
       
        # Calculate base rates and merge
        base_grouped['base_rate'] = np.where(
            base_grouped['distance_for_cpkm'] > 0,
            base_grouped['total_cost_usd'] / base_grouped['distance_for_cpkm'],
            0
        )
       
        merge_cols = ['orig_country','dest_country', 'distance_band', 'is_set']
        if 'business' in base_grouped.columns:
            merge_cols.append('business')
        merged = pd.merge(
            compare_grouped,
            base_grouped[merge_cols + ['base_rate', 'transporeon_contract_price_eur']],
            on=merge_cols,
            how='left'
        )
        # Calculate price evolution and impact
        merged['price_evolution'] = (merged['transporeon_contract_price_eur_x'] /
                                   merged['transporeon_contract_price_eur_y']) - 1
       
        merged['market_impact'] = (merged['distance_for_cpkm'] *
                                 merged['base_rate'] *
                                 merged['price_evolution'])
        total_distance = merged['distance_for_cpkm'].sum()
        total_market_impact = merged['market_impact'].sum()
       
        if return_total:
            return total_market_impact
        return total_market_impact / total_distance if total_distance > 0 else 0
       
    else:
        # For country-level calculations
        if 'Total' in base_data['business'].unique():
            # For Total, aggregate all businesses first
            base_data = base_data.groupby(['dest_country', 'distance_band', 'is_set']).agg({
                'distance_for_cpkm': 'sum',
                'total_cost_usd': 'sum',
                'transporeon_contract_price_eur': 'first'
            }).reset_index()
           
            compare_data = compare_data.groupby(['dest_country', 'distance_band', 'is_set']).agg({
                'distance_for_cpkm': 'sum',
                'transporeon_contract_price_eur': 'first'
            }).reset_index()
       
        return calculate_country_market_rate_impact(base_data, compare_data, return_total=False)
### fine
def calculate_country_market_rate_impact(base_data, compare_data, return_total=False):
    """Helper function for country-level market rate calculations"""
    # Group data by key dimensions
    base_grouped = base_data.groupby(['dest_country', 'distance_band', 'is_set']).agg({
        'distance_for_cpkm': 'sum',
        'total_cost_usd': 'sum',
        'transporeon_contract_price_eur': 'first'
    }).reset_index()
   
    compare_grouped = compare_data.groupby(['dest_country', 'distance_band', 'is_set']).agg({
        'distance_for_cpkm': 'sum',
        'transporeon_contract_price_eur': 'first'
    }).reset_index()
   
    # Calculate base rates and merge
    base_grouped['base_rate'] = np.where(
        base_grouped['distance_for_cpkm'] > 0,
        base_grouped['total_cost_usd'] / base_grouped['distance_for_cpkm'],
        0
    )
   
    merged = pd.merge(
        compare_grouped,
        base_grouped[['dest_country', 'distance_band', 'is_set', 'base_rate', 'transporeon_contract_price_eur']],
        on=['dest_country', 'distance_band', 'is_set'],
        how='left'
    )
   
    # Calculate price evolution and impact
    merged['price_evolution'] = (merged['transporeon_contract_price_eur_x'] /
                               merged['transporeon_contract_price_eur_y']) - 1
   
    merged['market_impact'] = (merged['distance_for_cpkm'] *
                             merged['base_rate'] *
                             merged['price_evolution'])
   
    total_distance = merged['distance_for_cpkm'].sum()
    total_market_impact = merged['market_impact'].sum()
   
    if return_total:
        return total_market_impact
    return total_market_impact / total_distance if total_distance > 0 else 0
### fine
### fine
def get_mtd_date_range(df, year, month):
    """
    Gets the available date range for a given month-to-date comparison
    """
    try:
        # Verify column exists
        if 'report_day' not in df.columns:
            logger.error("Column 'report_day' not found in DataFrame")
            logger.info(f"Available columns: {df.columns.tolist()}")
            raise KeyError("Column 'report_day' is required but not found")
           
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['report_day']):
            logger.info("Converting report_day to datetime")
            df['report_day'] = pd.to_datetime(df['report_day'])
       
        # Get data for specific month
        month_data = df[
            (df['report_year'] == year) &
            (df['report_month'] == f'M{month:02d}')
        ]
       
        if month_data.empty:
            logger.warning(f"No data found for year {year}, month {month}")
            return None, None
       
        max_date = month_data['report_day'].max()
        start_date = max_date.replace(day=1)
       
        return start_date, max_date
       
    except Exception as e:
        logger.error(f"Error in get_mtd_date_range: {str(e)}")
        logger.error(f"Year: {year}, Month: {month}")
        raise
 
###fine
def extract_year(report_year):
    """Extract numeric year from R20XX format"""
    return int(report_year.replace('R', ''))
###fine
def create_bridge_structure(df):
    """
    Creates a consistent bridge structure for all combinations
    """
    # Get unique values
    weeks = sorted(df['report_week'].unique())
    months = sorted(df['report_month'].unique())
    countries = sorted(df['orig_country'].unique())
    businesses = sorted(df['business'].unique())
    countries.append('EU')
    years = sorted(df['report_year'].unique())
 
    # Create base structure
    bridge_rows = []
    for country in countries:
        for business in businesses:
            # Create YoY combinations - ONLY CONSECUTIVE YEARS
            for i in range(len(years)-1):
                base_year = years[i]
                compare_year = years[i+1]  # Changed from nested loop
 
                # YoY weekly combinations
                for week in weeks:
                    base_row = create_yoy_row(base_year, compare_year, week, country, business)
                    bridge_rows.append(base_row)
 
                # MTD combinations
                for month in months:
                    base_row = create_mtd_row(base_year, compare_year, month, country, business)
                    bridge_rows.append(base_row)
 
            # WoW combinations (unchanged)
            for year in years:
                week_pairs = list(zip(weeks[:-1], weeks[1:]))
                for w1, w2 in week_pairs:
                    base_row = create_wow_row(year, w1, w2, country, business)
                    bridge_rows.append(base_row)
 
    return pd.DataFrame(bridge_rows)
 
### fine
def create_yoy_row(base_year, compare_year, week, country, business):
    """Creates a YoY bridge row"""
    return {
        'report_year': compare_year,
        'report_week': week,
        'orig_country': country,
        'business': business,
        'bridge_type': 'YoY',
        'bridging_value': f'{base_year}_to_{compare_year}',
        **create_base_metrics_dict()
    }
###fine
def create_mtd_row(base_year, compare_year, month, country, business):
    """Creates a MTD bridge row"""
    return {
        'report_year': compare_year,
        'report_month': month,
        'orig_country': country,
        'business': business,
        'bridge_type': 'MTD',
        'bridging_value': f'{base_year}_{month}_to_{compare_year}_{month}',
        **create_base_metrics_dict(),
        'm1_distance_km': None,
        'm1_costs_usd': None,
        'm1_loads': None,
        'm1_carriers': None,
        'm1_cpkm': None,
        'm2_distance_km': None,
        'm2_costs_usd': None,
        'm2_loads': None,
        'm2_carriers': None,
        'm2_cpkm': None
    }
###fine
def create_wow_row(year, w1, w2, country, business):
    """Creates a WoW bridge row"""
    return {
        'report_year': year,
        'report_week': w2,
        'orig_country': country,
        'business': business,
        'bridge_type': 'WoW',
        'bridging_value': f'{year}_{w1}_to_{w2}',
        **create_base_metrics_dict(),
        'w1_distance_km': None,
        'w1_costs_usd': None,
        'w1_loads': None,
        'w1_carriers': None,
        'w1_cpkm': None,
        'w2_distance_km': None,
        'w2_costs_usd': None,
        'w2_loads': None,
        'w2_carriers': None,
        'w2_cpkm': None
    }
 
### TBU with Premium
def create_base_metrics_dict():
    """Creates dictionary with base metrics"""
    return {
        'base_cpkm': None,
        'mix_impact': None,
        'normalised_cpkm': None,
        'supply_rates':None,
        'carrier_and_demand_impact':None,
        'carrier_impact': None,
        'demand_impact': None,
        'premium_impact':None,
        'market_rate_impact':None,
        'tech_impact': None,
        'set_impact': None,
        'compare_cpkm': None
    }
 
 
def calculate_set_impact(base_data, compare_data):
    """
    Calculate normalized cost using base year rates on comparison year volume
    Only includes SET costs where is_set=True
    """
    # Filter for SET loads only (is_set=True)
    base_set = base_data[base_data['is_set'] == True].copy()
    compare_set = compare_data[compare_data['is_set'] == True].copy()
   
    # Group data by distance band and equipment type
    base_grouped = base_set.groupby(['dest_country', 'distance_band', 'is_set']).agg({
        'distance_for_cpkm': 'sum',
        'total_cost_usd': 'sum'  # Changed from total_cost_usd to set_cost
    }).reset_index()
   
    compare_grouped = compare_set.groupby(['dest_country', 'distance_band', 'is_set']).agg({
        'distance_for_cpkm': 'sum',
        'total_cost_usd': 'sum'
    }).reset_index()
   
    # Calculate base rates using set_cost
    base_grouped['cpkm_calc'] = np.where(
        base_grouped['distance_for_cpkm'] > 0,
        base_grouped['total_cost_usd'] / base_grouped['distance_for_cpkm'],
        0
    )
    compare_grouped['cpkm_calc_compare'] = np.where(
        compare_grouped['distance_for_cpkm'] > 0,
        compare_grouped['total_cost_usd'] / compare_grouped['distance_for_cpkm'],
        0
    )
   
    # Merge and calculate normalized cost
    merged = pd.merge(
        compare_grouped,
        base_grouped[['dest_country', 'distance_band', 'is_set', 'cpkm_calc']],
        on=['dest_country', 'distance_band', 'is_set'],
        how='left'
    )
   
    merged['set_impact'] = merged['distance_for_cpkm'] * (merged['cpkm_calc_compare'] - merged['cpkm_calc'])
   
    return merged['set_impact'].sum()
 
### fine
def calculate_normalized_cost(base_data, compare_data):
    """
    Calculate normalized cost using base year rates on comparison year volume
    """
    # Group data by distance band and equipment type
    base_grouped = base_data.groupby(['dest_country','distance_band', 'is_set']).agg({
        'distance_for_cpkm': 'sum',
        'total_cost_usd': 'sum'
    }).reset_index()
   
    compare_grouped = compare_data.groupby(['dest_country','distance_band', 'is_set']).agg({
        'distance_for_cpkm': 'sum'
    }).reset_index()
   
    # Calculate base rates
    base_grouped['rate'] = np.where(
        base_grouped['distance_for_cpkm'] > 0,
        base_grouped['total_cost_usd'] / base_grouped['distance_for_cpkm'],
        0
    )
   
    # Merge and calculate normalized cost
    merged = pd.merge(
        compare_grouped,
        base_grouped[['dest_country','distance_band', 'is_set', 'rate']],
        on=['dest_country','distance_band', 'is_set'],
        how='left'
    )
   
    merged['normalized_cost'] = merged['distance_for_cpkm'] * merged['rate']
   
    return merged['normalized_cost'].sum()
### fine
def calculate_bridge_metrics(df, bridge_df, base_year, compare_year, df_carrier):
    """
    Calculates bridge metrics for specified base and comparison years
    """
    base_year_num = extract_year(base_year)
    compare_year_num = extract_year(compare_year)
 
    bridging_value = f'{base_year}_to_{compare_year}'
   
    i = 0
   
    # Process each week/country/business combination
    for idx, row in bridge_df[
        (bridge_df['bridge_type'] == 'YoY') &
        (bridge_df['bridging_value'] == bridging_value)
    ].iterrows():
       
        week = row['report_week']
        country = row['orig_country']
        business = row['business']
       
        # Get relevant data
        base_data = df[
            (df ['report_year'] == base_year) &
            (df['report_week'] == week) &
            (df['orig_country'] == country) &
            (df['business'] == business)
        ]
       
        compare_data = df[
            (df['report_year'] == compare_year) &
            (df['report_week'] == week) &
            (df['orig_country'] == country) &
            (df['business'] == business)
        ]
       
        # Calculate metrics
        metrics = calculate_detailed_bridge_metrics(
            base_data,
            compare_data,
            country,
            base_year_num,
            compare_year_num,
            df_carrier,
            report_week = week
        )
       
        # Update bridge_df
        for metric, value in metrics.items():
            bridge_df.loc[idx, metric] = value
### fine
def calculate_active_carriers(data, week, country, business=None):
    """
    Calculate active carriers count matching SQL logic exactly
    """
    if country == 'EU':
        # Calculate total loads by carrier
        carrier_total_loads = data.groupby('vehicle_carrier')['executed_loads'].sum()
       
        # For each carrier, divide its loads by its own total loads
        fractional_carriers = (data[data['executed_loads'] > 0]
            .groupby('vehicle_carrier')['executed_loads']
            .apply(lambda x: x / carrier_total_loads[x.name])
            .sum())
           
        return fractional_carriers
    else:
        # For country-level, use efficient numpy unique count
        mask = data['executed_loads'] > 0
        if business is not None:
            mask &= data['business'] == business
        return data.loc[mask, 'vehicle_carrier'].nunique()
### fine
def calculate_detailed_bridge_metrics(base_data, compare_data, country, base_year, compare_year, df_carrier, report_week=None, report_month=None):
    """
    Calculates detailed metrics for the bridge visualization
    """
    slope = COUNTRY_COEFFICIENTS_VOLUME.get(str(country)+str('_SLOPE'), COUNTRY_COEFFICIENTS_VOLUME['EU_SLOPE'])
    intercept = COUNTRY_COEFFICIENTS_VOLUME.get(str(country)+str('_INTERCEPT'), COUNTRY_COEFFICIENTS_VOLUME['EU_INTERCEPT'])
 
    slope_lcr = COUNTRY_COEFFICIENTS_LCR.get(str(country)+str('_SLOPE_LCR'), COUNTRY_COEFFICIENTS_LCR['EU_SLOPE_LCR'])
    poly_lcr = COUNTRY_COEFFICIENTS_LCR.get(str(country)+str('_POLY_LCR'), COUNTRY_COEFFICIENTS_LCR['EU_POLY_LCR'])
    intercept_lcr = COUNTRY_COEFFICIENTS_LCR.get(str(country)+str('_INTERCEPT_LCR'), COUNTRY_COEFFICIENTS_LCR['EU_INTERCEPT_LCR'])
   
    metrics = {}
   
    # Base metrics
    base_dist = base_data['distance_for_cpkm'].sum()
    base_cost = base_data['total_cost_usd'].sum()
    base_loads = base_data['executed_loads'].sum()
   
    # Calculate active carriers using new logic
    base_carriers = calculate_active_carriers(
        base_data,
        base_data['report_week'].iloc[0] if not base_data.empty else None,
        country,
        base_data['business'].iloc[0] if not base_data.empty else None
    )
   
    # Compare metrics
    compare_dist = compare_data['distance_for_cpkm'].sum()
    compare_cost = compare_data['total_cost_usd'].sum()
    compare_loads = compare_data['executed_loads'].sum()
   
    # Calculate active carriers for comparison period
    compare_carriers = calculate_active_carriers(
        compare_data,
        compare_data['report_week'].iloc[0] if not compare_data.empty else None,
        country,
        compare_data['business'].iloc[0] if not compare_data.empty else None
    )
   
    # Calculate CPKMs
    base_cpkm = base_cost / base_dist if base_dist > 0 else None
    compare_cpkm = compare_cost / compare_dist if compare_dist > 0 else None
   
    metrics.update({
        'base_cpkm': base_cpkm,
        'compare_cpkm': compare_cpkm
    })
   
    # Calculate bridge components if we have valid data
    if base_dist > 0 and compare_dist > 0:
 
        ### UPDATE NEEDED WHEN WE HAVE ALL FIGURES...
        # Mix Impact
        normalized_cost = calculate_normalized_cost(base_data, compare_data)
        mix_impact = (normalized_cost / compare_dist) - base_cpkm
        metrics['mix_impact'] = mix_impact
        metrics['normalised_cpkm'] = (normalized_cost / compare_dist)
 
        if base_data['distance_for_cpkm'].sum() > 0 and compare_data['distance_for_cpkm'].sum() > 0:
            # Calculate market rate impact
            market_rate_impact = calculate_market_rate_cost(base_data, compare_data)
            metrics['market_rate_impact'] = market_rate_impact
        else:
            metrics['market_rate_impact'] = 0
           
        if compare_carriers > 0 and base_carriers > 0:
            # Determine the period identifier for lookup
            if report_week:
                period = report_week  # e.g., 'W01'
            elif report_month:
                period = report_month  # e.g., 'M01'
            else:
                # Fallback: try to extract from data
                if not compare_data.empty and 'report_week' in compare_data.columns:
                    period = compare_data['report_week'].iloc[0]
                elif not compare_data.empty and 'report_month' in compare_data.columns:
                    period = compare_data['report_month'].iloc[0]
                else:
                    period = None
           
            # Lookup percentage from df_carrier
            percentage = 0.0  # Default value
            if period is not None:
                year_str = f'R{compare_year}'
                lookup_mask = (
                    (df_carrier['year'] == year_str) &
                    (df_carrier['period'] == period) &
                    (df_carrier['country'] == country)
                )
                matching_rows = df_carrier[lookup_mask]
               
                if not matching_rows.empty:
                    percentage = matching_rows['percentage'].iloc[0]
           
            # Calculate LCR values
            base_lcr = base_loads / base_carriers
            compare_lcr = compare_loads / compare_carriers
            base_expected_executing_carriers = (base_loads * slope + intercept) * (1 + percentage)
            compare_expected_executing_carriers = (compare_loads * slope + intercept) * (1 + percentage)
            base_expected_lcr = base_loads / base_expected_executing_carriers
            compare_expected_lcr = compare_loads / compare_expected_executing_carriers
 
            carrier_impact_score = compare_lcr - base_lcr + base_expected_lcr - compare_expected_lcr
            demand_impact_score = compare_expected_lcr - base_expected_lcr
 
            carrier_impact = ((intercept_lcr + slope_lcr*(carrier_impact_score + base_lcr) + poly_lcr*((carrier_impact_score + base_lcr)**2)) / (intercept_lcr + slope_lcr*base_lcr + poly_lcr*(base_lcr**2))-1)*base_cpkm
 
            demand_impact = ((intercept_lcr + slope_lcr*(demand_impact_score + base_lcr) + poly_lcr*((demand_impact_score + base_lcr)**2)) / (intercept_lcr + slope_lcr*base_lcr + poly_lcr*(base_lcr**2))-1)*base_cpkm
 
            metrics['carrier_impact'] = carrier_impact
            metrics['demand_impact'] = demand_impact
        else:
            metrics.update({
                'carrier_impact': 0.0,
                'demand_impact': 0.0
            })
       
        # SET Impact
        if base_dist > 0 and compare_dist > 0:
            set_impact = calculate_set_impact(base_data, compare_data)
            metrics['set_impact'] = set_impact / compare_dist
        else:
            metrics['set_impact'] = None
 
        # Tech Impact
        week_num = int(base_data['report_week'].iloc[0].replace('W', '')) if not base_data.empty else 0
        tech_rate = get_tech_savings_rate(compare_year, week_num)
        tech_impact = compare_cpkm * tech_rate if compare_cpkm is not None else 0.0
        metrics['tech_impact'] = tech_impact
 
    else:
        metrics.update({
            'mix_impact': None,
            'normalised_cpkm': None,
            'supply_rates':None,
            'carrier_and_demand_impact':None,
            'carrier_impact': None,
            'demand_impact': None,
            'premium_impact':None,
            'market_rate_impact':None,
            'set_impact': None,
            'tech_impact': None
        })
   
    # Store raw values for flexible comparison
    metrics.update({
        'base_distance_km': base_dist,
        'base_costs_usd': base_cost if base_cost > 0 else 0,
        'base_loads': base_loads,
        'base_carriers': base_carriers,
       
        'compare_distance_km': compare_dist,
        'compare_costs_usd': compare_cost if compare_cost > 0 else 0,
        'compare_loads': compare_loads,
        'compare_carriers': compare_carriers
    })
   
    return metrics
 
def update_wow_metrics(bridge_df, idx, metrics, year):
    """Update WoW metrics in bridge_df"""
    # Update WoW specific columns
    metrics_mapping = {
        'base_distance_km': 'w1_distance_km',
        'base_costs_usd': 'w1_costs_usd',
        'base_loads': 'w1_loads',
        'base_carriers': 'w1_carriers',
        'base_cpkm': 'w1_cpkm'
    }
   
    # Update base metrics
    for old_key, new_key in metrics_mapping.items():
        if old_key in metrics:
            bridge_df.loc[idx, new_key] = metrics[old_key]
   
    # Update comparison metrics
    bridge_df.loc[idx, 'w2_distance_km'] = metrics.get('compare_distance_km')
    bridge_df.loc[idx, 'w2_costs_usd'] = metrics.get('compare_costs_usd')
    bridge_df.loc[idx, 'w2_loads'] = metrics.get('compare_loads')
    bridge_df.loc[idx, 'w2_carriers'] = metrics.get('compare_carriers')
    bridge_df.loc[idx, 'w2_cpkm'] = metrics.get('compare_cpkm')
 
   
    # Update bridge components
    bridge_components = [
        'base_cpkm', 'mix_impact', 'normalised_cpkm', 'supply_rates',
        'carrier_and_demand_impact', 'carrier_impact', 'demand_impact',
        'premium_impact', 'market_rate_impact', 'tech_impact', 'set_impact', 'compare_cpkm'
    ]
   
    for component in bridge_components:
        if component in metrics:
            bridge_df.loc[idx, component] = metrics[component]
 
def calculate_wow_bridge_metrics(df, bridge_df, df_carrier):
    """Optimized week-over-week calculations"""
    years = sorted(df['report_year'].unique())
   
    # Pre-calculate all required data
    wow_data = {}
    for year in years:
        year_data = df[df['report_year'] == year]
        # Group by relevant columns to avoid repeated filtering
        wow_data[year] = year_data.groupby(['report_week', 'orig_country', 'business']).apply(
            lambda x: {
                'data': x,
                'distance': x['distance_for_cpkm'].sum(),
                'costs': x['total_cost_usd'].sum(),
                'loads': x['executed_loads'].sum(),
                'carriers': calculate_active_carriers(x, x['report_week'].iloc[0],
                                                   x['orig_country'].iloc[0],
                                                   x['business'].iloc[0])
            }
        ).to_dict()
 
    # Process WoW rows in batches
    batch_size = 1000
    wow_rows = bridge_df[bridge_df['bridge_type'] == 'WoW']
   
    for start_idx in range(0, len(wow_rows), batch_size):
        batch = wow_rows.iloc[start_idx:start_idx + batch_size]
       
        for idx, row in batch.iterrows():
            year = row['report_year']
            w1, w2 = row['bridging_value'].split('_')[1], row['bridging_value'].split('_')[3]
           
            key = (w1, row['orig_country'], row['business'])
            compare_key = (w2, row['orig_country'], row['business'])
           
            if key in wow_data[year] and compare_key in wow_data[year]:
                base_data = wow_data[year][key]['data']
                compare_data = wow_data[year][compare_key]['data']
               
                metrics = calculate_detailed_bridge_metrics(
                    base_data, compare_data, row['orig_country'],
                    extract_year(year), extract_year(year), df_carrier, report_week=w2
                )
               
                # Update metrics using vectorized operations
                update_wow_metrics(bridge_df, idx, metrics, year)
 
### fine
 
def calculate_mtd_bridge_metrics(df, bridge_df, df_carrier):
    """
    Calculates month-to-date bridge metrics - only consecutive years
    """
    years = sorted(df['report_year'].unique())
   
    # Process only consecutive year pairs
    for i in range(len(years)-1):
        base_year = years[i]
        compare_year = years[i+1]  # Only next consecutive year
       
        mtd_rows = bridge_df[
            (bridge_df['bridge_type'] == 'MTD') &
            (bridge_df['bridging_value'].str.contains(f'{base_year}_.*_to_{compare_year}'))
        ]
       
        for idx, row in mtd_rows.iterrows():
            month = row['report_month']
            country = row['orig_country']
            business = row['business']
           
            # Get MTD date range
            start_date, end_date = get_mtd_date_range(df, compare_year, int(month.replace('M', '')))
            if start_date is None:
                continue
           
            # Get base year data
            base_data = df[
                (df['report_year'] == base_year) &
                (df['report_month'] == month) &
                (df['report_day'] >= start_date.replace(year=extract_year(base_year))) &
                (df['report_day'] <= end_date.replace(year=extract_year(base_year))) &
                (df['orig_country'] == country) &
                (df['business'] == business)
            ]
           
            # Get comparison year data
            compare_data = df[
                (df['report_year'] == compare_year) &
                (df['report_month'] == month) &
                (df['report_day'] >= start_date) &
                (df['report_day'] <= end_date) &
                (df['orig_country'] == country) &
                (df['business'] == business)
            ]
           
            # Skip if either dataset is empty
            if base_data.empty or compare_data.empty:
                continue
           
            # Calculate metrics
            metrics = calculate_detailed_bridge_metrics(
                base_data,
                compare_data,
                country,
                extract_year(base_year),
                extract_year(compare_year),
                df_carrier,
                report_month=month
            )
           
            # Update bridge_df with MTD specific columns
            metrics_mapping = {
                'base_distance_km': 'm1_distance_km',
                'base_costs_usd': 'm1_costs_usd',
                'base_loads': 'm1_loads',
                'base_carriers': 'm1_carriers',
                'base_cpkm': 'm1_cpkm'
            }
           
            for old_key, new_key in metrics_mapping.items():
                if old_key in metrics:
                    bridge_df.loc[idx, new_key] = metrics[old_key]
           
            # Update comparison month metrics
            bridge_df.loc[idx, 'm2_distance_km'] = metrics.get('compare_distance_km')
            bridge_df.loc[idx, 'm2_costs_usd'] = metrics.get('compare_costs_usd')
            bridge_df.loc[idx, 'm2_loads'] = metrics.get('compare_loads')
            bridge_df.loc[idx, 'm2_carriers'] = metrics.get('compare_carriers')
            bridge_df.loc[idx, 'm2_cpkm'] = metrics.get('compare_cpkm')
 
           
            # Update bridge components
            for metric, value in metrics.items():
                if metric not in metrics_mapping:  # Don't overwrite already mapped metrics
                    bridge_df.loc[idx, metric] = value
 
 
 
def create_bridge_structure_for_totals(df):
    """Creates bridge structure for total calculations - only consecutive years"""
    weeks = sorted(df['report_week'].unique())
    months = sorted(df['report_month'].unique())
    countries = sorted(df['orig_country'].unique())
    countries.append('EU')
    years = sorted(df['report_year'].unique())
 
    bridge_rows = []
    for country in countries:
        # Create rows only for 'Total' business
        # YoY combinations - ONLY CONSECUTIVE YEARS
        for i in range(len(years)-1):
            base_year = years[i]
            compare_year = years[i+1]  # Only next consecutive year
           
            for week in weeks:
                base_row = create_yoy_row(base_year, compare_year, week, country, 'Total')
                bridge_rows.append(base_row)
           
            for month in months:
                base_row = create_mtd_row(base_year, compare_year, month, country, 'Total')
                bridge_rows.append(base_row)
       
        # WoW combinations (unchanged)
        for year in years:
            week_pairs = list(zip(weeks[:-1], weeks[1:]))
            for w1, w2 in week_pairs:
                base_row = create_wow_row(year, w1, w2, country, 'Total')
                bridge_rows.append(base_row)
   
    return pd.DataFrame(bridge_rows)
 
 
### fine
 
def calculate_aggregated_total_bridge_metrics(df, bridge_df, df_carrier):
    """Optimized total metrics calculation with pre-aggregated data - only consecutive years"""
    # Pre-aggregate data by period type
    aggs = {}
    for period_type in ['YoY', 'WoW', 'MTD']:
        aggs[period_type] = {}
       
    # Process in batches
    batch_size = 1000
    for period_type in ['YoY', 'WoW', 'MTD']:
        rows = bridge_df[bridge_df['bridge_type'] == period_type]
       
        for start_idx in range(0, len(rows), batch_size):
            batch = rows.iloc[start_idx:start_idx + batch_size]
           
            for idx, row in batch.iterrows():
                country = row['orig_country']
               
                # Get data based on period type
                if period_type == 'YoY':
                    base_year, compare_year = row['bridging_value'].split('_to_')
                    time_period = row['report_week']
                    key = (base_year, compare_year, time_period)
                   
                    if key not in aggs[period_type]:
                        aggs[period_type][key] = {
                            'base': df[
                                (df['report_year'] == base_year) &
                                (df['report_week'] == time_period)
                            ],
                            'compare': df[
                                (df['report_year'] == compare_year) &
                                (df['report_week'] == time_period)
                            ]
                        }
                    base_data = aggs[period_type][key]['base']
                    compare_data = aggs[period_type][key]['compare']
                    compare_year = extract_year(compare_year)
                   
                elif period_type == 'WoW':
                    year = row['report_year']
                    w1, w2 = row['bridging_value'].split('_')[1], row['bridging_value'].split('_')[3]
                    time_period = w2
                    key = (year, w1, w2)
                   
                    if key not in aggs[period_type]:
                        aggs[period_type][key] = {
                            'base': df[
                                (df['report_year'] == year) &
                                (df['report_week'] == w1)
                            ],
                            'compare': df[
                                (df['report_year'] == year) &
                                (df['report_week'] == w2)
                            ]
                        }
                    base_data = aggs[period_type][key]['base']
                    compare_data = aggs[period_type][key]['compare']
                    compare_year = extract_year(year)
                   
                else:  # MTD
                    base_year, month, compare_year = re.match(r'(.+)_(.+)_to_(.+)_.+', row['bridging_value']).groups()
                    start_date, end_date = get_mtd_date_range(df, compare_year, int(month.replace('M', '')))
                   
                    if start_date is None:
                        continue
                       
                    key = (base_year, compare_year, month)
                    if key not in aggs[period_type]:
                        aggs[period_type][key] = {
                            'base': df[
                                (df['report_year'] == base_year) &
                                (df['report_month'] == month) &
                                (df['report_day'] >= start_date.replace(year=extract_year(base_year))) &
                                (df['report_day'] <= end_date.replace(year=extract_year(base_year)))
                            ],
                            'compare': df[
                                (df['report_year'] == compare_year) &
                                (df['report_month'] == month) &
                                (df['report_day'] >= start_date) &
                                (df['report_day'] <= end_date)
                            ]
                        }
                    base_data = aggs[period_type][key]['base']
                    compare_data = aggs[period_type][key]['compare']
                    time_period = month
                    compare_year = extract_year(compare_year)
               
                # Apply country filter if needed
                if country != 'EU':
                    base_data = base_data[base_data['orig_country'] == country]
                    compare_data = compare_data[compare_data['orig_country'] == country]
               
                # Skip if either dataset is empty
                if base_data.empty or compare_data.empty:
                    continue
               
                # Calculate metrics using pre-aggregated data
                metrics = calculate_total_metrics(base_data, compare_data, country, time_period)
               
                # Update bridge DataFrame
                update_bridge_with_total_metrics(
                    bridge_df,
                    metrics,
                    idx,
                    base_data,
                    compare_data,
                    period_type,
                    time_period,
                    compare_year,
                    df_carrier
                )
 
 
def calculate_total_metrics(base_data, compare_data, country, time_period):
    """Optimized total metrics calculation"""
    # Pre-aggregate common metrics
    aggs = {
        'base': base_data.groupby('business').agg({
            'distance_for_cpkm': 'sum',
            'total_cost_usd': 'sum',
            'executed_loads': 'sum'
        }),
        'compare': compare_data.groupby('business').agg({
            'distance_for_cpkm': 'sum',
            'total_cost_usd': 'sum',
            'executed_loads': 'sum'
        })
    }
   
    total_metrics = {
        'base_distance_km': aggs['base']['distance_for_cpkm'].sum(),
        'base_costs_usd': aggs['base']['total_cost_usd'].sum(),
        'base_loads': aggs['base']['executed_loads'].sum(),
        'distance_km': aggs['compare']['distance_for_cpkm'].sum(),
        'costs_usd': aggs['compare']['total_cost_usd'].sum(),
        'loads': aggs['compare']['executed_loads'].sum(),
        'market_rate_impact': 0
    }
   
    # Calculate carriers once
    total_metrics['base_carriers'] = calculate_active_carriers(base_data, time_period, country)
    total_metrics['carriers'] = calculate_active_carriers(compare_data, time_period, country)
   
    # Calculate market rate impact
    if country == 'EU':
        total_market_impact = 0
        total_compare_distance = 0
       
        for orig_country in ['DE', 'ES', 'FR', 'IT', 'UK']:
            base_country = base_data[base_data['orig_country'] == orig_country]
            compare_country = compare_data[compare_data['orig_country'] == orig_country]
           
            if not base_country.empty and not compare_country.empty:
                market_rate = calculate_market_rate_cost(base_country, compare_country, orig_country)
                logger.info(f'{orig_country}: market rate:{market_rate}')
                compare_distance = compare_country['distance_for_cpkm'].sum()
                total_market_impact += market_rate * compare_distance
                total_compare_distance += compare_distance
       
        if total_compare_distance > 0:
            total_metrics['market_rate_impact'] = total_market_impact / total_compare_distance
    else:
        if total_metrics['distance_km'] > 0:
            market_rate = calculate_market_rate_cost(base_data, compare_data, country)
            total_metrics['market_rate_impact'] = market_rate
   
    # Calculate fe final metrics
    if total_metrics['base_distance_km'] > 0:
        total_metrics['base_cpkm'] = total_metrics['base_costs_usd'] / total_metrics['base_distance_km']
    else:
        total_metrics['base_cpkm'] = None
       
    if total_metrics['distance_km'] > 0:
        total_metrics['cpkm'] = total_metrics['costs_usd'] / total_metrics['distance_km']
    else:
        total_metrics['cpkm'] = None
 
   
    
    return total_metrics
 
def update_bridge_with_total_metrics(bridge_df, metrics, idx, base_data, compare_data, period_type='YoY', time_period=None, compare_year=None, df_carrier=None):
    """Optimized update function for total metrics"""
    # Determine prefix mapping and period-specific values
    if period_type == 'YoY':
        base_year = extract_year(base_data['report_year'].iloc[0])
        compare_year = extract_year(compare_data['report_year'].iloc[0])
        prefix_map = {
            'base': f'y{base_year}_',
            'compare': f'y{compare_year}_'
        }
        if time_period is None:
            time_period = bridge_df.loc[idx, 'report_week']
           
    elif period_type == 'WoW':
        year = base_data['report_year'].iloc[0]
        prefix_map = {
            'base': 'w1_',
            'compare': 'w2_'
        }
        if time_period is None:
            time_period = bridge_df.loc[idx, 'bridging_value'].split('_')[3]
        compare_year = extract_year(year)  # Fix for WoW tech impact
       
    else:  # MTD
        prefix_map = {
            'base': 'm1_',
            'compare': 'm2_'
        }
   
    # Update metrics in batch
    update_dict = {
        'base_cpkm': metrics['base_cpkm'],
        f"{prefix_map['base']}distance_km": metrics['base_distance_km'],
        f"{prefix_map['base']}costs_usd": metrics['base_costs_usd'],
        f"{prefix_map['base']}loads": metrics['base_loads'],
        f"{prefix_map['base']}carriers": metrics['base_carriers'],
        f"{prefix_map['base']}cpkm": metrics['base_cpkm'],
        'compare_cpkm': metrics['cpkm'],
        f"{prefix_map['compare']}distance_km": metrics['distance_km'],
        f"{prefix_map['compare']}costs_usd": metrics['costs_usd'],
        f"{prefix_map['compare']}loads": metrics['loads'],
        f"{prefix_map['compare']}carriers": metrics['carriers'],
        f"{prefix_map['compare']}cpkm": metrics['cpkm'],
        'market_rate_impact': metrics['market_rate_impact']
    }
   
    bridge_df.loc[idx, list(update_dict.keys())] = list(update_dict.values())
   
    # Calculate bridge components if we have valid CPKMs
    if metrics['base_cpkm'] is not None and metrics['cpkm'] is not None:
        country = bridge_df.loc[idx, 'orig_country']
       
        # Calculate mix impact
        normalized_costs_total = 0
        businesses = sorted(base_data['business'].unique())
       
        for business in businesses:
            base_business = base_data[base_data['business'] == business]
            compare_business = compare_data[compare_data['business'] == business]
           
            if country != 'EU':
                base_business = base_business[base_business['orig_country'] == country]
                compare_business = compare_business[compare_business['orig_country'] == country]
           
            normalized_cost = calculate_normalized_cost(base_business, compare_business)
            normalized_costs_total += normalized_cost
       
        mix_impact = (normalized_costs_total / metrics['distance_km']) - metrics['base_cpkm']
        bridge_df.loc[idx, ['mix_impact', 'normalised_cpkm']] = [
            mix_impact,
            (normalized_costs_total / metrics['distance_km'])
        ]
 
        set_impact = calculate_set_impact(base_data, compare_data) / metrics['distance_km']
        bridge_df.loc[idx, 'set_impact'] = set_impact
       
        # Calculate carrier and demand impacts
        if metrics['carriers'] > 0 and metrics['base_carriers'] > 0:
            if time_period.startswith('W'):
                period = time_period
            elif time_period.startswith('M'):
                period = time_period
            else:
                period = None
 
            percentage = 0.0
            if period is not None:
                year_str = f'R{compare_year}'
                lookup_mask = (
                    (df_carrier['year'] == year_str) &
                    (df_carrier['period'] == period) &
                    (df_carrier['country'] == country)
                )
                matching_rows = df_carrier[lookup_mask]
 
                if not matching_rows.empty:
                    percentage = matching_rows['percentage'].iloc[0]
 
 
            slope = COUNTRY_COEFFICIENTS_VOLUME.get(f'{country}_SLOPE', COUNTRY_COEFFICIENTS_VOLUME['EU_SLOPE'])
            intercept = COUNTRY_COEFFICIENTS_VOLUME.get(f'{country}_INTERCEPT', COUNTRY_COEFFICIENTS_VOLUME['EU_INTERCEPT'])
            slope_lcr = COUNTRY_COEFFICIENTS_LCR.get(f'{country}_SLOPE_LCR', COUNTRY_COEFFICIENTS_LCR['EU_SLOPE_LCR'])
            poly_lcr = COUNTRY_COEFFICIENTS_LCR.get(f'{country}_POLY_LCR', COUNTRY_COEFFICIENTS_LCR['EU_POLY_LCR'])
            intercept_lcr = COUNTRY_COEFFICIENTS_LCR.get(f'{country}_INTERCEPT_LCR', COUNTRY_COEFFICIENTS_LCR['EU_INTERCEPT_LCR'])
 
            base_lcr = metrics['base_loads'] / metrics['base_carriers']
            compare_lcr = metrics['loads'] / metrics['carriers']
            base_expected_executing_carriers = (metrics['base_loads'] * slope + intercept) * (1 + percentage)
            compare_expected_executing_carriers = (metrics['loads'] * slope + intercept) * (1 + percentage)
            base_expected_lcr = metrics['base_loads'] / base_expected_executing_carriers
            compare_expected_lcr = metrics['loads'] / compare_expected_executing_carriers
 
            carrier_impact_score = compare_lcr - base_lcr + base_expected_lcr - compare_expected_lcr
            demand_impact_score = compare_expected_lcr - base_expected_lcr
 
            carrier_impact = ((intercept_lcr + slope_lcr*(carrier_impact_score + base_lcr) + poly_lcr*((carrier_impact_score + base_lcr)**2)) / (intercept_lcr + slope_lcr*base_lcr + poly_lcr*(base_lcr**2))-1)*metrics['base_cpkm']
 
            demand_impact = ((intercept_lcr + slope_lcr*(demand_impact_score + base_lcr) + poly_lcr*((demand_impact_score + base_lcr)**2)) / (intercept_lcr + slope_lcr*base_lcr + poly_lcr*(base_lcr**2))-1)*metrics['base_cpkm']
 
            impacts = {
                'carrier_impact': carrier_impact,
                'demand_impact': demand_impact
            }
            bridge_df.loc[idx, list(impacts.keys())] = list(impacts.values())
       
        # Calculate tech impact
        if period_type in ['YoY', 'WoW'] and time_period and compare_year:
            try:
                week_num = int(time_period.replace('W', ''))
                tech_rate = get_tech_savings_rate(compare_year, week_num)
                bridge_df.loc[idx, 'tech_impact'] = metrics['cpkm'] * tech_rate
            except (ValueError, TypeError):
                bridge_df.loc[idx, 'tech_impact'] = 0
 
### fine
def adjust_carrier_demand_impacts(bridge_df):
    """
    Adjusts carrier and demand impacts and calculates impacts in millions
    """
    # First calculate expected CPKM and discrepancy
    bridge_df['expected_cpkm'] = bridge_df.apply(lambda row:
        row['normalised_cpkm'] +
        (row['carrier_impact'] if pd.notnull(row['carrier_impact']) else 0) +
        (row['demand_impact'] if pd.notnull(row['demand_impact']) else 0) +
        (row['premium_impact'] if pd.notnull(row['premium_impact']) else 0) +
        (row['market_rate_impact'] if pd.notnull(row['market_rate_impact']) else 0) +
        (row['set_impact'] if pd.notnull(row['set_impact']) else 0) +
        (row['tech_impact'] if pd.notnull(row['tech_impact']) else 0)
        if pd.notnull(row['normalised_cpkm']) else None,
        axis=1
    )
 
    bridge_df['discrepancy'] = bridge_df.apply(
        lambda row: row['compare_cpkm'] - row['expected_cpkm']
        if pd.notnull(row['expected_cpkm']) and pd.notnull(row['compare_cpkm'])
        else None,
        axis=1
    )
 
    # Adjust impacts and calculate carrier_and_demand_impact
    for idx, row in bridge_df.iterrows():
        if pd.notnull(row['discrepancy']) and pd.notnull(row['carrier_impact']) and pd.notnull(row['demand_impact']):
            total_impact = abs(row['carrier_impact']) + abs(row['demand_impact'])
            if total_impact > 0:
                carrier_weight = abs(row['carrier_impact']) / total_impact
                demand_weight = abs(row['demand_impact']) / total_impact
 
                # Update carrier and demand impacts
                new_carrier = row['carrier_impact'] + (row['discrepancy'] * carrier_weight)
                new_demand = row['demand_impact'] + (row['discrepancy'] * demand_weight)
 
                bridge_df.at[idx, 'carrier_impact'] = new_carrier
                bridge_df.at[idx, 'demand_impact'] = new_demand
                bridge_df.at[idx, 'carrier_and_demand_impact'] = new_carrier + new_demand
 
        # Find appropriate distance column - FIXED VERSION
        distance = None
 
        # For WoW, use w2_distance_km
        if pd.notnull(row.get('w2_distance_km')):
            distance = row['w2_distance_km']
        # For MTD, use m2_distance_km
        elif pd.notnull(row.get('m2_distance_km')):
            distance = row['m2_distance_km']
        # For YoY, extract the compare year from bridging_value and use that year's distance
        elif row['bridge_type'] == 'YoY' and pd.notnull(row.get('bridging_value')):
            # Extract compare year from bridging_value (format: R2024_to_R2025)
            try:
                compare_year_str = row['bridging_value'].split('_to_')[1]  # Get 'R2025'
                compare_year_num = extract_year(compare_year_str)
                distance_col = f'y{compare_year_num}_distance_km'
                if distance_col in row.index and pd.notnull(row[distance_col]):
                    distance = row[distance_col]
            except (IndexError, ValueError):
                # Fallback to old logic if parsing fails
                if pd.notnull(row.get('compare_distance_km')):
                    distance = row['compare_distance_km']
        else:
            # Fallback for any other cases
            if pd.notnull(row.get('compare_distance_km')):
                distance = row['compare_distance_km']
 
        # Calculate impacts in millions
        if pd.notnull(distance):
            impacts = ['mix_impact', 'carrier_impact', 'demand_impact',
                      'carrier_and_demand_impact', 'premium_impact',
                      'market_rate_impact', 'set_impact','tech_impact']
 
            for impact in impacts:
                if pd.notnull(bridge_df.at[idx, impact]):
                    bridge_df.at[idx, f'{impact}_mm'] = (bridge_df.at[idx, impact] * distance) / 1_000_000
 
    # Drop temporary columns
    bridge_df = bridge_df.drop(['expected_cpkm', 'discrepancy'], axis=1)
 
    return bridge_df
### fine
 
def calculate_eu_metrics(base_data, compare_data, time_period, business):
    """
    Calculate EU metrics using optimized data handling
    """
    # Pre-filter business data once
    base_business = base_data[base_data['business'] == business]
    compare_business = compare_data[compare_data['business'] == business]
   
    # Pre-calculate aggregations
    eu_base = base_business.agg({
        'distance_for_cpkm': 'sum',
        'total_cost_usd': 'sum',
        'executed_loads': 'sum'
    })
   
    eu_compare = compare_business.agg({
        'distance_for_cpkm': 'sum',
        'total_cost_usd': 'sum',
        'executed_loads': 'sum'
    })
   
    # Calculate metrics
    metrics = {
        'distance_km': eu_compare['distance_for_cpkm'],
        'costs_usd': eu_compare['total_cost_usd'],
        'loads': eu_compare['executed_loads'],
        'base_distance_km': eu_base['distance_for_cpkm'],
        'base_costs_usd': eu_base['total_cost_usd'],
        'base_loads': eu_base['executed_loads']
    }
   
    # Calculate CPKMs
    metrics['base_cpkm'] = metrics['base_costs_usd'] / metrics['base_distance_km'] if metrics['base_distance_km'] > 0 else None
    metrics['cpkm'] = metrics['costs_usd'] / metrics['distance_km'] if metrics['distance_km'] > 0 else None
   
    # Calculate carriers once and store
    metrics['base_carriers'] = calculate_active_carriers(base_business, time_period, 'EU')
    metrics['carriers'] = calculate_active_carriers(compare_business, time_period, 'EU')
   
    return metrics
 
def update_bridge_with_eu_metrics(bridge_df, metrics, idx, base_data, compare_data, period_type='YoY', time_period=None, compare_year=None, df_carrier=None):
    """
    Update bridge DataFrame with EU metrics using vectorized operations
    """
    # Add validation at the start
    if base_data.empty or compare_data.empty:
        return
    # Determine prefix mapping
    if period_type == 'YoY':
        base_year = extract_year(base_data['report_year'].iloc[0])
        compare_year = extract_year(compare_data['report_year'].iloc[0])
        prefix_map = {
            'base': f'y{base_year}_',
            'compare': f'y{compare_year}_'
        }
    elif period_type == 'WoW':
        prefix_map = {
            'base': 'w1_',
            'compare': 'w2_'
        }
    else:  # MTD
        prefix_map = {
            'base': 'm1_',
            'compare': 'm2_'
        }
   
    # Create metrics dictionary for vectorized update
    update_dict = {
        'base_cpkm': metrics['base_cpkm'],
        f"{prefix_map['base']}distance_km": metrics['base_distance_km'],
        f"{prefix_map['base']}costs_usd": metrics['base_costs_usd'],
        f"{prefix_map['base']}loads": metrics['base_loads'],
        f"{prefix_map['base']}carriers": metrics['base_carriers'],
        f"{prefix_map['base']}cpkm": metrics['base_cpkm'],
        'compare_cpkm': metrics['cpkm'],
        f"{prefix_map['compare']}distance_km": metrics['distance_km'],
        f"{prefix_map['compare']}costs_usd": metrics['costs_usd'],
        f"{prefix_map['compare']}loads": metrics['loads'],
        f"{prefix_map['compare']}carriers": metrics['carriers'],
        f"{prefix_map['compare']}cpkm": metrics['cpkm']
    }
   
    # Update metrics in one operation
    bridge_df.loc[idx, list(update_dict.keys())] = list(update_dict.values())
   
    # Calculate bridge components if we have valid CPKMs
    if metrics['base_cpkm'] is not None and metrics['cpkm'] is not None:
        business = bridge_df.loc[idx, 'business']
       
        # Calculate mix impact
        normalized_costs_total = 0
        business_data = {
            'base': base_data[base_data['business'] == business],
            'compare': compare_data[compare_data['business'] == business]
        }
       
        for country in ['DE', 'ES', 'FR', 'IT', 'UK']:
            country_data = {
                'base': business_data['base'][business_data['base']['orig_country'] == country],
                'compare': business_data['compare'][business_data['compare']['orig_country'] == country]
            }
           
            if not country_data['base'].empty and not country_data['compare'].empty:
                normalized_cost = calculate_normalized_cost(
                    country_data['base'].groupby(['dest_country', 'distance_band', 'is_set']).agg({
                        'distance_for_cpkm': 'sum',
                        'total_cost_usd': 'sum'
                    }).reset_index(),
                    country_data['compare'].groupby(['dest_country', 'distance_band', 'is_set']).agg({
                        'distance_for_cpkm': 'sum'
                    }).reset_index()
                )
                normalized_costs_total += normalized_cost
       
        # Calculate and update impacts
        mix_impact = (normalized_costs_total / metrics['distance_km']) - metrics['base_cpkm']
        market_rate_impact = calculate_market_rate_cost(base_data, compare_data, 'EU')
       
        impact_updates = {
            'mix_impact': mix_impact,
            'normalised_cpkm': (normalized_costs_total / metrics['distance_km']),
            'market_rate_impact': market_rate_impact
        }
 
        impact_updates['set_impact'] = calculate_set_impact(base_data, compare_data) / metrics['distance_km']
       
        # Calculate carrier and demand impacts if we have valid carriers
        if metrics['carriers'] > 0 and metrics['base_carriers'] > 0:
            if time_period.startswith('W'):
                period = time_period
            elif time_period.startswith('M'):
                period = time_period
            else:
                period = None
 
            percentage = 0.0
            if period is not None:
                year_str = f'R{compare_year}'
                lookup_mask = (
                    (df_carrier['year'] == year_str) &
                    (df_carrier['period'] == period) &
                    (df_carrier['country'] == country)
                )
                matching_rows = df_carrier[lookup_mask]
 
                if not matching_rows.empty:
                    percentage = matching_rows['percentage'].iloc[0]
 
 
            slope = COUNTRY_COEFFICIENTS_VOLUME.get(f'{country}_SLOPE', COUNTRY_COEFFICIENTS_VOLUME['EU_SLOPE'])
            intercept = COUNTRY_COEFFICIENTS_VOLUME.get(f'{country}_INTERCEPT', COUNTRY_COEFFICIENTS_VOLUME['EU_INTERCEPT'])
            slope_lcr = COUNTRY_COEFFICIENTS_LCR.get(f'{country}_SLOPE_LCR', COUNTRY_COEFFICIENTS_LCR['EU_SLOPE_LCR'])
            poly_lcr = COUNTRY_COEFFICIENTS_LCR.get(f'{country}_POLY_LCR', COUNTRY_COEFFICIENTS_LCR['EU_POLY_LCR'])
            intercept_lcr = COUNTRY_COEFFICIENTS_LCR.get(f'{country}_INTERCEPT_LCR', COUNTRY_COEFFICIENTS_LCR['EU_INTERCEPT_LCR'])
 
            base_lcr = metrics['base_loads'] / metrics['base_carriers']
            compare_lcr = metrics['loads'] / metrics['carriers']
            base_expected_executing_carriers = (metrics['base_loads'] * slope + intercept) * (1 + percentage)
            compare_expected_executing_carriers = (metrics['loads'] * slope + intercept) * (1 + percentage)
            base_expected_lcr = metrics['base_loads'] / base_expected_executing_carriers
            compare_expected_lcr = metrics['loads'] / compare_expected_executing_carriers
 
            carrier_impact_score = compare_lcr - base_lcr + base_expected_lcr - compare_expected_lcr
            demand_impact_score = compare_expected_lcr - base_expected_lcr
 
            carrier_impact = ((intercept_lcr + slope_lcr*(carrier_impact_score + base_lcr) + poly_lcr*((carrier_impact_score + base_lcr)**2)) / (intercept_lcr + slope_lcr*base_lcr + poly_lcr*(base_lcr**2))-1)*metrics['base_cpkm']
 
            demand_impact = ((intercept_lcr + slope_lcr*(demand_impact_score + base_lcr) + poly_lcr*((demand_impact_score + base_lcr)**2)) / (intercept_lcr + slope_lcr*base_lcr + poly_lcr*(base_lcr**2))-1)*metrics['base_cpkm']
 
            impact_updates = {
                'carrier_impact': carrier_impact,
                'demand_impact': demand_impact
            }
       
        # Calculate tech impact
        if period_type in ['YoY', 'WoW'] and time_period and compare_year:
            week_num = int(time_period.replace('W', ''))
            tech_rate = get_tech_savings_rate(compare_year, week_num)
            impact_updates['tech_impact'] = metrics['cpkm'] * tech_rate
       
        # Update all impacts in one operation
        bridge_df.loc[idx, list(impact_updates.keys())] = list(impact_updates.values())
 
def calculate_aggregated_bridge_metrics(df, bridge_df, df_carrier):
    """
    Calculate bridge metrics including EU-level aggregation with optimized data handling
    """
    # Pre-calculate common aggregations
    print("Pre-calculating aggregations...")
    df_aggs = {}
    for period_type in ['YoY', 'WoW', 'MTD']:
        df_aggs[period_type] = {}
       
    # Process EU rows by period type
    print("Processing EU metrics...")
    for period_type in ['YoY', 'WoW', 'MTD']:
        eu_rows = bridge_df[
            (bridge_df['bridge_type'] == period_type) &
            (bridge_df['orig_country'] == 'EU')
        ]
       
        # Process in batches
        batch_size = 1000
        for start_idx in range(0, len(eu_rows), batch_size):
            batch = eu_rows.iloc[start_idx:start_idx + batch_size]
           
            for idx, row in batch.iterrows():
                # Get time period data
                if period_type == 'YoY':
                    base_year, compare_year = row['bridging_value'].split('_to_')
                    time_period = row['report_week']
                    key = (base_year, compare_year, time_period)
                   
                    if key not in df_aggs[period_type]:
                        df_aggs[period_type][key] = {
                            'base': df[(df['report_year'] == base_year) &
                                     (df['report_week'] == time_period)],
                            'compare': df[(df['report_year'] == compare_year) &
                                        (df['report_week'] == time_period)]
                        }
                   
                    base_data = df_aggs[period_type][key]['base']
                    compare_data = df_aggs[period_type][key]['compare']
                   
                elif period_type == 'WoW':
                    year = row['report_year']
                    w1, w2 = row['bridging_value'].split('_')[1], row['bridging_value'].split('_')[3]
                    time_period = w2
                    key = (year, w1, w2)
                   
                    if key not in df_aggs[period_type]:
                        df_aggs[period_type][key] = {
                            'base': df[(df['report_year'] == year) &
                                     (df['report_week'] == w1)],
                            'compare': df[(df['report_year'] == year) &
                                        (df['report_week'] == w2)]
                        }
                   
                    base_data = df_aggs[period_type][key]['base']
                    compare_data = df_aggs[period_type][key]['compare']
                   
                else:  # MTD
                    base_year, month, compare_year = re.match(r'(.+)_(.+)_to_(.+)_.+', row['bridging_value']).groups()
                    start_date, end_date = get_mtd_date_range(df, compare_year, int(month.replace('M', '')))
                   
                    if start_date is None:
                        continue
                       
                    key = (base_year, compare_year, month)
                    if key not in df_aggs[period_type]:
                        df_aggs[period_type][key] = {
                            'base': df[
                                (df['report_year'] == base_year) &
                                (df['report_month'] == month) &
                                (df['report_day'] >= start_date.replace(year=extract_year(base_year))) &
                                (df['report_day'] <= end_date.replace(year=extract_year(base_year)))
                            ],
                            'compare': df[
                                (df['report_year'] == compare_year) &
                                (df['report_month'] == month) &
                                (df['report_day'] >= start_date) &
                                (df['report_day'] <= end_date)
                            ]
                        }
                   
                    base_data = df_aggs[period_type][key]['base']
                    compare_data = df_aggs[period_type][key]['compare']
                    time_period = month
               
                # Calculate and update metrics
                eu_metrics = calculate_eu_metrics(
                    base_data,
                    compare_data,
                    time_period,
                    row['business']
                )
               
                update_bridge_with_eu_metrics(
                    bridge_df,
                    eu_metrics,
                    idx,
                    base_data,
                    compare_data,
                    period_type,
                    time_period,
                    compare_year if period_type in ['YoY', 'MTD'] else extract_year(row['report_year']),
                    df_carrier=df_carrier
                )
 
 
################## OP2 ####################
def get_previous_report_year(report_year, df):
    """
    Given report_year like 'R2026', return 'R2025' if it exists in df,
    otherwise return None.
    """
    try:
        year_num = int(report_year.replace('R', ''))
        prev_year = f'R{year_num - 1}'
    except Exception:
        return None
 
    if prev_year in df['report_year'].unique():
        return prev_year
 
    return None
 
def extract_op2_base_cpkm_weekly(df_op2):
    """
    Read OP2 base weekly CPKM directly from weekly_agg
    """
    base = df_op2[
        df_op2['Bridge type'] == 'weekly_agg'
    ].copy()
 
    base = base.rename(columns={
        'Report Year': 'report_year',
        'Week': 'report_week',
        'Orig_EU5': 'orig_country',
        'CpKM': 'op2_base_cpkm',
        'Cost': 'op2_base_cost',
        'Distance': 'op2_base_distance',
        'Loads': 'op2_base_loads',
        'Active Carriers': 'op2_carriers',   # ✅
        'LCR': 'op2_lcr' 
    })
 
    return base[
        [
            'report_year',
            'report_week',
            'orig_country',
            'op2_base_cpkm',
            'op2_base_cost',
            'op2_base_distance',
            'op2_base_loads',
            'op2_carriers',
            'op2_lcr'
        ]
    ]
 
def extract_op2_base_weekly_by_business(df_op2):
    """
    Apply OP2 weekly rates to ACTUAL weekly volumes
    """
    # --- 1. Filter OP2 weekly detailed ---
    op2 = df_op2[
        df_op2['Bridge type'] == 'weekly'
    ].copy()
 
    op2 = op2.rename(columns={
        'Report Year': 'report_year',
        'Week': 'report_week',
        'Orig_EU5': 'orig_country',
        'Dest_EU5': 'dest_country',
        'Business Flow': 'business',
        'Distance Band': 'distance_band',
        'Distance': 'op2_distance',
        'Cost': 'op2_cost',
        'Loads': 'op2_base_loads'
    })
 
    op2['report_year'] = op2['report_year'].astype(str)
    op2['report_week'] = op2['report_week'].astype(str)
    op2['orig_country'] = op2['orig_country'].astype(str)
    op2['dest_country'] = op2['dest_country'].astype(str)
    op2['business'] = op2['business'].astype(str)
    op2['distance_band'] = op2['distance_band'].astype(str)
    op2['business'] = op2['business'].str.upper()
 
    # --- 2. Aggregate actual weekly volumes ---
    out = op2.groupby(
        [
            'report_year',
            'report_week',
            'orig_country',
            'business',
        ],
        as_index=False
    ).agg(
        op2_base_distance=('op2_distance', 'sum'),
        op2_base_cost=('op2_cost', 'sum'),
        op2_base_loads=('op2_base_loads', 'sum')
    )
    out['op2_base_cpkm'] = out['op2_base_cost'] / out['op2_base_distance']
 
    carriers = extract_op2_base_cpkm_weekly(df_op2)
    out = out.merge(
        carriers[
            [
                'report_year',
                'report_week',
                'orig_country',
                'op2_carriers'
            ]
        ],
        on=[
            'report_year',
            'report_week',
            'orig_country'
        ],
        how='inner'  # critical: only where OP2 rate exists
    )
    return out
 
def compute_op2_normalized_cpkm_weekly(actual_df, df_op2, by_business=False):
    """
    Apply OP2 weekly rates to ACTUAL weekly volumes
    """
   
 
    # --- 1. Filter OP2 weekly detailed ---
    op2 = df_op2[
        df_op2['Bridge type'] == 'weekly'
    ].copy()
 
    op2 = op2.rename(columns={
        'Report Year': 'report_year',
        'Week': 'report_week',
        'Orig_EU5': 'orig_country',
        'Dest_EU5': 'dest_country',
        'Business Flow': 'business',
        'Distance Band': 'distance_band',
        'CpKM': 'op2_cpkm'
    })
 
    op2['report_year'] = op2['report_year'].astype(str)
    op2['report_week'] = op2['report_week'].astype(str)
    op2['orig_country'] = op2['orig_country'].astype(str)
    op2['dest_country'] = op2['dest_country'].astype(str)
    op2['business'] = op2['business'].astype(str)
    op2['distance_band'] = op2['distance_band'].astype(str)
    op2['business'] = op2['business'].str.upper()
 
 
    actual_df = actual_df.copy()
    actual_df['business'] = actual_df['business'].str.upper()
    actual_df['report_year'] = actual_df['report_year'].astype(str)
    actual_df['distance_band'] = (
        actual_df['distance_band']
        .astype(str)
        .str.strip()
        # remove leading "NN." (e.g. "05.")
        .str.replace(r'^\d+\.', '', regex=True)
    )
 
 
    # --- 2. Aggregate actual weekly volumes ---
    actual = actual_df.groupby(
        [
            'report_year',
            'report_week',
            'orig_country',
            'dest_country',
            'business',
            'distance_band'
        ],
        as_index=False
    ).agg(
        actual_distance=('distance_for_cpkm', 'sum')
    )
    # --- 3. Join OP2 rates to actual volumes ---
    merged = actual.merge(
        op2[
            [
                'report_year',
                'report_week',
                'orig_country',
                'dest_country',
                'business',
                'distance_band',
                'op2_cpkm'
            ]
        ],
        on=[
            'report_year',
            'report_week',
            'orig_country',
            'dest_country',
            'business',
            'distance_band'
        ],
        how='inner'  # critical: only where OP2 rate exists
    )
 
    # --- 4. Normalize ---
    merged['normalized_cost'] = (
        merged['op2_cpkm'] * merged['actual_distance']
    )
    if by_business:
        # --- 5. Aggregate back to origin-week ---
        out = merged.groupby(
            ['report_year', 'report_week', 'orig_country','business'],
            as_index=False
        ).agg(
            op2_normalized_cost=('normalized_cost', 'sum'),
            actual_distance=('actual_distance', 'sum')
        )
        out['op2_normalized_cpkm'] = (
            out['op2_normalized_cost'] / out['actual_distance']
        )
        logger.info(f"sum op2 norm: {out['op2_normalized_cpkm'].sum()}")
 
        return out[
            ['report_year', 'report_week', 'orig_country', 'business', 'op2_normalized_cost', 'op2_normalized_cpkm']
        ]
 
    else:
        # --- 5. Aggregate back to origin-week ---
        out = merged.groupby(
            ['report_year', 'report_week', 'orig_country'],
            as_index=False
        ).agg(
            op2_normalized_cost=('normalized_cost', 'sum'),
            actual_distance=('actual_distance', 'sum')
        )
 
        out['op2_normalized_cpkm'] = (
            out['op2_normalized_cost'] / out['actual_distance']
        )
        logger.info(f"sum op2 norm: {out['op2_normalized_cpkm'].sum()}")
 
        return out[
            ['report_year', 'report_week', 'orig_country', 'op2_normalized_cost', 'op2_normalized_cpkm']
        ]
 
def calculate_op2_carrier_demand_impacts(
    *,
    actual_loads: float,
    actual_carriers: float,
    op2_loads: float,
    op2_carriers: float,
    base_cpkm: float,
    country: str,
    df_carrier,
    compare_year: int,
    report_week: str = None,
    report_month: str = None
):
    """
    Carrier & demand impacts for OP2 weekly bridge
 
    OP2 side is EXPECTED:
      - expected_loads = op2_loads
      - expected_carriers = op2_carriers
      - expected_lcr = op2_loads / op2_carriers
 
    Compare side is ACTUAL
   
    Args:
        actual_loads: Actual number of loads
        actual_carriers: Actual number of carriers
        op2_loads: OP2 planned loads
        op2_carriers: OP2 planned carriers
        base_cpkm: Base cost per kilometer
        country: Country code for coefficient lookup
        df_carrier: DataFrame containing carrier percentage adjustments
        compare_year: Year for the comparison period
        report_week: Week identifier (e.g., 'W01')
        report_month: Month identifier (e.g., 'M01')
   
    Returns:
        tuple: (carrier_impact, demand_impact)
    """
 
    if (
        actual_loads <= 0 or actual_carriers <= 0 or
        op2_loads <= 0 or op2_carriers <= 0 or
        base_cpkm is None
    ):
        return 0.0, 0.0
   
    # Get volume coefficients
    slope = COUNTRY_COEFFICIENTS_VOLUME.get(
        str(country) + str('_SLOPE'),
        COUNTRY_COEFFICIENTS_VOLUME['EU_SLOPE']
    )
    intercept = COUNTRY_COEFFICIENTS_VOLUME.get(
        str(country) + str('_INTERCEPT'),
        COUNTRY_COEFFICIENTS_VOLUME['EU_INTERCEPT']
    )
 
    # Get LCR coefficients
    slope_lcr = COUNTRY_COEFFICIENTS_LCR.get(
        f"{country}_SLOPE_LCR",
        COUNTRY_COEFFICIENTS_LCR["EU_SLOPE_LCR"]
    )
    poly_lcr = COUNTRY_COEFFICIENTS_LCR.get(
        f"{country}_POLY_LCR",
        COUNTRY_COEFFICIENTS_LCR["EU_POLY_LCR"]
    )
    intercept_lcr = COUNTRY_COEFFICIENTS_LCR.get(
        f"{country}_INTERCEPT_LCR",
        COUNTRY_COEFFICIENTS_LCR["EU_INTERCEPT_LCR"]
    )
 
    # Determine the period identifier for lookup
    period = None
    if report_week:
        period = report_week  # e.g., 'W01'
    elif report_month:
        period = report_month  # e.g., 'M01'
   
    # Lookup percentage from df_carrier
    percentage = 0.0  # Default value
    if period is not None:
        year_str = f'R{compare_year}'
        lookup_mask = (
            (df_carrier['year'] == year_str) &
            (df_carrier['period'] == period) &
            (df_carrier['country'] == country)
        )
        matching_rows = df_carrier[lookup_mask]
       
        if not matching_rows.empty:
            percentage = matching_rows['percentage'].iloc[0]
 
    # Calculate expected executing carriers with percentage adjustment
    actual_expected_executing_carriers = (actual_loads * slope + intercept) * (1 + percentage)
   
    # Calculate LCRs
    actual_lcr = actual_loads / actual_carriers
    op2_lcr = op2_loads / op2_carriers  # OP2 expected
    actual_expected_lcr = actual_loads / actual_expected_executing_carriers
 
    # Split logic (same as YoY, but OP2 is baseline)
    carrier_impact_score = actual_lcr - actual_expected_lcr
    demand_impact_score = actual_expected_lcr - op2_lcr  # OP2 already absorbs demand
 
    # Calculate carrier impact
    carrier_impact = (
        (
            intercept_lcr +
            slope_lcr * (carrier_impact_score + op2_lcr) +
            poly_lcr * ((carrier_impact_score + op2_lcr) ** 2)
        ) /
        (
            intercept_lcr +
            slope_lcr * op2_lcr +
            poly_lcr * (op2_lcr ** 2)
        ) - 1
    ) * base_cpkm
 
    # Calculate demand impact
    demand_impact = (
        (
            intercept_lcr +
            slope_lcr * (demand_impact_score + op2_lcr) +
            poly_lcr * ((demand_impact_score + op2_lcr) ** 2)
        ) /
        (
            intercept_lcr +
            slope_lcr * op2_lcr +
            poly_lcr * (op2_lcr ** 2)
        ) - 1
    ) * base_cpkm
 
    return carrier_impact, demand_impact
 
def create_op2_weekly_bridge(df, df_op2, df_carrier, final_bridge_df):
    """
    Create OP2 weekly benchmark bridge
    Fully compatible with bridge schema
    """
 
    # --- Actual weekly CPKM ---
    # aggregate numeric metrics
    actual = (
        df[df['report_week'].notna()]
        .groupby(['report_year', 'report_week', 'orig_country'], as_index=False)
        .agg(
            actual_cost=('total_cost_usd', 'sum'),
            actual_distance=('distance_for_cpkm', 'sum'),
            actual_loads=('executed_loads', 'sum')
        )
    )
 
    # compute carriers using SAME logic as YoY
    actual['actual_carriers'] = actual.apply(
        lambda r: calculate_active_carriers(
            df[
                (df['report_year'] == r['report_year']) &
                (df['report_week'] == r['report_week']) &
                (df['orig_country'] == r['orig_country'])
            ],
            r['report_week'],
            r['orig_country']
        ),
        axis=1
    )
 
 
    actual['compare_cpkm'] = actual['actual_cost'] / actual['actual_distance']
 
    # --- OP2 metrics ---
    op2_base = extract_op2_base_cpkm_weekly(df_op2)
    op2_norm = compute_op2_normalized_cpkm_weekly(df, df_op2)
    tech_impact = calculate_tech_impact(df, df_op2, by_business=False)
    market_impact_op2 = calculate_market_rate_impact_op2(df, df_op2, by_business=False)
    logger.info(f"op2_norm shape: {op2_norm.shape}")
    logger.info(f"op2_norm columns: {op2_norm.columns.tolist()}")
    logger.info(f"op2_norm sample: {op2_norm.head()}")
   
 
 
    bridge = (
        actual
        .merge(op2_base, on=['report_year', 'report_week', 'orig_country'], how='left')
        .merge(op2_norm, on=['report_year', 'report_week', 'orig_country'], how='left')
        .merge(tech_impact, on=['report_year', 'report_week', 'orig_country'], how='left')
        .merge(market_impact_op2, on=['report_year', 'report_week', 'orig_country'], how='left')
    )
    logger.info(f"Bridge columns after merge: {bridge.columns.tolist()}")
    logger.info(f"op2_normalized_cpkm null count: {bridge['op2_normalized_cpkm'].isna().sum()}")
 
    # --- Fill bridge schema ---
    bridge['bridge_type'] = 'op2_weekly'
    bridge['business'] = 'Total'
    bridge['base_cpkm'] = bridge['op2_base_cpkm']
    bridge['normalised_cpkm'] = bridge['op2_normalized_cpkm']
 
    ### get set impact
    yoy_set_impact = get_set_impact_for_op2(final_bridge_df=final_bridge_df)
    bridge = bridge.merge(
        yoy_set_impact,
        on=[
            'report_year',
            'report_week',
            'orig_country',
            'business'
        ],
        how='left')
 
    # ✅ CALCULATE VARIANCE METRICS
    bridge['loads_variance'] = bridge['actual_loads'] - bridge['op2_base_loads']
    bridge['loads_variance_pct'] = (bridge['loads_variance'] / bridge['op2_base_loads']) * 100
 
    bridge['distance_variance_km'] = bridge['actual_distance'] - bridge['op2_base_distance']
    bridge['distance_variance_pct'] = (bridge['distance_variance_km'] / bridge['op2_base_distance']) * 100
 
    bridge['cost_variance_mm'] = (bridge['actual_cost'] - bridge['op2_base_cost']) / 1_000_000
    bridge['cost_variance_pct'] = ((bridge['actual_cost'] - bridge['op2_base_cost']) / bridge['op2_base_cost']) * 100
 
    # ✅ NORMALIZED METRICS (already calculated)
    bridge['compare_cpkm_vs_normalised_op2_cpkm'] = bridge['compare_cpkm'] - bridge['op2_normalized_cpkm']
    bridge['mix_impact'] = bridge['op2_normalized_cpkm'] - bridge['op2_base_cpkm']
 
    bridge['tech_impact'] = np.where(
        bridge['actual_distance'] > 0,
        bridge['op2_tech_impact_value'] / bridge['actual_distance'],
        0
    )
 
    bridge['market_rate_impact'] = np.where(
        bridge['actual_distance'] > 0,
        bridge['op2_market_impact'] / bridge['actual_distance'],
        0
    )
 
    bridge['op2_active_carriers'] = None
    bridge['op2_lcr'] = None
 
    bridge['bridging_value'] = (
        bridge['report_year'] + '_' + bridge['report_week'] + '_OP2'
    )
 
    bridge['w2_distance_km'] = bridge['actual_distance']
    bridge['normalised_cpkm'] = bridge['op2_normalized_cpkm']
    bridge['benchmark_gap'] = bridge['compare_cpkm'] - bridge['op2_base_cpkm']
 
    bridge[['carrier_impact', 'demand_impact']] = bridge.apply(
    lambda r: calculate_op2_carrier_demand_impacts(
        actual_loads=r['actual_loads'],
        actual_carriers=r['actual_carriers'],
        op2_loads=r['op2_base_loads'],
        op2_carriers=r['op2_carriers'],
        base_cpkm=r['op2_base_cpkm'],
        country=r['orig_country'],
        df_carrier=df_carrier,
        compare_year=int(r['report_year'].replace('R','')),
        report_week=r['report_week']
    ),
    axis=1,
    result_type='expand'
    )
 
    bridge['carrier_and_demand_impact'] = (
        bridge['carrier_impact'] + bridge['demand_impact']
    )
 
 
    # Explicitly null out non-applicable fields
    null_cols = [
        'premium_impact',
        'supply_rates'
    ]
    for col in null_cols:
        bridge[col] = None
 
    return bridge
 
def create_op2_weekly_country_business(df, df_op2, df_carrier, final_bridge_df):
    """
    OP2 weekly bridge
    Grain: Country × Week × Business
    Uses ONLY OP2 `weekly`
    """
 
    # -------------------------
    # ACTUALS
    # -------------------------
    actual = (
        df[df['report_week'].notna()]
        .groupby(
            ['report_year', 'report_week', 'orig_country', 'business'],
            as_index=False
        )
        .agg(
            actual_cost=('total_cost_usd', 'sum'),
            actual_distance=('distance_for_cpkm', 'sum'),
            actual_loads=('executed_loads', 'sum')
        )
    )
 
    actual['actual_carriers'] = actual.apply(
        lambda r: calculate_active_carriers(
            df[
                (df['report_year'] == r['report_year']) &
                (df['report_week'] == r['report_week']) &
                (df['orig_country'] == r['orig_country'])
            ],
            r['report_week'],
            r['orig_country']
        ),
        axis=1
    )
 
    actual['compare_cpkm'] = actual['actual_cost'] / actual['actual_distance']
 
    # -------------------------
    # OP2 NORMALIZED (business-aware)
    # -------------------------
    # --- OP2 metrics ---
    op2_base = extract_op2_base_weekly_by_business(df_op2)
    op2_norm = compute_op2_normalized_cpkm_weekly(df, df_op2, by_business=True)
    tech_impact = calculate_tech_impact(df, df_op2, by_business=True)
    market_impact_op2 = calculate_market_rate_impact_op2(df, df_op2, by_business=True)
    logger.info(f"op2_norm shape: {op2_norm.shape}")
    logger.info(f"op2_norm columns: {op2_norm.columns.tolist()}")
    logger.info(f"op2_norm sample: {op2_norm.head()}")
   
    bridge = (
        actual
        .merge(op2_base, on=['report_year', 'report_week', 'orig_country', 'business'], how='left')
        .merge(op2_norm, on=['report_year', 'report_week', 'orig_country', 'business'], how='left')
        .merge(tech_impact, on=['report_year', 'report_week', 'orig_country', 'business'], how='left')
        .merge(market_impact_op2, on=['report_year', 'report_week', 'orig_country', 'business'], how='left')
    )
 
    # -------------------------
    # METRICS
    # -------------------------
    bridge['bridge_type'] = 'op2_weekly'
    bridge['business'] = bridge['business'].fillna('UNKNOWN')
    bridge['base_cpkm'] = bridge['op2_base_cpkm']
    bridge['normalised_cpkm'] = bridge['op2_normalized_cpkm']
 
    ### get set impact
    yoy_set_impact = get_set_impact_for_op2(final_bridge_df=final_bridge_df)
    bridge = bridge.merge(
        yoy_set_impact,
        on=[
            'report_year',
            'report_week',
            'orig_country',
            'business'
        ],
        how='left')
 
    # ✅ CALCULATE VARIANCE METRICS
    bridge['loads_variance'] = bridge['actual_loads'] - bridge['op2_base_loads']
    bridge['loads_variance_pct'] = (bridge['loads_variance'] / bridge['op2_base_loads']) * 100
 
    bridge['distance_variance_km'] = bridge['actual_distance'] - bridge['op2_base_distance']
    bridge['distance_variance_pct'] = (bridge['distance_variance_km'] / bridge['op2_base_distance']) * 100
 
    bridge['cost_variance_mm'] = (bridge['actual_cost'] - bridge['op2_base_cost']) / 1_000_000
    bridge['cost_variance_pct'] = ((bridge['actual_cost'] - bridge['op2_base_cost']) / bridge['op2_base_cost']) * 100
 
    # ✅ NORMALIZED METRICS (already calculated)
    bridge['compare_cpkm_vs_normalised_op2_cpkm'] = bridge['compare_cpkm'] - bridge['op2_normalized_cpkm']
    bridge['mix_impact'] = bridge['op2_normalized_cpkm'] - bridge['op2_base_cpkm']
 
    # ✅ PLACEHOLDERS FOR METRICS REQUIRING ADDITIONAL DATA
 
    bridge['tech_impact'] = np.where(
        bridge['actual_distance'] > 0,
        bridge['op2_tech_impact_value'] / bridge['actual_distance'],
        0
    )
 
    bridge['market_rate_impact'] = np.where(
        bridge['actual_distance'] > 0,
        bridge['op2_market_impact'] / bridge['actual_distance'],
        0
    )
 
    bridge['op2_active_carriers'] = None
    bridge['op2_lcr'] = None
 
    bridge['bridging_value'] = (
        bridge['report_year'] + '_' + bridge['report_week'] + '_OP2'
    )
 
    # -------------------------
    # CARRIER / DEMAND
    # -------------------------
    bridge[['carrier_impact', 'demand_impact']] = bridge.apply(
        lambda r: calculate_op2_carrier_demand_impacts(
            actual_loads=r['actual_loads'],
            actual_carriers=r['actual_carriers'],
            op2_loads=r['actual_loads'],        # OP2 expected = normalized
            op2_carriers=r['op2_carriers'],
            base_cpkm=r['op2_normalized_cpkm'],
            country=r['orig_country'],
            df_carrier=df_carrier,
            compare_year=int(r['report_year'].replace('R', '')),
            report_week=r['report_week']
        ),
        axis=1,
        result_type='expand'
    )
 
    bridge['carrier_and_demand_impact'] = (
        bridge['carrier_impact'] + bridge['demand_impact']
    )
 
    return bridge
 
def calculate_tech_impact(df, df_op2, by_business=False):
    """
    Apply OP2 weekly tech impact to ACTUAL weekly volumes
    Grain:
    report_year, report_week, orig_country, dest_country, business, distance_band
    """
 
    # --------------------------------------------------
    # 1. Filter + prepare OP2 weekly detailed
    # --------------------------------------------------
    op2 = df_op2[df_op2['Bridge type'] == 'weekly'].copy()
 
    op2 = op2.rename(columns={
        'Report Year': 'report_year',
        'Week': 'report_week',
        'Orig_EU5': 'orig_country',
        'Dest_EU5': 'dest_country',
        'Business Flow': 'business',
        'Distance Band': 'distance_band',
        'Distance': 'op2_distance',
        'Tech Initiative': 'op2_tech_impact',
    })
 
    logger.info("OP2 TECH INITIATIVE STATS")
    logger.info(op2['op2_tech_impact'].describe())
    logger.info(
        op2[['op2_tech_impact']]
        .dropna()
        .head(10)
    )
 
 
    # normalize types
    for c in [
        'report_year', 'report_week',
        'orig_country', 'dest_country',
        'business', 'distance_band'
    ]:
        op2[c] = op2[c].astype(str)
 
    op2['business'] = op2['business'].str.upper()
 
    # OP2 tech impact value (distance × tech rate)
    op2['op2_tech_impact_value'] = (
        op2['op2_distance'] * op2['op2_tech_impact']
    )
 
    logger.info("OP2 TECH IMPACT VALUE")
    logger.info(
        op2[['op2_tech_impact_value']]
        .sum()
    )
 
 
    # --------------------------------------------------
    # 2. Aggregate ACTUAL weekly volumes
    # --------------------------------------------------
    actual = df.groupby(
        [
            'report_year',
            'report_week',
            'orig_country',
            'dest_country',
            'business',
            'distance_band'
        ],
        as_index=False
    ).agg(
        actual_cost=('total_cost_usd', 'sum')
    )
 
    # normalize types
    for c in [
        'report_year', 'report_week',
        'orig_country', 'dest_country',
        'business', 'distance_band'
    ]:
        actual[c] = actual[c].astype(str)
 
    actual['business'] = actual['business'].str.upper()
 
    actual['distance_band'] = (
        actual['distance_band']
        .astype(str)
        .str.strip()
        .str.replace(r'^\d+\.', '', regex=True)
    )
 
    # --------------------------------------------------
    # 3. Compute tech rate PER ROW (all weeks, all years)
    # --------------------------------------------------
    def _get_rate(row):
        week_num = int(row['report_week'].replace('W', ''))
        year_num = int(row['report_year'].replace('R', ''))
        return get_tech_savings_rate(year_num, week_num)
 
    actual['tech_rate'] = actual.apply(_get_rate, axis=1)
 
    actual['TY_tech_impact'] = actual['actual_cost'] * actual['tech_rate']
 
    logger.info("OP2 TECH IMPACT VALUE")
    logger.info(actual['TY_tech_impact'].describe())
    logger.info(
        actual[['TY_tech_impact']]
        .dropna()
        .head(10)
    )
 
    # --------------------------------------------------
    # 4. Join OP2 weekly tech impact
    # --------------------------------------------------
    merged = actual.merge(
        op2[
            [
                'report_year',
                'report_week',
                'orig_country',
                'dest_country',
                'business',
                'distance_band',
                'op2_tech_impact_value'
            ]
        ],
        on=[
            'report_year',
            'report_week',
            'orig_country',
            'dest_country',
            'business',
            'distance_band'
        ],
        how='inner'   # ONLY where OP2 exists
    )
 
    # --------------------------------------------------
    # 5. Final deltas
    # --------------------------------------------------
    merged['tech_impact_delta'] = (
        merged['TY_tech_impact'] - merged['op2_tech_impact_value']
    )
 
    if by_business:
   
        # --- 5. Aggregate back to origin-week ---
        out = merged.groupby(
            ['report_year', 'report_week', 'orig_country','business'],
            as_index=False
        ).agg(
            op2_tech_impact_value=('tech_impact_delta', 'sum')
        )
        return out[
            ['report_year', 'report_week', 'orig_country', 'business', 'op2_tech_impact_value']
        ]
    else:
        # --- 5. Aggregate back to origin-week ---
        out = merged.groupby(
            ['report_year', 'report_week', 'orig_country'],
            as_index=False
        ).agg(
            op2_tech_impact_value=('tech_impact_delta', 'sum')
        )
        return out[
            ['report_year', 'report_week', 'orig_country', 'op2_tech_impact_value']
        ]
 
def calculate_market_rate_impact_op2(df, df_op2, by_business=False):
    """
    Calculate OP2 market rate impact (ABSOLUTE USD) applied to ACTUAL volumes.
 
    Logic:
    - If LY distance == 0:
        impact = TY_cost * mr_by_cpkm - TY_cost
    - Else:
        impact = LY_cpkm * TY_distance * mr_by_cpkm
 
    Output grain:
    - by_business=False:
        report_year, report_week, orig_country
    - by_business=True:
        report_year, report_week, orig_country, business
    """
 
    # ==================================================
    # 1. Prepare OP2 weekly MARKET RATE signal
    # ==================================================
    op2 = df_op2[df_op2['Bridge type'] == 'weekly'].copy()
 
    op2 = op2.rename(columns={
        'Report Year': 'report_year',
        'Week': 'report_week',
        'Orig_EU5': 'orig_country',
        'Dest_EU5': 'dest_country',
        'Business Flow': 'business',
        'Distance Band': 'distance_band',
        'CpKM': 'op2_cpkm',
        'Market Rate': 'op2_market_rate',
    })
 
    dim_cols = [
        'report_year', 'report_week',
        'orig_country', 'dest_country',
        'business', 'distance_band'
    ]
 
    for c in dim_cols:
        op2[c] = op2[c].astype(str)
 
    op2['business'] = op2['business'].str.upper()
 
    # OP2 market-rate multiplier
    op2['mr_by_cpkm'] = np.where(
        op2['op2_cpkm'] > 0,
        op2['op2_market_rate'] / op2['op2_cpkm'],
        np.nan
    )
 
    op2 = op2[dim_cols + ['mr_by_cpkm']]
 
    # ==================================================
    # 2. Prepare ACTUAL data (TY)
    # ==================================================
    actual = df.copy()
 
    for c in dim_cols:
        actual[c] = actual[c].astype(str)
 
    actual['business'] = actual['business'].str.upper()
    actual['distance_band'] = (
        actual['distance_band']
        .astype(str)
        .str.strip()
        .str.replace(r'^\d+\.', '', regex=True)
    )
 
    group_cols = [
        'report_year', 'report_week',
        'orig_country', 'dest_country',
        'distance_band', 'business'
    ]
 
    actual_agg = actual.groupby(group_cols, as_index=False).agg(
        distance=('distance_for_cpkm', 'sum'),
        cost=('total_cost_usd', 'sum')
    )
 
    actual_agg['cpkm'] = np.where(
        actual_agg['distance'] > 0,
        actual_agg['cost'] / actual_agg['distance'],
        0.0
    )
 
    # ==================================================
    # 3. Build LY via SELF-JOIN (SAFE)
    # ==================================================
    actual_agg['year_num'] = actual_agg['report_year'].str.replace('R', '').astype(int)
    actual_agg['ly_report_year'] = 'R' + (actual_agg['year_num'] - 1).astype(str)
 
    ly = actual_agg[
        group_cols + ['cpkm', 'distance', 'cost']
    ].rename(columns={
        'cpkm': 'ly_cpkm',
        'distance': 'ly_distance',
        'cost': 'ly_cost'
    })
 
    merged = actual_agg.merge(
        ly,
        left_on=[
            'ly_report_year', 'report_week',
            'orig_country', 'dest_country',
            'distance_band', 'business'
        ],
        right_on=[
            'report_year', 'report_week',
            'orig_country', 'dest_country',
            'distance_band', 'business'
        ],
        how='left',
        suffixes=('', '_drop')
    )
 
    merged.drop(columns=[c for c in merged.columns if c.endswith('_drop')], inplace=True)
 
    # ==================================================
    # 4. Merge OP2 market-rate signal
    # ==================================================
    merged = merged.merge(
        op2,
        on=group_cols,
        how='left'
    )
 
    # ==================================================
    # 5. Apply OP2 MARKET RATE LOGIC (ABSOLUTE USD)
    # ==================================================
    merged['op2_market_impact'] = np.where(
        merged['mr_by_cpkm'].notna(),
        np.where(
            merged['ly_distance'].fillna(0) == 0,
            # No LY → apply multiplier to TY cost
            merged['cost'] * merged['mr_by_cpkm'] - merged['cost'],
            # Normal case
            merged['ly_cpkm'] * merged['distance'] * merged['mr_by_cpkm']
        ),
        0.0
    )
 
    merged['op2_market_impact'] = merged['op2_market_impact'].fillna(0.0)
 
    # ==================================================
    # 6. FINAL AGGREGATION
    # ==================================================
    if by_business:
        out = merged.groupby(
            ['report_year', 'report_week', 'orig_country', 'business'],
            as_index=False
        ).agg(
            op2_market_impact=('op2_market_impact', 'sum')
        )
 
        return out
 
    else:
        out = merged.groupby(
            ['report_year', 'report_week', 'orig_country'],
            as_index=False
        ).agg(
            op2_market_impact=('op2_market_impact', 'sum')
        )
 
        return out
 
def get_set_impact_for_op2(final_bridge_df):
    """
    Create OP2 weekly bridge rows.
    SET impact is sourced from YoY bridge before carrier/demand adjustment.
    """
 
    logger.info("Preparing YoY SET lookup for OP2 weekly bridge...")
 
    yoy_set_lookup = (
        final_bridge_df[
            (final_bridge_df['bridge_type'] == 'YoY') &
            (final_bridge_df['set_impact'].notna())
        ][
            [
                'report_year',
                'report_week',
                'orig_country',
                'business',
                'set_impact'
            ]
        ]
    )
    return yoy_set_lookup
 
def get_market_rate_impact_total_of_ty(final_bridge_df):
    """
    Create OP2 weekly bridge rows.
    SET impact is sourced from YoY bridge before carrier/demand adjustment.
    """
 
    logger.info("Preparing YoY SET lookup for OP2 weekly bridge...")
 
    yoy_market_rate_lookup = (
        final_bridge_df[
            (final_bridge_df['bridge_type'] == 'YoY') &
            (final_bridge_df['set_impact'].notna())
        ][
            [
                'report_year',
                'report_week',
                'orig_country',
                'business',
                'market_rate_impact'
            ]
        ]
    )
 
    yoy_market_rate_lookup['market_rate_impact'] = yoy_market_rate_lookup['market_rate_impact']*1_000_000
 
    return yoy_market_rate_lookup
 
def adjust_op2_carrier_demand_impacts(bridge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust carrier and demand impacts for OP2 weekly bridges
    using EXACT same reconciliation logic as other bridges.
    """
 
    logger.info("Adjusting OP2 carrier & demand impacts (reconciling)...")
 
    mask_op2 = bridge_df['bridge_type'] == 'op2_weekly'
 
    # --------------------------------------------------
    # 1. Expected CPKM
    # --------------------------------------------------
    bridge_df.loc[mask_op2, 'expected_cpkm'] = bridge_df.loc[mask_op2].apply(
        lambda row:
            row['normalised_cpkm']
            + (row['carrier_impact'] or 0)
            + (row['demand_impact'] or 0)
            + (row['premium_impact'] or 0)
            + (row['market_rate_impact'] or 0)
            + (row['set_impact'] or 0)
            + (row['tech_impact'] or 0),
        axis=1
    )
 
    # --------------------------------------------------
    # 2. Discrepancy
    # --------------------------------------------------
    bridge_df.loc[mask_op2, 'discrepancy'] = (
        bridge_df.loc[mask_op2, 'compare_cpkm']
        - bridge_df.loc[mask_op2, 'expected_cpkm']
    )
 
    # --------------------------------------------------
    # 3. Rebalance carrier & demand (IDENTICAL LOGIC)
    # --------------------------------------------------
    for idx, row in bridge_df.loc[mask_op2].iterrows():
 
        if (
            pd.notnull(row['discrepancy']) and
            pd.notnull(row['carrier_impact']) and
            pd.notnull(row['demand_impact'])
        ):
            total = abs(row['carrier_impact']) + abs(row['demand_impact'])
 
            if total > 0:
                carrier_weight = abs(row['carrier_impact']) / total
                demand_weight = abs(row['demand_impact']) / total
 
                new_carrier = row['carrier_impact'] + row['discrepancy'] * carrier_weight
                new_demand = row['demand_impact'] + row['discrepancy'] * demand_weight
 
                bridge_df.at[idx, 'carrier_impact'] = new_carrier
                bridge_df.at[idx, 'demand_impact'] = new_demand
                bridge_df.at[idx, 'carrier_and_demand_impact'] = new_carrier + new_demand
 
    # --------------------------------------------------
    # 4. Distance (OP2 always weekly actual)
    # --------------------------------------------------
    distance = bridge_df.loc[mask_op2, 'actual_distance']
 
    impacts = [
        'mix_impact',
        'carrier_impact',
        'demand_impact',
        'carrier_and_demand_impact',
        'premium_impact',
        'market_rate_impact',
        'set_impact',
        'tech_impact'
    ]
 
    for impact in impacts:
        bridge_df.loc[mask_op2, f'{impact}_mm'] = (
            bridge_df.loc[mask_op2, impact].fillna(0)
            * distance
            / 1_000_000
        )
 
    # --------------------------------------------------
    # 5. Cleanup
    # --------------------------------------------------
    bridge_df.drop(
        columns=['expected_cpkm', 'discrepancy'],
        inplace=True,
        errors='ignore'
    )
 
    logger.info(
        f"OP2 rows rebalanced: {mask_op2.sum()}"
    )
 
    return bridge_df
 
 
def extract_op2_base_cpkm_monthly(df_op2):
    """
    Read OP2 base monthly CPKM directly from monthly_agg
    """
    base = df_op2[
        df_op2['Bridge type'] == 'monthly_agg'
    ].copy()
 
    base = base.rename(columns={
        'Report Year': 'report_year',
        'Period': 'report_month',  # This is your M01, M02, etc.
        'Orig_EU5': 'orig_country',
        'CpKM': 'op2_base_cpkm',
        'Cost': 'op2_base_cost',
        'Distance': 'op2_base_distance',
        'Loads': 'op2_base_loads',
        'Active Carriers': 'op2_carriers',
        'LCR': 'op2_lcr'
    })
 
    return base[
        [
            'report_year',
            'report_month',
            'orig_country',
            'op2_base_cpkm',
            'op2_base_cost',
            'op2_base_distance',
            'op2_base_loads',
            'op2_carriers',
            'op2_lcr'
        ]
    ]
 
def compute_op2_normalized_cpkm_monthly(actual_df, df_op2):
    """
    Apply OP2 monthly rates to ACTUAL monthly volumes
    """
 
    # Filter OP2 monthly detailed bridge
    op2 = df_op2[
        df_op2['Bridge type'] == 'monthly_bridge'
    ].copy()
 
    op2 = op2.rename(columns={
        'Report Year': 'report_year',
        'Report Month': 'report_month',  # Use Report Month column
        'Orig_EU5': 'orig_country',
        'Dest_EU5': 'dest_country',
        'Business Flow': 'business',
        'Distance Band': 'distance_band',
        'CpKM': 'op2_cpkm'
    })
 
    # Normalize data types
    op2['report_year'] = op2['report_year'].astype(str)
    op2['report_month'] = op2['report_month'].astype(str)
    op2['orig_country'] = op2['orig_country'].astype(str)
    op2['dest_country'] = op2['dest_country'].astype(str)
    op2['business'] = op2['business'].str.upper()
    op2['distance_band'] = op2['distance_band'].astype(str)
 
    actual_df = actual_df.copy()
    actual_df['report_year'] = actual_df['report_year'].astype(str)
    actual_df['report_month'] = actual_df['report_month'].astype(str)
    actual_df['business'] = actual_df['business'].str.upper()
    actual_df['distance_band'] = (
        actual_df['distance_band']
        .astype(str)
        .str.strip()
        .str.replace(r'^\d+\.', '', regex=True)
    )
 
    # Aggregate actual monthly volumes
    actual = actual_df.groupby(
        [
            'report_year',
            'report_month',
            'orig_country',
            'dest_country',
            'business',
            'distance_band'
        ],
        as_index=False
    ).agg(
        actual_distance=('distance_for_cpkm', 'sum')
    )
 
    # Join OP2 rates to actual volumes
    merged = actual.merge(
        op2[
            [
                'report_year',
                'report_month',
                'orig_country',
                'dest_country',
                'business',
                'distance_band',
                'op2_cpkm'
            ]
        ],
        on=[
            'report_year',
            'report_month',
            'orig_country',
            'dest_country',
            'business',
            'distance_band'
        ],
        how='inner'
    )
 
    # Calculate normalized cost
    merged['normalized_cost'] = (
        merged['op2_cpkm'] * merged['actual_distance']
    )
 
    # Aggregate back to origin-month
    out = merged.groupby(
        ['report_year', 'report_month', 'orig_country'],
        as_index=False
    ).agg(
        op2_normalized_cost=('normalized_cost', 'sum'),
        actual_distance=('actual_distance', 'sum')
    )
 
    out['op2_normalized_cpkm'] = (
        out['op2_normalized_cost'] / out['actual_distance']
    )
 
    return out[
        ['report_year', 'report_month', 'orig_country', 'op2_normalized_cost', 'op2_normalized_cpkm']
    ]
 
def create_op2_monthly_bridge(df, df_op2, df_carrier):
    """
    Create OP2 monthly benchmark bridge
    """
 
    # Actual monthly CPKM
    actual = (
        df[df['report_month'].notna()]
        .groupby(['report_year', 'report_month', 'orig_country'], as_index=False)
        .agg(
            actual_cost=('total_cost_usd', 'sum'),
            actual_distance=('distance_for_cpkm', 'sum'),
            actual_loads=('executed_loads', 'sum')
        )
    )
 
    # Compute carriers using same logic as YoY
    actual['actual_carriers'] = actual.apply(
        lambda r: calculate_active_carriers(
            df[
                (df['report_year'] == r['report_year']) &
                (df['report_month'] == r['report_month']) &
                (df['orig_country'] == r['orig_country'])
            ],
            r['report_month'],  # Pass month instead of week
            r['orig_country']
        ),
        axis=1
    )
 
    actual['compare_cpkm'] = actual['actual_cost'] / actual['actual_distance']
 
    # OP2 metrics
    op2_base = extract_op2_base_cpkm_monthly(df_op2)
    op2_norm = compute_op2_normalized_cpkm_monthly(df, df_op2)
 
    bridge = (
        actual
        .merge(op2_base, on=['report_year', 'report_month', 'orig_country'], how='left')
        .merge(op2_norm, on=['report_year', 'report_month', 'orig_country'], how='left')
    )
 
    # Fill bridge schema
    bridge['bridge_type'] = 'op2_monthly'
    bridge['business'] = 'Total'
 
    # Populate OP2 columns
    bridge['op2_loads'] = bridge['op2_base_loads']
    bridge['op2_distance_km'] = bridge['op2_base_distance']
    bridge['op2_cost_usd'] = bridge['op2_base_cost']
    bridge['op2_cpkm'] = bridge['op2_base_cpkm']
 
    # Calculate variance metrics
    bridge['loads_variance'] = bridge['actual_loads'] - bridge['op2_base_loads']
    bridge['loads_variance_pct'] = (bridge['loads_variance'] / bridge['op2_base_loads']) * 100
 
    bridge['distance_variance_km'] = bridge['actual_distance'] - bridge['op2_base_distance']
    bridge['distance_variance_pct'] = (bridge['distance_variance_km'] / bridge['op2_base_distance']) * 100
 
    bridge['cpkm_variance'] = bridge['compare_cpkm'] - bridge['op2_base_cpkm']
    bridge['cpkm_variance_pct'] = (bridge['cpkm_variance'] / bridge['op2_base_cpkm']) * 100
 
    bridge['cost_variance_mm'] = (bridge['actual_cost'] - bridge['op2_base_cost']) / 1_000_000
    bridge['cost_variance_pct'] = ((bridge['actual_cost'] - bridge['op2_base_cost']) / bridge['op2_base_cost']) * 100
 
    # Normalized metrics
    bridge['normalized_variance'] = bridge['compare_cpkm'] - bridge['op2_normalized_cpkm']
    bridge['mix_impact'] = bridge['op2_normalized_cpkm'] - bridge['op2_base_cpkm']
 
    bridge['bridging_value'] = (
        bridge['report_year'] + '_' + bridge['report_month'] + '_OP2'
    )
 
    bridge['m2_distance_km'] = bridge['actual_distance']
    bridge['normalised_cpkm'] = bridge['op2_normalized_cpkm']
    bridge['benchmark_gap'] = bridge['compare_cpkm'] - bridge['op2_cpkm']
 
    # Calculate carrier and demand impacts
    bridge[['carrier_impact', 'demand_impact']] = bridge.apply(
        lambda r: calculate_op2_carrier_demand_impacts(
            actual_loads=r['actual_loads'],
            actual_carriers=r['actual_carriers'],
            op2_loads=r['op2_base_loads'],
            op2_carriers=r['op2_carriers'],
            base_cpkm=r['op2_base_cpkm'],
            country=r['orig_country'],
            df_carrier=df_carrier,
            compare_year=int(r['report_year'].replace('R','')),
            report_month=r['report_month']  # Pass month instead of week
        ),
        axis=1,
        result_type='expand'
    )
 
    bridge['carrier_and_demand_impact'] = (
        bridge['carrier_impact'] + bridge['demand_impact']
    )
 
    # Null out non-applicable fields
    null_cols = [
        'base_cpkm', 'market_rate_impact',
        'tech_impact', 'set_impact', 'premium_impact',
        'supply_rates', 'report_week'
    ]
    for col in null_cols:
        bridge[col] = None
 
    return bridge
 
def process_s3_data(source_bucket, source_key, carrier_scaling_bucket, carrier_scaling_key, op2_bucket, op2_key, destination_bucket, destination_key):
    """
    Main processing function for S3 data
    """
    # Load data from S3
    logger.info("Loading and processsing data from S3...")
    start_time_main = time.time()  
    df, df_carrier, df_op2 = load_and_process_data(source_bucket, source_key, carrier_scaling_bucket, carrier_scaling_key, op2_bucket, op2_key)
    df = df[df['report_year'].isin(['R2025','R2026'])]
    df = df[df['report_week'].isin(['W01','W02','W03','W04','W05'])]
    df_carrier.columns = df_carrier.columns.str.strip()
 
    # ✅ ADD THIS - Convert percentage to numeric
    df_carrier['percentage'] = pd.to_numeric(df_carrier['percentage'], errors='coerce')
   
    # Validate data after loading
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"DataFrame dtypes: \n{df.dtypes}")
    logger.info(f"Sample of report_day values: \n{df['report_day'].head()}")
    logger.info(f'Loading from S3 took: {time.time()-start_time_main:.2f} seconds')
   
    # Create bridge structure
    start_time_main = time.time()
    logger.info("Creating bridge structure...")
    bridge_df = create_bridge_structure(df)
    logger.info(f'Bridge structure creation took: {time.time()-start_time_main:.2f} seconds')
   
    # Create total bridge structure
    start_time_main = time.time()
    logger.info("Creating total bridge structure...")
    total_bridge_df = create_bridge_structure_for_totals(df)
    logger.info(f'Total bridge structure creation took: {time.time()-start_time_main:.2f} seconds')
   
    # Calculate YoY metrics
    start_time_main = time.time()
    logger.info("Calculating YoY metrics...")
    years = sorted(df['report_year'].unique())
    total_combinations = len(years) - 1
    processed = 0
   
    for i in range(len(years)-1):
        base_year = years[i]
        compare_year = years[i+1]  # Only next year, not all subsequent years
        processed += 1
        logger.info(f"Processing year pair {base_year}-{compare_year} ({processed}/{total_combinations})")
        calculate_bridge_metrics(df, bridge_df, base_year, compare_year, df_carrier)
    logger.info(f'YoY calculations took: {time.time()-start_time_main:.2f} seconds')
   
    # Calculate WoW metrics
    start_time_main = time.time()
    logger.info("Calculating WoW metrics...")
    calculate_wow_bridge_metrics(df, bridge_df, df_carrier)
    logger.info(f'WoW calculations took: {time.time()-start_time_main:.2f} seconds')
 
    # Calculate MTD metrics
    start_time_main = time.time()
    logger.info("Calculating MTD metrics...")
    calculate_mtd_bridge_metrics(df, bridge_df, df_carrier)
    logger.info(f'MTD calculations took: {time.time()-start_time_main:.2f} seconds')
   
    # Calculate EU aggregated metrics
    start_time_main = time.time()
    logger.info("Calculating EU aggregated metrics...")
    calculate_aggregated_bridge_metrics(df, bridge_df, df_carrier)
    logger.info(f'EU aggregated calculations took: {time.time()-start_time_main:.2f} seconds')
 
    # Calculate total metrics
    start_time_main = time.time()
    logger.info("Calculating total metrics...")
    calculate_aggregated_total_bridge_metrics(df, total_bridge_df, df_carrier)
    logger.info(f'Total metrics calculations took: {time.time()-start_time_main:.2f} seconds')
 
    # Combine results
    logger.info("Combining and adjusting final results...")
    final_bridge_df = pd.concat([bridge_df, total_bridge_df],ignore_index=True) #adding total_bridge
 
    ### adjusting carrier and demand impacts:
 
    final_bridge_df = adjust_carrier_demand_impacts(final_bridge_df)
   
    logger.info("Creating OP2 weekly COUNTRY-TOTAL bridge (test phase)...")
    start_time = time.time()
 
    op2_weekly_country_total_df = create_op2_weekly_bridge(
        df=df,
        df_op2=df_op2,
        df_carrier=df_carrier,
        final_bridge_df=final_bridge_df
    )
 
    logger.info(
        f"OP2 weekly COUNTRY-TOTAL rows: "
        f"{len(op2_weekly_country_total_df)}"
    )
 
    final_bridge_df = pd.concat(
    [final_bridge_df, op2_weekly_country_total_df],
    ignore_index=True
    )
 
    logger.info(
        f"OP2 weekly COUNTRY-TOTAL creation took "
        f"{time.time() - start_time:.2f} seconds"
    )
 
    logger.info("Creating OP2 weekly COUNTRY-BUSINESS bridge...")
    start_time = time.time()
 
    op2_weekly_country_business_df = create_op2_weekly_country_business(
        df=df,
        df_op2=df_op2,
        df_carrier=df_carrier,
        final_bridge_df=final_bridge_df
    )
 
    logger.info(
        f"OP2 weekly COUNTRY-BUSINESS rows: "
        f"{len(op2_weekly_country_business_df)}"
    )
 
    # Guardrails
    assert op2_weekly_country_business_df['business'].ne('Total').any()
    assert op2_weekly_country_business_df['orig_country'].ne('EU').all()
 
    final_bridge_df = pd.concat(
        [final_bridge_df, op2_weekly_country_business_df],
        ignore_index=True
    )
 
    # 4. Adjust OP2 impacts (NO rebalancing)
    final_bridge_df = adjust_op2_carrier_demand_impacts(final_bridge_df)
   
    # Export to S3
    logger.info("Exporting results to S3...")
    start_time_main = time.time()
    export_for_quicksight(final_bridge_df, destination_bucket, destination_key)
    logger.info(f'Export to S3 took: {time.time()-start_time_main:.2f} seconds')
   
    logger.info("Processing completed successfully")
   
    # Log memory usage
    memory_usage = final_bridge_df.memory_usage(deep=True).sum() / 1024 / 1024  # Convert to MB
    logger.info(f'Final DataFrame memory usage: {memory_usage:.2f} MB')
   
    return {
        'message': "Processing completed successfully",
        'rows_processed': len(final_bridge_df),
        'years_processed': len(years),
        'source_file': f's3://{source_bucket}/{source_key}',
        'destination_file': f's3://{destination_bucket}/{destination_key}',
        'memory_usage_mb': round(memory_usage, 2)
    }
 
def main():
    try:
        # Get environment variables
        source_bucket = os.environ.get('SOURCE_BUCKET')
        source_key = os.environ.get('SOURCE_KEY')
        carrier_scaling_bucket = os.environ.get('CARRIER_SCALING_BUCKET')
        carrier_scaling_key = os.environ.get('CARRIER_SCALING_KEY')
        op2_bucket = os.environ.get('OP2_BUCKET')
        op2_key = os.environ.get('OP2_KEY')
        destination_bucket = os.environ.get('DESTINATION_BUCKET')
        destination_key = os.environ.get('DESTINATION_KEY')
       
        # Log the start of execution
        logger.info(f'Starting analytics processing')
        logger.info(f'Source: s3://{source_bucket}/{source_key}')
        logger.info(f'Scaling: s3://{carrier_scaling_bucket}/{carrier_scaling_key}')
        logger.info(f'Destination: s3://{destination_bucket}/{destination_key}')
       
        # Validate that all required environment variables are set
        if not all([source_bucket, source_key, destination_bucket, destination_key]):
            raise ValueError("Missing required environment variables. Please check SOURCE_BUCKET, SOURCE_KEY, DESTINATION_BUCKET, and DESTINATION_KEY are set.")
       
        # Process the data
        results = process_s3_data(source_bucket, source_key, carrier_scaling_bucket, carrier_scaling_key, op2_bucket, op2_key, destination_bucket, destination_key)
       
        # Log successful completion
        logger.info({
            'message': 'Execution completed successfully',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
       
        return {
            'statusCode': 200,
            'body': results
        }
       
    except Exception as e:
        # Log the error with full details
        logger.error({
            'message': 'Error in execution',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, exc_info=True)
       
        return {
            'statusCode': 500,
            'body': f'Error processing file: {str(e)}'
        } 
if __name__ == "__main__":
    logger.info("=== STARTING MAIN FUNCTION ===")
    main()
    logger.info("=== MAIN FUNCTION COMPLETED ===")
 