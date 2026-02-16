"""
Configuration Settings

Contains country-specific coefficients for volume and LCR calculations,
as well as tech savings rate configuration.

Environment Variables Required:
    Volume Coefficients:
        - DE_SLOPE, DE_INTERCEPT
        - ES_SLOPE, ES_INTERCEPT
        - FR_SLOPE, FR_INTERCEPT
        - IT_SLOPE, IT_INTERCEPT
        - UK_SLOPE, UK_INTERCEPT
        - EU_SLOPE, EU_INTERCEPT

    LCR Coefficients:
        - {COUNTRY}_SLOPE_LCR, {COUNTRY}_POLY_LCR, {COUNTRY}_INTERCEPT_LCR
        (for DE, ES, FR, IT, UK, EU)

    Tech Savings Rates:
        - TECH_SAVINGS_RATE_{YEAR}_W{START}_W{END}
        Example: TECH_SAVINGS_RATE_2025_W27_W46=-0.0571
"""

import os
import re
from .logging_config import logger


# Volume to Carriers Coefficients
# Used to calculate expected executing carriers based on load volume
COUNTRY_COEFFICIENTS_VOLUME = {
    "DE_SLOPE": float(os.environ.get("DE_SLOPE", 0)),
    "ES_SLOPE": float(os.environ.get("ES_SLOPE", 0)),
    "FR_SLOPE": float(os.environ.get("FR_SLOPE", 0)),
    "IT_SLOPE": float(os.environ.get("IT_SLOPE", 0)),
    "UK_SLOPE": float(os.environ.get("UK_SLOPE", 0)),
    "EU_SLOPE": float(os.environ.get("EU_SLOPE", 0)),
    "DE_INTERCEPT": float(os.environ.get("DE_INTERCEPT", 0)),
    "ES_INTERCEPT": float(os.environ.get("ES_INTERCEPT", 0)),
    "FR_INTERCEPT": float(os.environ.get("FR_INTERCEPT", 0)),
    "IT_INTERCEPT": float(os.environ.get("IT_INTERCEPT", 0)),
    "UK_INTERCEPT": float(os.environ.get("UK_INTERCEPT", 0)),
    "EU_INTERCEPT": float(os.environ.get("EU_INTERCEPT", 0)),
}

# LCR (Load-to-Carrier Ratio) Coefficients
# Used in polynomial formula: intercept + slope*LCR + poly*LCR^2
COUNTRY_COEFFICIENTS_LCR = {
    "DE_SLOPE_LCR": float(os.environ.get("DE_SLOPE_LCR", 0)),
    "ES_SLOPE_LCR": float(os.environ.get("ES_SLOPE_LCR", 0)),
    "FR_SLOPE_LCR": float(os.environ.get("FR_SLOPE_LCR", 0)),
    "IT_SLOPE_LCR": float(os.environ.get("IT_SLOPE_LCR", 0)),
    "UK_SLOPE_LCR": float(os.environ.get("UK_SLOPE_LCR", 0)),
    "EU_SLOPE_LCR": float(os.environ.get("EU_SLOPE_LCR", 0)),
    "DE_POLY_LCR": float(os.environ.get("DE_POLY_LCR", 0)),
    "ES_POLY_LCR": float(os.environ.get("ES_POLY_LCR", 0)),
    "FR_POLY_LCR": float(os.environ.get("FR_POLY_LCR", 0)),
    "IT_POLY_LCR": float(os.environ.get("IT_POLY_LCR", 0)),
    "UK_POLY_LCR": float(os.environ.get("UK_POLY_LCR", 0)),
    "EU_POLY_LCR": float(os.environ.get("EU_POLY_LCR", 0)),
    "DE_INTERCEPT_LCR": float(os.environ.get("DE_INTERCEPT_LCR", 0)),
    "ES_INTERCEPT_LCR": float(os.environ.get("ES_INTERCEPT_LCR", 0)),
    "FR_INTERCEPT_LCR": float(os.environ.get("FR_INTERCEPT_LCR", 0)),
    "IT_INTERCEPT_LCR": float(os.environ.get("IT_INTERCEPT_LCR", 0)),
    "UK_INTERCEPT_LCR": float(os.environ.get("UK_INTERCEPT_LCR", 0)),
    "EU_INTERCEPT_LCR": float(os.environ.get("EU_INTERCEPT_LCR", 0)),
}


def get_tech_savings_rate(year: int, week: int) -> float:
    """
    Get tech savings rate for a specific year and week.

    Dynamically reads environment variables matching the pattern:
    TECH_SAVINGS_RATE_{YEAR}_W{START}_W{END}

    No code changes needed when adding new time periods - just add new env variables.

    Args:
        year: The year to check (e.g., 2025, 2026)
        week: The week number to check (1-52)

    Returns:
        The tech savings rate as a decimal (negative = savings).
        Returns 0 if no matching configuration is found.

    Example:
        >>> get_tech_savings_rate(2025, 30)  # If TECH_SAVINGS_RATE_2025_W27_W46=-0.0571 is set
        -0.0571
    """
    pattern = re.compile(r"^TECH_SAVINGS_RATE_(\d{4})_W(\d+)_W(\d+)$")

    for env_var, value in os.environ.items():
        match = pattern.match(env_var)
        if match:
            env_year = int(match.group(1))
            week_start = int(match.group(2))
            week_end = int(match.group(3))

            if env_year == year and week_start <= week <= week_end:
                try:
                    rate = float(value)
                    return rate
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: '{value}'. Skipping.")
                    continue

    logger.debug(f"No tech savings rate configured for year {year}, week {week}. Returning 0")
    return 0
