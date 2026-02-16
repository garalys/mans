"""
OP2 Benchmarking Module

Contains functions for creating and calculating OP2 benchmark bridges.
OP2 represents operational planning targets against which actual performance is measured.
"""

from .op2_data_extractor import (
    extract_op2_weekly_base_cpkm,
    extract_op2_weekly_base_by_business,
    extract_op2_monthly_base_cpkm,
)
from .op2_normalizer import (
    compute_op2_normalized_cpkm_weekly,
    compute_op2_normalized_cpkm_monthly,
)
from .op2_impacts import (
    calculate_op2_carrier_demand_impacts,
    calculate_op2_tech_impact,
    calculate_op2_market_rate_impact,
)
from .op2_weekly_bridge import (
    create_op2_weekly_bridge,
    create_op2_weekly_country_business_bridge,
)
from .op2_monthly_bridge import create_op2_monthly_bridge
from .op2_impact_adjuster import adjust_op2_carrier_demand_impacts
from .op2_helpers import get_set_impact_for_op2
from .op2_eu_aggregation import (
    create_op2_eu_weekly_bridge,
    create_op2_eu_weekly_business_bridge,
)

__all__ = [
    "extract_op2_weekly_base_cpkm",
    "extract_op2_weekly_base_by_business",
    "extract_op2_monthly_base_cpkm",
    "compute_op2_normalized_cpkm_weekly",
    "compute_op2_normalized_cpkm_monthly",
    "calculate_op2_carrier_demand_impacts",
    "calculate_op2_tech_impact",
    "calculate_op2_market_rate_impact",
    "create_op2_weekly_bridge",
    "create_op2_weekly_country_business_bridge",
    "create_op2_monthly_bridge",
    "adjust_op2_carrier_demand_impacts",
    "get_set_impact_for_op2",
    "create_op2_eu_weekly_bridge",
    "create_op2_eu_weekly_business_bridge",
]
