"""
Bridges Module

Contains functions for creating and calculating bridge analysis structures.
Supports YoY (Year-over-Year), WoW (Week-over-Week), and MTD (Month-to-Date) comparisons.
"""

from .bridge_builder import (
    create_bridge_structure,
    create_bridge_structure_for_totals,
)
from .bridge_metrics import calculate_detailed_bridge_metrics
from .yoy_bridge import calculate_yoy_bridge_metrics
from .wow_bridge import calculate_wow_bridge_metrics
from .mtd_bridge import calculate_mtd_bridge_metrics
from .eu_aggregation import calculate_eu_aggregated_metrics
from .total_aggregation import calculate_total_aggregated_metrics
from .impact_adjuster import adjust_carrier_demand_impacts

__all__ = [
    "create_bridge_structure",
    "create_bridge_structure_for_totals",
    "calculate_detailed_bridge_metrics",
    "calculate_yoy_bridge_metrics",
    "calculate_wow_bridge_metrics",
    "calculate_mtd_bridge_metrics",
    "calculate_eu_aggregated_metrics",
    "calculate_total_aggregated_metrics",
    "adjust_carrier_demand_impacts",
]
