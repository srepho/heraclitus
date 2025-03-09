"""
Main interface module for the heraclitus package.
"""
from heraclitus.data import EventLog
from heraclitus.visualization import visualize_process_map, plot_activity_frequency
from heraclitus.metrics import (
    calculate_cycle_time,
    calculate_waiting_time,
    calculate_processing_time,
)
from heraclitus.statistics import (
    compare_cycle_times,
    bottleneck_analysis,
    fit_distribution,
)

# Re-export key classes and functions
__all__ = [
    "EventLog",
    "visualize_process_map",
    "plot_activity_frequency",
    "calculate_cycle_time",
    "calculate_waiting_time",
    "calculate_processing_time",
    "compare_cycle_times",
    "bottleneck_analysis",
    "fit_distribution",
]