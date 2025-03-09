"""
Main interface module for the heraclitus package.
"""
from heraclitus.data import EventLog, DuckDBConnector, eventlog_to_duckdb
from heraclitus.visualization import (
    visualize_process_map,
    plot_activity_frequency,
    create_interactive_process_map,
    create_cycle_time_distribution,
    create_activity_timeline,
    create_bottleneck_dashboard,
)
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
from heraclitus.ml import (
    # Feature Engineering
    FeatureExtractor,
    create_target_variable,
    combine_features_and_target,
    
    # Prediction
    OutcomePredictor,
    DurationPredictor,
    
    # Anomaly Detection
    ProcessAnomalyDetector,
    detect_variant_anomalies,
)

# Re-export key classes and functions
__all__ = [
    # Data
    "EventLog",
    "DuckDBConnector",
    "eventlog_to_duckdb",
    
    # Visualization
    "visualize_process_map",
    "plot_activity_frequency",
    "create_interactive_process_map",
    "create_cycle_time_distribution",
    "create_activity_timeline",
    "create_bottleneck_dashboard",
    
    # Metrics
    "calculate_cycle_time",
    "calculate_waiting_time",
    "calculate_processing_time",
    
    # Statistics
    "compare_cycle_times",
    "bottleneck_analysis",
    "fit_distribution",
    
    # Machine Learning - Feature Engineering
    "FeatureExtractor",
    "create_target_variable",
    "combine_features_and_target",
    
    # Machine Learning - Prediction
    "OutcomePredictor", 
    "DurationPredictor",
    
    # Machine Learning - Anomaly Detection
    "ProcessAnomalyDetector",
    "detect_variant_anomalies",
]