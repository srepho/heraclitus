"""Machine learning module for predictive analytics."""

from heraclitus.ml.feature_engineering import (
    FeatureExtractor,
    create_target_variable,
    combine_features_and_target,
    encode_categorical_features,
    scale_features,
)

from heraclitus.ml.prediction import (
    OutcomePredictor,
    DurationPredictor,
)

from heraclitus.ml.anomaly_detection import (
    ProcessAnomalyDetector,
    detect_variant_anomalies,
)

__all__ = [
    # Feature Engineering
    "FeatureExtractor",
    "create_target_variable",
    "combine_features_and_target",
    "encode_categorical_features",
    "scale_features",
    
    # Prediction
    "OutcomePredictor",
    "DurationPredictor",
    
    # Anomaly Detection
    "ProcessAnomalyDetector",
    "detect_variant_anomalies",
]