"""
Tests for machine learning module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
from pathlib import Path

from heraclitus.data import EventLog
from heraclitus.ml import (
    FeatureExtractor,
    create_target_variable,
    combine_features_and_target,
    encode_categorical_features,
    scale_features,
    OutcomePredictor,
    DurationPredictor,
    ProcessAnomalyDetector,
    detect_variant_anomalies,
)


@pytest.fixture
def sample_event_log():
    """Create a sample EventLog for testing."""
    # Generate sample data
    np.random.seed(42)  # For reproducibility
    
    case_ids = []
    activities = []
    timestamps = []
    resources = []
    priorities = []
    segments = []
    
    # Define process flow
    activities_flow = ["start", "process", "review", "decide", "end"]
    
    # Generate 50 cases
    for i in range(1, 51):
        case_id = f"case{i}"
        priority = np.random.choice(["low", "medium", "high"])
        segment = np.random.choice(["A", "B", "C"])
        
        # Start with a random time
        current_time = datetime(2023, 1, 1) + timedelta(hours=i*5)
        
        # Generate events for this case
        for activity in activities_flow:
            # Skip some activities randomly
            if activity == "review" and np.random.random() < 0.3:
                continue
                
            case_ids.append(case_id)
            activities.append(activity)
            timestamps.append(current_time)
            resources.append(f"user{np.random.randint(1, 6)}")
            priorities.append(priority)
            segments.append(segment)
            
            # Advance time
            activity_duration = np.random.randint(10, 120)  # minutes
            current_time += timedelta(minutes=activity_duration)
    
    # Create DataFrame
    df = pd.DataFrame({
        "case_id": case_ids,
        "activity": activities,
        "timestamp": timestamps,
        "resource": resources,
        "priority": priorities,
        "segment": segments
    })
    
    return EventLog(df)


def test_feature_extractor(sample_event_log):
    """Test FeatureExtractor functionality."""
    # Create feature extractor
    feature_extractor = FeatureExtractor(
        sample_event_log,
        case_attributes=["priority", "segment"],
        include_time_features=True,
        include_flow_features=True,
        include_resource_features=True
    )
    
    # Extract features
    features_df = feature_extractor.extract_case_features()
    
    # Check basic properties
    assert len(features_df) == sample_event_log.case_count()
    assert "case_id" in features_df.columns
    
    # Check feature categories
    assert any(col.startswith("attr_") for col in features_df.columns)
    assert "case_duration_minutes" in features_df.columns
    assert any(col.startswith("activity_") for col in features_df.columns)
    assert "unique_resources_count" in features_df.columns


def test_create_target_variable(sample_event_log):
    """Test create_target_variable functionality."""
    # Test outcome target
    outcome_targets = create_target_variable(
        sample_event_log,
        target_type="outcome"
    )
    
    assert len(outcome_targets) == sample_event_log.case_count()
    assert "outcome" in outcome_targets.columns
    
    # Test duration target
    duration_targets = create_target_variable(
        sample_event_log,
        target_type="duration",
        duration_unit="minutes"
    )
    
    assert len(duration_targets) == sample_event_log.case_count()
    assert "duration" in duration_targets.columns
    assert all(duration_targets["duration"] >= 0)
    
    # Test binary duration target
    binary_targets = create_target_variable(
        sample_event_log,
        target_type="binary_duration",
        duration_unit="minutes",
        threshold=120
    )
    
    assert len(binary_targets) == sample_event_log.case_count()
    assert "duration_class" in binary_targets.columns
    assert set(binary_targets["duration_class"].unique()).issubset({0, 1})


def test_combine_features_and_target(sample_event_log):
    """Test combine_features_and_target functionality."""
    # Create features and targets
    feature_extractor = FeatureExtractor(sample_event_log)
    features_df = feature_extractor.extract_case_features()
    
    targets_df = create_target_variable(
        sample_event_log,
        target_type="outcome"
    )
    
    # Combine
    combined_df = combine_features_and_target(
        features_df,
        targets_df,
        case_id_column="case_id"
    )
    
    # Check result
    assert len(combined_df) == len(features_df)
    assert "outcome" in combined_df.columns
    assert all(col in combined_df.columns for col in features_df.columns)


def test_encode_categorical_features():
    """Test encode_categorical_features functionality."""
    # Create sample DataFrame
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "category": ["A", "B", "A", "C", "B"],
        "value": [10, 20, 30, 40, 50]
    })
    
    # Test one-hot encoding
    onehot_df = encode_categorical_features(
        df,
        categorical_columns=["category"],
        encoding_method="onehot"
    )
    
    assert "category" not in onehot_df.columns
    assert "category_A" in onehot_df.columns
    assert "category_B" in onehot_df.columns
    assert "category_C" in onehot_df.columns
    
    # Test label encoding
    label_df = encode_categorical_features(
        df,
        categorical_columns=["category"],
        encoding_method="label"
    )
    
    assert "category" in label_df.columns
    assert label_df["category"].dtype != "object"


def test_scale_features():
    """Test scale_features functionality."""
    # Create sample DataFrame
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value1": [10, 20, 30, 40, 50],
        "value2": [100, 200, 300, 400, 500]
    })
    
    # Test standard scaling
    scaled_df, scaler = scale_features(
        df,
        numeric_columns=["value1", "value2"],
        scaler_type="standard"
    )
    
    assert "value1" in scaled_df.columns
    assert scaled_df["value1"].mean() < 0.01  # Close to zero mean
    assert 0.9 < scaled_df["value1"].std() < 1.1  # Close to unit variance
    
    # Test minmax scaling
    scaled_df, scaler = scale_features(
        df,
        numeric_columns=["value1", "value2"],
        scaler_type="minmax"
    )
    
    assert scaled_df["value1"].min() >= 0
    assert scaled_df["value1"].max() <= 1


def test_outcome_predictor(sample_event_log):
    """Test OutcomePredictor functionality."""
    # Skip if scikit-learn is not installed
    try:
        import sklearn
    except ImportError:
        pytest.skip("scikit-learn not installed")
    
    # Create predictor
    predictor = OutcomePredictor()
    
    # Train model
    model_info = predictor.train(
        sample_event_log,
        model_type="random_forest",
        test_size=0.3,
        random_state=42
    )
    
    # Check model info
    assert "metrics" in model_info
    assert "accuracy" in model_info["metrics"]
    assert predictor.model is not None
    
    # Test prediction
    feature_extractor = FeatureExtractor(sample_event_log)
    features_df = feature_extractor.extract_case_features()
    
    predictions = predictor.predict(features_df)
    
    assert len(predictions) == len(features_df)
    assert "predicted_outcome" in predictions.columns
    
    # Test save and load
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "outcome_model"
        
        # Save model
        predictor.save(model_path)
        
        # Check files
        assert (model_path.with_suffix(".joblib")).exists()
        assert (model_path.with_suffix(".json")).exists()
        
        # Load model
        loaded_predictor = OutcomePredictor()
        loaded_predictor.load(model_path)
        
        # Check loaded model
        assert loaded_predictor.model is not None
        assert len(loaded_predictor.feature_columns) > 0
        
        # Test prediction with loaded model
        loaded_predictions = loaded_predictor.predict(features_df)
        assert len(loaded_predictions) == len(features_df)


def test_duration_predictor(sample_event_log):
    """Test DurationPredictor functionality."""
    # Skip if scikit-learn is not installed
    try:
        import sklearn
    except ImportError:
        pytest.skip("scikit-learn not installed")
    
    # Create predictor
    predictor = DurationPredictor()
    
    # Train regression model
    model_info = predictor.train(
        sample_event_log,
        model_type="random_forest",
        prediction_type="regression",
        duration_unit="minutes",
        test_size=0.3,
        random_state=42
    )
    
    # Check model info
    assert "metrics" in model_info
    assert "mean_squared_error" in model_info["metrics"]
    assert predictor.model is not None
    assert not predictor.is_classifier
    
    # Test prediction
    feature_extractor = FeatureExtractor(sample_event_log)
    features_df = feature_extractor.extract_case_features()
    
    predictions = predictor.predict(features_df)
    
    assert len(predictions) == len(features_df)
    assert "predicted_duration" in predictions.columns
    
    # Train classification model
    model_info = predictor.train(
        sample_event_log,
        model_type="random_forest",
        prediction_type="classification",
        duration_unit="minutes",
        duration_threshold=60,
        test_size=0.3,
        random_state=42
    )
    
    # Check model info
    assert "metrics" in model_info
    assert "accuracy" in model_info["metrics"]
    assert predictor.model is not None
    assert predictor.is_classifier
    
    # Test prediction
    predictions = predictor.predict(features_df)
    
    assert len(predictions) == len(features_df)
    assert "predicted_class" in predictions.columns


def test_process_anomaly_detector(sample_event_log):
    """Test ProcessAnomalyDetector functionality."""
    # Skip if scikit-learn is not installed
    try:
        import sklearn
    except ImportError:
        pytest.skip("scikit-learn not installed")
    
    # Create detector
    detector = ProcessAnomalyDetector()
    
    # Train model
    model_info = detector.train(
        sample_event_log,
        method="isolation_forest",
        contamination=0.1,
        random_state=42
    )
    
    # Check model info
    assert "anomaly_count" in model_info
    assert "anomaly_threshold" in model_info
    assert detector.model is not None
    
    # Test detection
    feature_extractor = FeatureExtractor(sample_event_log)
    features_df = feature_extractor.extract_case_features()
    
    anomaly_results = detector.detect_anomalies(features_df)
    
    assert len(anomaly_results) == len(features_df)
    assert "anomaly_score" in anomaly_results.columns
    assert "is_anomaly" in anomaly_results.columns
    
    # Check that some anomalies were detected
    assert anomaly_results["is_anomaly"].sum() > 0


def test_detect_variant_anomalies(sample_event_log):
    """Test detect_variant_anomalies functionality."""
    # Run detection
    variant_results, case_variants = detect_variant_anomalies(
        sample_event_log,
        min_support=0.05,
        max_variants=3
    )
    
    # Check variant results
    assert "variant" in variant_results.columns
    assert "count" in variant_results.columns
    assert "support" in variant_results.columns
    assert "is_anomaly" in variant_results.columns
    
    # Check case variants
    assert "case_id" in case_variants.columns
    assert "variant" in case_variants.columns
    assert "is_anomaly" in case_variants.columns
    
    # Check total cases
    assert len(case_variants) == sample_event_log.case_count()
    
    # Check variant counts
    normal_variants = variant_results[~variant_results["is_anomaly"]]
    assert len(normal_variants) <= 3  # max_variants parameter