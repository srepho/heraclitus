# Heraclitus Machine Learning Guide

This guide provides a comprehensive overview of the machine learning capabilities in Heraclitus. The ML module offers tools for predictive analytics and anomaly detection in process mining.

## Overview

Heraclitus provides three main areas of machine learning functionality:

1. **Feature Engineering**: Transform process data into features suitable for ML models
2. **Prediction**: Build models to predict process outcomes and durations
3. **Anomaly Detection**: Identify unusual cases or process variants

## Installation

To use the ML features, install Heraclitus with the ML extras:

```bash
pip install -e ".[ml]"
```

This will install the required dependencies, including scikit-learn and other ML libraries.

## Feature Engineering

The `FeatureExtractor` class provides a robust framework for transforming process data into ML-ready features.

### Key Capabilities

- Generate features based on case attributes
- Create temporal features (duration, timing, etc.)
- Extract process flow features (activity sequences, transitions)
- Create resource-based features (handovers, resource utilization)

### Basic Usage

```python
from heraclitus.data import EventLog
from heraclitus.ml import FeatureExtractor

# Create feature extractor
feature_extractor = FeatureExtractor(
    event_log,
    case_attributes=["customer_segment", "channel"],
    include_time_features=True,
    include_flow_features=True,
    include_resource_features=True
)

# Extract features
features_df = feature_extractor.extract_case_features()
```

### Target Variables

In addition to features, you need target variables for supervised learning:

```python
from heraclitus.ml import create_target_variable

# Create target for outcome prediction
outcome_targets = create_target_variable(
    event_log,
    target_type="outcome",
    outcome_activity="Complete"  # Optional: specific activity to predict
)

# Create target for duration prediction
duration_targets = create_target_variable(
    event_log,
    target_type="duration",
    duration_unit="minutes"
)

# Create binary duration target with threshold
binary_duration_targets = create_target_variable(
    event_log,
    target_type="binary_duration",
    duration_unit="minutes",
    threshold=120  # Classify cases taking more than 2 hours as 'long'
)
```

## Prediction Models

Heraclitus provides two main predictors:

1. `OutcomePredictor`: Predicts categorical outcomes (e.g., approved/rejected)
2. `DurationPredictor`: Predicts process durations or duration categories

### Outcome Prediction

```python
from heraclitus.ml import OutcomePredictor

# Create and train the predictor
outcome_predictor = OutcomePredictor()
model_info = outcome_predictor.train(
    event_log,
    model_type="random_forest",  # Options: random_forest, logistic_regression, svm, xgboost
    case_attributes=["customer_segment", "channel"],
    include_time_features=True,
    include_flow_features=True,
    include_resource_features=True,
    test_size=0.3,
    random_state=42
)

# Print model information
print(f"Model accuracy: {model_info['metrics']['accuracy']:.4f}")
print(f"Top features: {list(model_info['feature_importances'].keys())[:5]}")

# Make predictions on new data
predictions = outcome_predictor.predict(new_features_df)

# Save and load the model
outcome_predictor.save("models/outcome_model")
loaded_predictor = OutcomePredictor()
loaded_predictor.load("models/outcome_model")
```

### Duration Prediction

```python
from heraclitus.ml import DurationPredictor

# Create and train the predictor (regression)
duration_predictor = DurationPredictor()
model_info = duration_predictor.train(
    event_log,
    model_type="random_forest",  # Options: random_forest, linear_regression, svr, xgboost
    prediction_type="regression",
    duration_unit="minutes",
    case_attributes=["customer_segment", "channel"],
    include_time_features=True,
    include_flow_features=True,
    include_resource_features=True,
    test_size=0.3,
    random_state=42
)

# Print model information
print(f"RMSE: {model_info['metrics']['root_mean_squared_error']:.2f} minutes")
print(f"R-squared: {model_info['metrics']['r2_score']:.4f}")

# Make predictions
predictions = duration_predictor.predict(new_features_df)

# Train a classification model for duration
classification_model_info = duration_predictor.train(
    event_log,
    model_type="random_forest",
    prediction_type="classification",
    duration_unit="minutes",
    duration_threshold=120,  # Classify as above/below 2 hours
    case_attributes=["customer_segment", "channel"]
)
```

## Anomaly Detection

Anomaly detection identifies unusual process behaviors that may require investigation.

### Process Anomaly Detection

```python
from heraclitus.ml import ProcessAnomalyDetector

# Create and train the detector
anomaly_detector = ProcessAnomalyDetector()
model_info = anomaly_detector.train(
    event_log,
    method="isolation_forest",  # Options: isolation_forest, local_outlier_factor, dbscan
    case_attributes=["customer_segment", "channel"],
    include_time_features=True,
    include_flow_features=True,
    include_resource_features=True,
    contamination=0.05,  # Expected proportion of anomalies
    random_state=42
)

# Print model information
print(f"Detected {model_info['anomaly_count']} anomalies ({model_info['anomaly_percentage']:.1f}%)")

# Extract features for detection
feature_extractor = FeatureExtractor(event_log)
features_df = feature_extractor.extract_case_features()

# Detect anomalies
anomaly_results = anomaly_detector.detect_anomalies(features_df)

# Visualize anomalies
fig = anomaly_detector.visualize_anomalies(anomaly_results, features_df)
fig.savefig("anomaly_detection.png")
```

### Variant Anomaly Detection

In addition to feature-based anomaly detection, Heraclitus can identify unusual process variants based on their frequency:

```python
from heraclitus.ml import detect_variant_anomalies

# Detect variant anomalies
variant_results, case_variants = detect_variant_anomalies(
    event_log,
    min_support=0.05,  # Variants with less than 5% support are considered anomalous
    max_variants=5     # At most 5 variants are considered normal
)

# Print variant information
print(f"Detected {len(variant_results)} distinct process variants")
print(f"Normal variants: {sum(~variant_results['is_anomaly'])}")
print(f"Anomalous variants: {sum(variant_results['is_anomaly'])}")

# Get cases with anomalous variants
anomalous_cases = case_variants[case_variants["is_anomaly"]]
```

## Advanced Usage

### Hyperparameter Tuning

All model training functions accept a `model_params` dictionary for customizing model hyperparameters:

```python
# Train with custom hyperparameters
model_info = outcome_predictor.train(
    event_log,
    model_type="random_forest",
    model_params={
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5
    }
)
```

### Custom Feature Engineering

For advanced use cases, you can extract features and create custom transformations:

```python
from heraclitus.ml import encode_categorical_features, scale_features

# Extract features
feature_extractor = FeatureExtractor(event_log)
features_df = feature_extractor.extract_case_features()

# Encode categorical features
categorical_columns = ["customer_segment", "channel"]
encoded_df = encode_categorical_features(
    features_df,
    categorical_columns,
    encoding_method="onehot"
)

# Scale numeric features
numeric_columns = [col for col in encoded_df.columns
                  if encoded_df[col].dtype in ['int64', 'float64']]
scaled_df, scaler = scale_features(
    encoded_df,
    numeric_columns,
    scaler_type="standard"
)
```

## Integration with Visualization

Heraclitus ML features integrate well with the visualization module:

```python
import plotly.express as px
from heraclitus.ml import ProcessAnomalyDetector

# Train anomaly detector
anomaly_detector = ProcessAnomalyDetector()
anomaly_detector.train(event_log)

# Detect anomalies
features_df = FeatureExtractor(event_log).extract_case_features()
anomaly_results = anomaly_detector.detect_anomalies(features_df)

# Merge with original event log
merged_df = pd.merge(
    event_log.to_dataframe(),
    anomaly_results[["case_id", "is_anomaly"]],
    on="case_id"
)

# Create interactive visualization
fig = px.scatter(
    merged_df,
    x="timestamp",
    y="cost",
    color="is_anomaly",
    hover_data=["case_id", "activity", "resource"],
    title="Process Events with Anomalies Highlighted"
)
fig.write_html("anomalies.html")
```

## Best Practices

1. **Data Preparation**: Clean and preprocess your data before ML modeling
2. **Feature Selection**: Use `model_info['feature_importances']` to identify key features
3. **Class Imbalance**: For imbalanced outcomes, consider adjusting model parameters or sampling techniques
4. **Model Evaluation**: Always check multiple metrics, not just accuracy
5. **Anomaly Investigation**: Anomalies flagged by the system require human investigation to determine root causes

## Performance Considerations

- Feature extraction can be memory-intensive for large event logs
- Consider using DuckDB integration for large datasets:
  ```python
  from heraclitus.data import DuckDBConnector
  
  db = DuckDBConnector("process_data.duckdb")
  db.load_csv("large_event_log.csv", table_name="events")
  
  # Query to extract only needed data
  filtered_log = db.query_to_eventlog(
      "SELECT * FROM events WHERE timestamp >= '2023-01-01'"
  )
  
  # Now extract features from the filtered log
  feature_extractor = FeatureExtractor(filtered_log)
  features_df = feature_extractor.extract_case_features()
  ```

## Summary

The machine learning module in Heraclitus provides powerful tools for predictive analytics and anomaly detection in process mining. By leveraging these capabilities, analysts can gain insights into future process behavior, identify bottlenecks, and detect unusual cases that may require attention.