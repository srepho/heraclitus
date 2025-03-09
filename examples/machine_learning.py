"""
Example demonstrating machine learning capabilities in Heraclitus.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

from heraclitus.data import EventLog
from heraclitus.ml import (
    FeatureExtractor,
    create_target_variable,
    OutcomePredictor,
    DurationPredictor,
    ProcessAnomalyDetector,
    detect_variant_anomalies
)


def generate_sample_data(num_cases=500, include_anomalies=True):
    """Generate sample process data with normal and anomalous cases."""
    np.random.seed(42)  # For reproducibility
    
    # Define normal process flow
    normal_flow = [
        ("Register", 10, 30),       # Activity, min duration, max duration (minutes)
        ("Verify", 20, 60),
        ("Review", 30, 90),
        ("Decide", 15, 45),
        ("Process", 45, 120),
        ("Complete", 10, 30)
    ]
    
    # Define alternative normal flows
    alt_flows = [
        # Express flow
        [
            ("Register", 10, 30),
            ("Verify", 20, 60),
            ("Decide", 15, 45),  # No Review step
            ("Process", 45, 120),
            ("Complete", 10, 30)
        ],
        # Rejection flow
        [
            ("Register", 10, 30),
            ("Verify", 20, 60),
            ("Review", 30, 90),
            ("Decide", 15, 45),
            ("Reject", 15, 30)   # Ends with Reject instead of Process->Complete
        ]
    ]
    
    # Define anomalous flows
    anomalous_flows = [
        # Flow with unusual order
        [
            ("Register", 10, 30),
            ("Review", 30, 90),    # Review before Verify (unusual)
            ("Verify", 20, 60),
            ("Decide", 15, 45),
            ("Process", 45, 120),
            ("Complete", 10, 30)
        ],
        # Flow with unusual timing
        [
            ("Register", 10, 30),
            ("Verify", 80, 180),   # Unusually long verification
            ("Review", 30, 90),
            ("Decide", 15, 45),
            ("Process", 45, 120),
            ("Complete", 10, 30)
        ],
        # Flow with repeated activities
        [
            ("Register", 10, 30),
            ("Verify", 20, 60),
            ("Review", 30, 90),
            ("Decide", 15, 45),
            ("Process", 45, 120),
            ("Review", 30, 90),    # Unusual repetition of Review
            ("Decide", 15, 45),    # Second decision
            ("Process", 45, 120),
            ("Complete", 10, 30)
        ]
    ]
    
    # Define possible resources
    resources = {
        "Register": ["Alice", "Bob", "Charlie"],
        "Verify": ["Diana", "Eve", "Frank"],
        "Review": ["Grace", "Heidi", "Ivan"],
        "Decide": ["Jack", "Kelly", "Liam"],
        "Process": ["Mike", "Nancy", "Oscar"],
        "Reject": ["Jack", "Kelly", "Liam"],  # Same as Decide
        "Complete": ["Patty", "Quinn", "Robert"]
    }
    
    # Generate data
    case_ids = []
    activities = []
    timestamps = []
    resources_list = []
    durations = []
    costs = []
    priorities = []
    
    base_time = datetime(2023, 1, 1, 8, 0)
    
    for i in range(1, num_cases + 1):
        case_id = f"case-{i:04d}"
        
        # Determine case type
        if include_anomalies and i > num_cases * 0.9:  # Last 10% are anomalies
            flow = np.random.choice(anomalous_flows)
            priority = np.random.choice(["low", "medium", "high", "critical"])
            outcome_class = 0  # Failed
        else:
            # 70% normal flow, 30% alternative flows
            if np.random.random() < 0.7:
                flow = normal_flow
            else:
                flow = np.random.choice(alt_flows)
            
            priority = np.random.choice(["low", "medium", "high"], p=[0.2, 0.6, 0.2])
            outcome_class = 1  # Successful
        
        # Process the case
        current_time = base_time + timedelta(days=i//20, hours=i%20)  # Distribute start times
        
        for activity, min_duration, max_duration in flow:
            case_ids.append(case_id)
            activities.append(activity)
            
            # Add timestamp
            timestamps.append(current_time)
            
            # Select resource
            resource = np.random.choice(resources[activity])
            resources_list.append(resource)
            
            # Add priority
            priorities.append(priority)
            
            # Calculate activity duration
            # Adjust duration based on priority
            if priority == "critical":
                factor = 0.7  # Critical cases are faster
            elif priority == "high":
                factor = 0.9
            elif priority == "medium":
                factor = 1.0
            else:  # low
                factor = 1.2  # Low priority cases are slower
            
            activity_duration = np.random.randint(
                int(min_duration * factor),
                int(max_duration * factor) + 1
            )
            durations.append(activity_duration)
            
            # Add cost based on duration and resource
            # More experienced resources (later in the list) are more efficient
            resource_efficiency = 1.0 - (resources[activity].index(resource) * 0.1)
            activity_cost = activity_duration * 2 * resource_efficiency
            costs.append(activity_cost)
            
            # Advance time
            current_time += timedelta(minutes=activity_duration)
    
    # Create DataFrame
    df = pd.DataFrame({
        "case_id": case_ids,
        "activity": activities,
        "timestamp": timestamps,
        "resource": resources_list,
        "duration_minutes": durations,
        "cost": costs,
        "priority": priorities
    })
    
    # Add some additional attributes
    case_attributes = {}
    
    for case_id in set(case_ids):
        # Determine if this is a high-value customer
        high_value = np.random.random() < 0.3
        
        # Determine customer segment
        segment = np.random.choice(["retail", "corporate", "vip"], p=[0.7, 0.2, 0.1])
        
        # Determine channel
        channel = np.random.choice(["web", "mobile", "branch", "phone"], p=[0.4, 0.3, 0.2, 0.1])
        
        case_attributes[case_id] = {
            "high_value_customer": high_value,
            "customer_segment": segment,
            "channel": channel,
            "outcome_class": outcome_class if case_id in df[df["activity"].isin(["Complete", "Reject"])]["case_id"].values else None
        }
    
    # Add case attributes to DataFrame
    df["high_value_customer"] = df["case_id"].map(lambda x: case_attributes[x]["high_value_customer"])
    df["customer_segment"] = df["case_id"].map(lambda x: case_attributes[x]["customer_segment"])
    df["channel"] = df["case_id"].map(lambda x: case_attributes[x]["channel"])
    
    return df, case_attributes


def example_outcome_prediction(event_log, case_attributes):
    """Demonstrate outcome prediction."""
    print("\n--- Outcome Prediction Example ---")
    
    # Create outcome predictor
    outcome_predictor = OutcomePredictor()
    
    # Define case attributes to include
    case_attrs = ["high_value_customer", "customer_segment", "channel"]
    
    # Train the model
    model_info = outcome_predictor.train(
        event_log,
        model_type="random_forest",
        case_attributes=case_attrs,
        include_time_features=True,
        include_flow_features=True,
        include_resource_features=True,
        test_size=0.3,
        random_state=42
    )
    
    # Print model information
    print(f"Model type: {model_info['model_type']}")
    print(f"Target type: {model_info['target_type']}")
    print(f"Feature count: {model_info['feature_count']}")
    print(f"Training samples: {model_info['training_samples']}")
    print(f"Test samples: {model_info['test_samples']}")
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for metric, value in model_info['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Print top feature importances
    if model_info.get('feature_importances'):
        print("\nTop 10 Feature Importances:")
        for i, (feature, importance) in enumerate(list(model_info['feature_importances'].items())[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Make predictions on test data
    print("\nPredicting outcomes for new cases...")
    
    # Extract features from the event log for demonstration
    feature_extractor = FeatureExtractor(
        event_log,
        case_attributes=case_attrs,
        include_time_features=True,
        include_flow_features=True,
        include_resource_features=True
    )
    features_df = feature_extractor.extract_case_features()
    
    # Get a sample of cases
    sample_features = features_df.sample(5, random_state=42)
    
    # Make predictions
    predictions = outcome_predictor.predict(sample_features)
    
    # Print predictions
    print("\nSample Predictions:")
    for i, (_, row) in enumerate(predictions.iterrows()):
        print(f"  Case {row['case_id']}: Predicted outcome = {row['predicted_outcome']}")
        if 'probability' in row:
            print(f"    Probability: {row['probability']:.4f}")
    
    # Save and load the model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    outcome_predictor.save(f"{model_dir}/outcome_model")
    print(f"\nModel saved to {model_dir}/outcome_model")
    
    # Load the model
    loaded_predictor = OutcomePredictor()
    loaded_predictor.load(f"{model_dir}/outcome_model")
    print("Model loaded successfully")


def example_duration_prediction(event_log, case_attributes):
    """Demonstrate duration prediction."""
    print("\n--- Duration Prediction Example ---")
    
    # Create duration predictor
    duration_predictor = DurationPredictor()
    
    # Define case attributes to include
    case_attrs = ["high_value_customer", "customer_segment", "channel"]
    
    # Train the model (regression)
    model_info = duration_predictor.train(
        event_log,
        model_type="random_forest",
        prediction_type="regression",
        duration_unit="minutes",
        case_attributes=case_attrs,
        include_time_features=True,
        include_flow_features=True,
        include_resource_features=True,
        test_size=0.3,
        random_state=42,
        normalize_target=True
    )
    
    # Print model information
    print(f"Model type: {model_info['model_type']}")
    print(f"Prediction type: {model_info['prediction_type']}")
    print(f"Duration unit: {model_info['duration_unit']}")
    print(f"Feature count: {model_info['feature_count']}")
    print(f"Training samples: {model_info['training_samples']}")
    print(f"Test samples: {model_info['test_samples']}")
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for metric, value in model_info['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Print top feature importances
    if model_info.get('feature_importances'):
        print("\nTop 10 Feature Importances:")
        for i, (feature, importance) in enumerate(list(model_info['feature_importances'].items())[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Make predictions on test data
    print("\nPredicting durations for new cases...")
    
    # Extract features from the event log for demonstration
    feature_extractor = FeatureExtractor(
        event_log,
        case_attributes=case_attrs,
        include_time_features=True,
        include_flow_features=True,
        include_resource_features=True
    )
    features_df = feature_extractor.extract_case_features()
    
    # Get a sample of cases
    sample_features = features_df.sample(5, random_state=42)
    
    # Make predictions
    predictions = duration_predictor.predict(sample_features)
    
    # Print predictions
    print("\nSample Predictions:")
    for i, (_, row) in enumerate(predictions.iterrows()):
        print(f"  Case {row['case_id']}: Predicted duration = {row['predicted_duration']:.1f} minutes")
    
    # Save and load the model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    duration_predictor.save(f"{model_dir}/duration_model")
    print(f"\nModel saved to {model_dir}/duration_model")
    
    # Load the model
    loaded_predictor = DurationPredictor()
    loaded_predictor.load(f"{model_dir}/duration_model")
    print("Model loaded successfully")
    
    # Train a classification model for duration
    print("\nTraining a classification model for duration:")
    
    # Calculate median duration to use as threshold
    targets = create_target_variable(event_log, target_type="duration", duration_unit="minutes")
    median_duration = targets["duration"].median()
    print(f"Median case duration: {median_duration:.1f} minutes")
    
    # Train classification model
    classification_model_info = duration_predictor.train(
        event_log,
        model_type="random_forest",
        prediction_type="classification",
        duration_unit="minutes",
        duration_threshold=median_duration,  # Classify as above/below median
        case_attributes=case_attrs,
        include_time_features=True,
        include_flow_features=True,
        include_resource_features=True,
        test_size=0.3,
        random_state=42
    )
    
    # Print classification metrics
    print("\nDuration Classification Metrics:")
    for metric, value in classification_model_info['metrics'].items():
        print(f"  {metric}: {value:.4f}")


def example_anomaly_detection(event_log):
    """Demonstrate anomaly detection."""
    print("\n--- Anomaly Detection Example ---")
    
    # Create anomaly detector
    anomaly_detector = ProcessAnomalyDetector()
    
    # Define case attributes to include
    case_attrs = ["high_value_customer", "customer_segment", "channel"]
    
    # Train the model
    model_info = anomaly_detector.train(
        event_log,
        method="isolation_forest",
        case_attributes=case_attrs,
        include_time_features=True,
        include_flow_features=True,
        include_resource_features=True,
        contamination=0.1,  # Expect 10% anomalies
        random_state=42
    )
    
    # Print model information
    print(f"Method: {model_info['method']}")
    print(f"Total cases: {model_info['total_cases']}")
    print(f"Detected anomalies: {model_info['anomaly_count']} ({model_info['anomaly_percentage']:.1f}%)")
    print(f"Anomaly threshold: {model_info['anomaly_threshold']:.4f}")
    print(f"Feature count: {model_info['feature_count']}")
    
    # Extract features for detection
    feature_extractor = FeatureExtractor(
        event_log,
        case_attributes=case_attrs,
        include_time_features=True,
        include_flow_features=True,
        include_resource_features=True
    )
    features_df = feature_extractor.extract_case_features()
    
    # Detect anomalies
    anomaly_results = anomaly_detector.detect_anomalies(features_df)
    
    # Print top anomalies
    print("\nTop 10 Anomalous Cases:")
    for i, (_, row) in enumerate(anomaly_results.head(10).iterrows()):
        print(f"  {i+1}. Case {row['case_id']}: Anomaly score = {row['anomaly_score']:.4f}")
    
    # Create visualization
    print("\nCreating anomaly visualization...")
    fig = anomaly_detector.visualize_anomalies(anomaly_results, features_df)
    
    # Save the figure
    fig.savefig("anomaly_detection.png")
    print("Saved visualization to 'anomaly_detection.png'")
    
    # Detect variant anomalies
    print("\nDetecting variant anomalies...")
    variant_results, case_variants = detect_variant_anomalies(
        event_log,
        min_support=0.05,  # Variants with less than 5% support are considered anomalous
        max_variants=5     # At most 5 variants are considered normal
    )
    
    # Print variant information
    print(f"Detected {len(variant_results)} distinct process variants")
    print(f"Normal variants: {sum(~variant_results['is_anomaly'])} ({sum(~case_variants['is_anomaly']) / len(case_variants) * 100:.1f}% of cases)")
    print(f"Anomalous variants: {sum(variant_results['is_anomaly'])} ({sum(case_variants['is_anomaly']) / len(case_variants) * 100:.1f}% of cases)")
    
    # Print top normal variants
    print("\nTop 3 Normal Variants:")
    for i, (_, row) in enumerate(variant_results[~variant_results['is_anomaly']].head(3).iterrows()):
        print(f"  {i+1}. Support: {row['support']:.2f}, Count: {row['count']}")
        print(f"     Variant: {row['variant']}")
    
    # Print top anomalous variants
    print("\nTop 3 Anomalous Variants:")
    for i, (_, row) in enumerate(variant_results[variant_results['is_anomaly']].sort_values(by="count", ascending=False).head(3).iterrows()):
        print(f"  {i+1}. Support: {row['support']:.2f}, Count: {row['count']}")
        print(f"     Variant: {row['variant']}")


def main():
    """Run the machine learning examples."""
    print("Generating sample data...")
    df, case_attributes = generate_sample_data(num_cases=500, include_anomalies=True)
    
    print(f"Generated dataset with {len(df)} events across {len(case_attributes)} cases")
    print(f"Activities: {df['activity'].nunique()} unique activities")
    print(f"Resources: {df['resource'].nunique()} unique resources")
    
    # Create EventLog
    event_log = EventLog(df)
    
    # Run examples
    example_outcome_prediction(event_log, case_attributes)
    example_duration_prediction(event_log, case_attributes)
    example_anomaly_detection(event_log)
    
    print("\nAll examples completed successfully.")


if __name__ == "__main__":
    main()