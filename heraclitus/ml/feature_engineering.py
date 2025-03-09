"""
Feature engineering module for creating ML features from process data.
"""
from typing import List, Dict, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from heraclitus.data import EventLog


class FeatureExtractor:
    """
    Feature extraction utility for process mining data.
    
    This class provides methods to transform EventLog data into features
    suitable for machine learning models, including:
    - Case attributes
    - Process flow features
    - Temporal features
    - Aggregated metrics
    
    Attributes:
        event_log: The EventLog to extract features from
        case_attributes: List of case-level attributes to include as features
        include_time_features: Whether to include time-based features
        include_flow_features: Whether to include process flow features
        include_resource_features: Whether to include resource-based features
    """
    
    def __init__(
        self,
        event_log: EventLog,
        case_attributes: Optional[List[str]] = None,
        include_time_features: bool = True,
        include_flow_features: bool = True,
        include_resource_features: bool = True,
    ):
        """
        Initialize a FeatureExtractor.
        
        Args:
            event_log: The EventLog to extract features from
            case_attributes: Optional list of case-level attributes to include
            include_time_features: Whether to include time-based features
            include_flow_features: Whether to include process flow features
            include_resource_features: Whether to include resource-based features
        """
        self.event_log = event_log
        self.case_attributes = case_attributes or []
        self.include_time_features = include_time_features
        self.include_flow_features = include_flow_features
        self.include_resource_features = include_resource_features
        
        # Get all available attributes
        self.available_attributes = event_log.get_attributes()
        
        # Get all activities for one-hot encoding
        df = event_log.to_dataframe()
        self.unique_activities = df[event_log.activity_column].unique().tolist()
        
        # Get resources if needed
        if self.include_resource_features and "resource" in self.available_attributes:
            self.unique_resources = df["resource"].unique().tolist()
        else:
            self.unique_resources = []
    
    def extract_case_features(self) -> pd.DataFrame:
        """
        Extract features for each case in the event log.
        
        Returns:
            DataFrame with one row per case and extracted features as columns
        """
        df = self.event_log.to_dataframe()
        case_id_col = self.event_log.case_id_column
        activity_col = self.event_log.activity_column
        timestamp_col = self.event_log.timestamp_column
        
        # Group by case ID to process each case
        case_features = []
        
        for case_id, case_df in df.groupby(case_id_col):
            # Sort by timestamp
            case_df = case_df.sort_values(by=timestamp_col)
            
            # Initialize features dictionary with case ID
            features = {case_id_col: case_id}
            
            # Add case attributes
            for attr in self.case_attributes:
                if attr in case_df.columns:
                    # Use the first non-null value (assuming case attributes are consistent)
                    non_null_values = case_df[attr].dropna()
                    if not non_null_values.empty:
                        features[f"attr_{attr}"] = non_null_values.iloc[0]
                    else:
                        features[f"attr_{attr}"] = None
            
            # Add basic case features
            features["case_event_count"] = len(case_df)
            features["case_unique_activities"] = case_df[activity_col].nunique()
            
            # Extract time features
            if self.include_time_features:
                self._add_time_features(features, case_df)
            
            # Extract flow features
            if self.include_flow_features:
                self._add_flow_features(features, case_df)
            
            # Extract resource features
            if self.include_resource_features and "resource" in self.available_attributes:
                self._add_resource_features(features, case_df)
            
            case_features.append(features)
        
        # Create DataFrame from features
        feature_df = pd.DataFrame(case_features)
        
        return feature_df
    
    def _add_time_features(self, features: Dict[str, Any], case_df: pd.DataFrame) -> None:
        """
        Add time-based features to the features dictionary.
        
        Args:
            features: Dictionary to add features to
            case_df: DataFrame containing events for a single case
        """
        timestamp_col = self.event_log.timestamp_column
        
        # Get start and end times
        start_time = case_df[timestamp_col].min()
        end_time = case_df[timestamp_col].max()
        
        # Calculate case duration in seconds
        duration = (end_time - start_time).total_seconds()
        features["case_duration_seconds"] = duration
        features["case_duration_minutes"] = duration / 60
        features["case_duration_hours"] = duration / 3600
        features["case_duration_days"] = duration / 86400
        
        # Add start time features
        features["start_hour"] = start_time.hour
        features["start_day"] = start_time.day
        features["start_month"] = start_time.month
        features["start_weekday"] = start_time.weekday()
        features["start_weekend"] = 1 if start_time.weekday() >= 5 else 0
        features["start_quarter"] = (start_time.month - 1) // 3 + 1
        
        # Calculate average time between events
        if len(case_df) > 1:
            timestamps = case_df[timestamp_col].sort_values().tolist()
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                          for i in range(len(timestamps)-1)]
            features["avg_time_between_events_seconds"] = sum(time_diffs) / len(time_diffs)
            features["max_time_between_events_seconds"] = max(time_diffs)
            features["min_time_between_events_seconds"] = min(time_diffs)
            features["std_time_between_events_seconds"] = np.std(time_diffs)
        else:
            features["avg_time_between_events_seconds"] = 0
            features["max_time_between_events_seconds"] = 0
            features["min_time_between_events_seconds"] = 0
            features["std_time_between_events_seconds"] = 0
    
    def _add_flow_features(self, features: Dict[str, Any], case_df: pd.DataFrame) -> None:
        """
        Add process flow-based features to the features dictionary.
        
        Args:
            features: Dictionary to add features to
            case_df: DataFrame containing events for a single case
        """
        activity_col = self.event_log.activity_column
        
        # Count occurrences of each activity
        activities = case_df[activity_col].tolist()
        
        # Activity counts (one-hot encoding)
        for activity in self.unique_activities:
            features[f"activity_{activity}_count"] = activities.count(activity)
            features[f"has_activity_{activity}"] = 1 if activity in activities else 0
        
        # First and last activity
        features["first_activity"] = activities[0]
        features["last_activity"] = activities[-1]
        
        # Create one-hot encoding for first and last activity
        for activity in self.unique_activities:
            features[f"first_activity_{activity}"] = 1 if activities[0] == activity else 0
            features[f"last_activity_{activity}"] = 1 if activities[-1] == activity else 0
        
        # Activity transitions
        if len(activities) > 1:
            transitions = [f"{activities[i]}_to_{activities[i+1]}" for i in range(len(activities)-1)]
            unique_transitions = set(transitions)
            
            # Count self-loops (same activity repeating)
            self_loops = sum(1 for i in range(len(activities)-1) 
                              if activities[i] == activities[i+1])
            features["self_loops_count"] = self_loops
            
            # Repetitions (revisiting activities)
            visited = set()
            repetitions = 0
            for activity in activities:
                if activity in visited:
                    repetitions += 1
                else:
                    visited.add(activity)
            features["activity_repetitions"] = repetitions
    
    def _add_resource_features(self, features: Dict[str, Any], case_df: pd.DataFrame) -> None:
        """
        Add resource-based features to the features dictionary.
        
        Args:
            features: Dictionary to add features to
            case_df: DataFrame containing events for a single case
        """
        if "resource" not in case_df.columns:
            return
        
        # Count unique resources
        features["unique_resources_count"] = case_df["resource"].nunique()
        
        # Resource handover count (number of resource changes)
        if len(case_df) > 1:
            resources = case_df["resource"].tolist()
            handovers = sum(1 for i in range(len(resources)-1) 
                             if resources[i] != resources[i+1])
            features["resource_handovers"] = handovers
        else:
            features["resource_handovers"] = 0
        
        # Most active resource
        if not case_df["resource"].empty:
            resource_counts = case_df["resource"].value_counts()
            features["most_active_resource"] = resource_counts.index[0]
            features["most_active_resource_count"] = resource_counts.iloc[0]
            
            # Resource counts (one-hot encoding)
            for resource in self.unique_resources:
                features[f"resource_{resource}_count"] = case_df["resource"].tolist().count(resource)
                features[f"has_resource_{resource}"] = 1 if resource in case_df["resource"].tolist() else 0


def create_target_variable(
    event_log: EventLog,
    target_type: str = "outcome",
    outcome_activity: Optional[str] = None,
    duration_unit: str = "days",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Create target variables for supervised learning from an event log.
    
    Args:
        event_log: EventLog to process
        target_type: Type of target variable ('outcome', 'duration', 'binary_duration')
        outcome_activity: Activity indicating a specific outcome (for 'outcome' type)
        duration_unit: Time unit for duration calculation ('seconds', 'minutes', 'hours', 'days')
        threshold: Threshold for binary duration classification (for 'binary_duration' type)
    
    Returns:
        DataFrame with case IDs and corresponding target variables
    """
    df = event_log.to_dataframe()
    case_id_col = event_log.case_id_column
    activity_col = event_log.activity_column
    timestamp_col = event_log.timestamp_column
    
    # Create result DataFrame
    result = []
    
    # Process each case
    for case_id, case_df in df.groupby(case_id_col):
        # Sort by timestamp
        case_df = case_df.sort_values(by=timestamp_col)
        
        # Initialize result row
        row = {case_id_col: case_id}
        
        # Extract target based on type
        if target_type == "outcome":
            # Use the last activity as outcome
            last_activity = case_df[activity_col].iloc[-1]
            row["outcome"] = last_activity
            
            # Check for specific outcome activity
            if outcome_activity:
                row["has_outcome"] = 1 if last_activity == outcome_activity else 0
        
        elif target_type in ["duration", "binary_duration"]:
            # Calculate case duration
            start_time = case_df[timestamp_col].min()
            end_time = case_df[timestamp_col].max()
            duration_seconds = (end_time - start_time).total_seconds()
            
            # Convert to requested unit
            if duration_unit == "minutes":
                duration = duration_seconds / 60
            elif duration_unit == "hours":
                duration = duration_seconds / 3600
            elif duration_unit == "days":
                duration = duration_seconds / 86400
            else:  # seconds
                duration = duration_seconds
            
            row["duration"] = duration
            
            # For binary duration, classify based on threshold
            if target_type == "binary_duration" and threshold is not None:
                row["duration_class"] = 1 if duration > threshold else 0
        
        result.append(row)
    
    return pd.DataFrame(result)


def combine_features_and_target(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    case_id_column: str = "case_id",
) -> pd.DataFrame:
    """
    Combine feature and target DataFrames into a single DataFrame.
    
    Args:
        features: DataFrame containing case features
        targets: DataFrame containing target variables
        case_id_column: Name of the case ID column
    
    Returns:
        Combined DataFrame with features and targets
    """
    return pd.merge(features, targets, on=case_id_column, how="inner")


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    encoding_method: str = "onehot",
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    Encode categorical features using various encoding methods.
    
    Args:
        df: DataFrame containing features
        categorical_columns: List of categorical columns to encode
        encoding_method: Encoding method ('onehot', 'label', 'ordinal')
        drop_first: Whether to drop the first category in one-hot encoding
    
    Returns:
        DataFrame with encoded categorical features
    """
    result_df = df.copy()
    
    for col in categorical_columns:
        if col not in result_df.columns:
            continue
        
        if encoding_method == "onehot":
            # One-hot encoding
            dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=drop_first)
            result_df = pd.concat([result_df, dummies], axis=1)
            result_df.drop(col, axis=1, inplace=True)
            
        elif encoding_method == "label":
            # Label encoding
            unique_values = result_df[col].dropna().unique()
            value_map = {val: i for i, val in enumerate(unique_values)}
            result_df[col] = result_df[col].map(value_map)
            
        elif encoding_method == "ordinal":
            # Ordinal encoding (assumes categories are provided in order)
            categories = result_df[col].dropna().unique()
            value_map = {cat: i for i, cat in enumerate(categories)}
            result_df[col] = result_df[col].map(value_map)
    
    return result_df


def scale_features(
    df: pd.DataFrame,
    numeric_columns: List[str],
    scaler_type: str = "standard",
) -> pd.DataFrame:
    """
    Scale numeric features using various scaling methods.
    
    Args:
        df: DataFrame containing features
        numeric_columns: List of numeric columns to scale
        scaler_type: Scaling method ('standard', 'minmax', 'robust')
    
    Returns:
        DataFrame with scaled numeric features
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    result_df = df.copy()
    numeric_df = result_df[numeric_columns].copy()
    
    # Handle missing values
    numeric_df.fillna(numeric_df.mean(), inplace=True)
    
    # Select scaler based on type
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Apply scaling
    scaled_values = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_columns, index=df.index)
    
    # Replace original columns with scaled values
    for col in numeric_columns:
        result_df[col] = scaled_df[col]
    
    return result_df, scaler