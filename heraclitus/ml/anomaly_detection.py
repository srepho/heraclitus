"""
Anomaly detection module for process mining.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from heraclitus.data import EventLog
from heraclitus.ml.feature_engineering import FeatureExtractor, encode_categorical_features


class ProcessAnomalyDetector:
    """
    Detects anomalies in process data using machine learning.
    
    This class provides methods to identify unusual process behavior,
    such as anomalous cases, activities, or process flows.
    
    Attributes:
        model: The trained anomaly detection model
        feature_columns: List of feature column names
        categorical_columns: List of categorical feature columns
        model_info: Dictionary with model metadata
        anomaly_threshold: Threshold for determining anomalies
    """
    
    def __init__(self):
        """Initialize a ProcessAnomalyDetector instance."""
        self.model = None
        self.feature_columns = []
        self.categorical_columns = []
        self.model_info = {}
        self.anomaly_threshold = None
    
    def train(
        self,
        event_log: EventLog,
        method: str = "isolation_forest",
        case_attributes: Optional[List[str]] = None,
        include_time_features: bool = True,
        include_flow_features: bool = True,
        include_resource_features: bool = True,
        contamination: float = 0.1,
        random_state: int = 42,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train an anomaly detection model.
        
        Args:
            event_log: EventLog containing process data
            method: Anomaly detection method ('isolation_forest', 'local_outlier_factor', 'dbscan')
            case_attributes: List of case attributes to include as features
            include_time_features: Whether to include time-based features
            include_flow_features: Whether to include process flow features
            include_resource_features: Whether to include resource-based features
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
            model_params: Additional parameters for the model
        
        Returns:
            Dictionary with training results and information
        
        Raises:
            ValueError: If the method is not supported
        """
        # Extract features
        feature_extractor = FeatureExtractor(
            event_log,
            case_attributes=case_attributes,
            include_time_features=include_time_features,
            include_flow_features=include_flow_features,
            include_resource_features=include_resource_features
        )
        features_df = feature_extractor.extract_case_features()
        
        # Store case IDs
        case_ids = features_df[event_log.case_id_column].copy()
        
        # Identify categorical columns
        categorical_columns = []
        for col in features_df.columns:
            if (col != event_log.case_id_column and 
                (features_df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(features_df[col]))):
                categorical_columns.append(col)
        
        # Handle categorical features
        if categorical_columns:
            features_df = encode_categorical_features(
                features_df, 
                categorical_columns,
                encoding_method="onehot"
            )
        
        # Remove case_id column from features
        feature_columns = [col for col in features_df.columns if col != event_log.case_id_column]
        X = features_df[feature_columns]
        
        # Create and train the model
        if method == "isolation_forest":
            # Default parameters
            default_params = {
                "contamination": contamination,
                "random_state": random_state
            }
            
            # Combine with user-provided parameters
            params = {**default_params, **(model_params or {})}
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', IsolationForest(**params))
            ])
            
            # Train model
            pipeline.fit(X)
            
            # Calculate anomaly scores
            anomaly_scores = -pipeline.named_steps['model'].decision_function(X)
            predictions = pipeline.named_steps['model'].predict(X)
            # Convert from {1: normal, -1: anomaly} to {0: normal, 1: anomaly}
            is_anomaly = (predictions == -1).astype(int)
            
        elif method == "local_outlier_factor":
            # Default parameters
            default_params = {
                "contamination": contamination,
                "novelty": True,  # Enable predict method
                "n_neighbors": 20
            }
            
            # Combine with user-provided parameters
            params = {**default_params, **(model_params or {})}
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', LocalOutlierFactor(**params))
            ])
            
            # Train model
            pipeline.fit(X)
            
            # Calculate anomaly scores
            anomaly_scores = -pipeline.named_steps['model'].decision_function(X)
            predictions = pipeline.named_steps['model'].predict(X)
            # Convert from {1: normal, -1: anomaly} to {0: normal, 1: anomaly}
            is_anomaly = (predictions == -1).astype(int)
            
        elif method == "dbscan":
            # Default parameters
            default_params = {
                "eps": 0.5,
                "min_samples": 5
            }
            
            # Combine with user-provided parameters
            params = {**default_params, **(model_params or {})}
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', DBSCAN(**params))
            ])
            
            # Train model
            pipeline.fit(X)
            
            # Get cluster labels (-1 means outlier)
            cluster_labels = pipeline.named_steps['model'].labels_
            is_anomaly = (cluster_labels == -1).astype(int)
            
            # Calculate anomaly scores (distance to nearest cluster center)
            # For DBSCAN, we'll use a simple heuristic: outliers have score 1, others have score based on cluster size
            unique_clusters = np.unique(cluster_labels)
            unique_clusters = unique_clusters[unique_clusters != -1]  # Remove outlier cluster
            
            if len(unique_clusters) > 0:
                # Count samples in each cluster
                cluster_sizes = {}
                for cluster in unique_clusters:
                    cluster_sizes[cluster] = np.sum(cluster_labels == cluster)
                
                # Normalize by largest cluster
                max_size = max(cluster_sizes.values()) if cluster_sizes else 1
                
                # Assign scores
                anomaly_scores = np.zeros(len(X))
                for i, label in enumerate(cluster_labels):
                    if label == -1:
                        anomaly_scores[i] = 1.0  # Maximum anomaly score for outliers
                    else:
                        # Smaller clusters are more anomalous
                        anomaly_scores[i] = 1.0 - (cluster_sizes[label] / max_size)
            else:
                # If no clusters were found, all points are outliers
                anomaly_scores = np.ones(len(X))
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Create results DataFrame
        results = pd.DataFrame({
            "case_id": case_ids,
            "anomaly_score": anomaly_scores,
            "is_anomaly": is_anomaly
        })
        
        # Calculate anomaly threshold (if not DBSCAN)
        if method != "dbscan":
            # Calculate threshold based on contamination
            threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
        else:
            threshold = 0.5  # For DBSCAN, 0.5 is a reasonable default
        
        self.anomaly_threshold = threshold
        
        # Calculate metrics
        anomaly_count = results["is_anomaly"].sum()
        total_count = len(results)
        anomaly_percentage = (anomaly_count / total_count) * 100
        
        # Store model and metadata
        self.model = pipeline
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns
        
        # Store model information
        self.model_info = {
            "method": method,
            "total_cases": total_count,
            "anomaly_count": int(anomaly_count),
            "anomaly_percentage": anomaly_percentage,
            "anomaly_threshold": threshold,
            "contamination": contamination,
            "feature_count": len(feature_columns)
        }
        
        return self.model_info
    
    def detect_anomalies(
        self,
        features_df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Detect anomalies in the provided features.
        
        Args:
            features_df: DataFrame containing features for detection
            threshold: Optional custom threshold for anomaly detection
        
        Returns:
            DataFrame with case IDs, anomaly scores, and anomaly flags
        
        Raises:
            ValueError: If the model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Use stored threshold if not provided
        if threshold is None:
            threshold = self.anomaly_threshold
        
        # Extract case IDs
        if "case_id" in features_df.columns:
            case_ids = features_df["case_id"].copy()
        else:
            case_ids = pd.Series(range(len(features_df)), name="case_id")
        
        # Ensure all required feature columns are present
        missing_columns = [col for col in self.feature_columns if col not in features_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")
        
        # Extract features
        X = features_df[self.feature_columns]
        
        # Get method name from model pipeline
        method = self.model_info.get("method", "unknown")
        
        # Detect anomalies
        if method == "isolation_forest" or method == "local_outlier_factor":
            # Calculate anomaly scores
            anomaly_scores = -self.model.named_steps['model'].decision_function(X)
            predictions = self.model.named_steps['model'].predict(X)
            # Convert from {1: normal, -1: anomaly} to {0: normal, 1: anomaly}
            is_anomaly = (predictions == -1).astype(int)
            
        elif method == "dbscan":
            # Get cluster labels
            cluster_labels = self.model.named_steps['model'].predict(X)
            is_anomaly = (cluster_labels == -1).astype(int)
            
            # Calculate anomaly scores
            unique_clusters = np.unique(cluster_labels)
            unique_clusters = unique_clusters[unique_clusters != -1]  # Remove outlier cluster
            
            if len(unique_clusters) > 0:
                # Count samples in each cluster
                cluster_sizes = {}
                for cluster in unique_clusters:
                    cluster_sizes[cluster] = np.sum(cluster_labels == cluster)
                
                # Normalize by largest cluster
                max_size = max(cluster_sizes.values()) if cluster_sizes else 1
                
                # Assign scores
                anomaly_scores = np.zeros(len(X))
                for i, label in enumerate(cluster_labels):
                    if label == -1:
                        anomaly_scores[i] = 1.0  # Maximum anomaly score for outliers
                    else:
                        # Smaller clusters are more anomalous
                        anomaly_scores[i] = 1.0 - (cluster_sizes[label] / max_size)
            else:
                # If no clusters were found, all points are outliers
                anomaly_scores = np.ones(len(X))
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Create results DataFrame
        results = pd.DataFrame({
            "case_id": case_ids,
            "anomaly_score": anomaly_scores,
            "is_anomaly": is_anomaly,
            "threshold": threshold
        })
        
        # Order by anomaly score (descending)
        results = results.sort_values(by="anomaly_score", ascending=False)
        
        return results
    
    def visualize_anomalies(
        self,
        results: pd.DataFrame,
        features_df: pd.DataFrame,
        top_n_features: int = 10,
        figsize: Tuple[int, int] = (15, 8),
    ) -> plt.Figure:
        """
        Visualize anomalies with key distinguishing features.
        
        Args:
            results: DataFrame returned by detect_anomalies
            features_df: Original features DataFrame
            top_n_features: Number of top features to highlight
            figsize: Figure size (width, height)
        
        Returns:
            Matplotlib Figure object
        """
        # Merge results with features
        merged_df = pd.merge(
            results,
            features_df,
            on="case_id",
            how="inner"
        )
        
        # Separate normal and anomalous cases
        normal_df = merged_df[merged_df["is_anomaly"] == 0]
        anomaly_df = merged_df[merged_df["is_anomaly"] == 1]
        
        # If no anomalies found, return empty figure
        if len(anomaly_df) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No anomalies detected", ha="center", va="center")
            return fig
        
        # Find features with largest differences between normal and anomalous cases
        feature_diffs = {}
        for col in self.feature_columns:
            if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col]):
                try:
                    # Calculate mean difference
                    normal_mean = normal_df[col].mean()
                    anomaly_mean = anomaly_df[col].mean()
                    
                    # Calculate standard deviations
                    normal_std = normal_df[col].std()
                    anomaly_std = anomaly_df[col].std()
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt((normal_std**2 + anomaly_std**2) / 2)
                    effect_size = abs(normal_mean - anomaly_mean) / pooled_std if pooled_std else 0
                    
                    feature_diffs[col] = effect_size
                except:
                    # Skip this feature if there's an error
                    continue
        
        # Get top differentiating features
        top_features = sorted(feature_diffs.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
        top_feature_names = [f[0] for f in top_features]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # 1. Distribution of anomaly scores
        plt.subplot(2, 2, 1)
        sns.histplot(data=merged_df, x="anomaly_score", hue="is_anomaly", bins=30, kde=True)
        plt.axvline(x=results["threshold"].iloc[0], color="red", linestyle="--", 
                    label=f"Threshold: {results['threshold'].iloc[0]:.3f}")
        plt.title("Distribution of Anomaly Scores")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Count")
        plt.legend()
        
        # 2. Feature importance for distinguishing anomalies
        plt.subplot(2, 2, 2)
        feature_importance = [f[1] for f in top_features]
        feature_names = [f[0][:20] + "..." if len(f[0]) > 20 else f[0] for f in top_features]
        
        y_pos = np.arange(len(feature_names))
        plt.barh(y_pos, feature_importance, align="center")
        plt.yticks(y_pos, feature_names)
        plt.xlabel("Effect Size (Cohen's d)")
        plt.title("Top Features Distinguishing Anomalies")
        
        # 3. Scatter plot of top 2 features
        if len(top_feature_names) >= 2:
            plt.subplot(2, 2, 3)
            sns.scatterplot(
                data=merged_df,
                x=top_feature_names[0],
                y=top_feature_names[1],
                hue="is_anomaly",
                style="is_anomaly",
                s=100,
                alpha=0.7
            )
            plt.title(f"Anomalies by Top 2 Features")
            plt.xlabel(top_feature_names[0])
            plt.ylabel(top_feature_names[1])
            plt.legend()
        
        # 4. Box plots of top features
        plt.subplot(2, 2, 4)
        boxplot_df = pd.melt(
            merged_df[["is_anomaly"] + top_feature_names[:5]],
            id_vars=["is_anomaly"],
            var_name="Feature",
            value_name="Value"
        )
        sns.boxplot(data=boxplot_df, x="Feature", y="Value", hue="is_anomaly")
        plt.title("Distribution of Top 5 Features")
        plt.xticks(rotation=45)
        plt.legend(title="Is Anomaly")
        
        plt.tight_layout()
        return fig
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model and metadata to disk.
        
        Args:
            filepath: Path to save the model (without extension)
        
        Raises:
            ValueError: If the model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model using joblib
        joblib.dump(self.model, f"{filepath}.joblib")
        
        # Save metadata as JSON
        metadata = {
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "model_info": self.model_info,
            "anomaly_threshold": self.anomaly_threshold
        }
        
        with open(f"{filepath}.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained model and metadata from disk.
        
        Args:
            filepath: Path to the saved model (without extension)
        """
        filepath = Path(filepath)
        
        # Load model using joblib
        self.model = joblib.load(f"{filepath}.joblib")
        
        # Load metadata from JSON
        with open(f"{filepath}.json", "r") as f:
            metadata = json.load(f)
        
        # Restore metadata
        self.feature_columns = metadata["feature_columns"]
        self.categorical_columns = metadata["categorical_columns"]
        self.model_info = metadata["model_info"]
        self.anomaly_threshold = metadata["anomaly_threshold"]


def detect_variant_anomalies(
    event_log: EventLog,
    min_support: float = 0.01,
    max_variants: int = 10,
) -> pd.DataFrame:
    """
    Detect anomalous process variants based on frequency analysis.
    
    Args:
        event_log: EventLog to analyze
        min_support: Minimum support (proportion of cases) for a variant to be considered normal
        max_variants: Maximum number of variants to consider normal
    
    Returns:
        DataFrame with variant information and anomaly flags
    """
    df = event_log.to_dataframe()
    case_id_col = event_log.case_id_column
    activity_col = event_log.activity_column
    timestamp_col = event_log.timestamp_column
    
    # Extract variants
    variants = []
    
    for case_id, case_df in df.groupby(case_id_col):
        # Sort by timestamp
        case_df = case_df.sort_values(by=timestamp_col)
        
        # Extract activity sequence
        activities = case_df[activity_col].tolist()
        variant = "->".join(activities)
        
        variants.append({
            "case_id": case_id,
            "variant": variant,
            "length": len(activities)
        })
    
    # Create variants DataFrame
    variants_df = pd.DataFrame(variants)
    
    # Count variant occurrences
    variant_counts = variants_df["variant"].value_counts()
    total_cases = len(variants_df)
    
    # Calculate variant support
    variant_support = variant_counts / total_cases
    
    # Identify normal variants (frequent variants)
    normal_variants = variant_support[variant_support >= min_support].index.tolist()
    
    # If there are too many normal variants, limit by frequency
    if len(normal_variants) > max_variants:
        normal_variants = variant_counts.nlargest(max_variants).index.tolist()
    
    # Add variant information
    result = []
    
    for variant, count in variant_counts.items():
        support = count / total_cases
        is_anomaly = variant not in normal_variants
        
        result.append({
            "variant": variant,
            "count": count,
            "support": support,
            "is_anomaly": is_anomaly
        })
    
    # Create result DataFrame
    result_df = pd.DataFrame(result)
    
    # Add case-level information
    variants_df["variant_count"] = variants_df["variant"].map(variant_counts)
    variants_df["variant_support"] = variants_df["variant"].map(variant_support)
    variants_df["is_anomaly"] = ~variants_df["variant"].isin(normal_variants)
    
    # Sort by support
    result_df = result_df.sort_values(by="support", ascending=False)
    
    return result_df, variants_df