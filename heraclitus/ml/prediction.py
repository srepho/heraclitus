"""
Prediction module for process mining machine learning models.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

from heraclitus.data import EventLog
from heraclitus.ml.feature_engineering import (
    FeatureExtractor, create_target_variable, 
    combine_features_and_target, encode_categorical_features
)


class OutcomePredictor:
    """
    Predicts process outcomes based on case features.
    
    This class builds and applies machine learning models to predict
    categorical process outcomes (e.g., approved, rejected, completed).
    
    Attributes:
        model: The trained machine learning model
        feature_columns: List of feature column names
        target_column: Name of the target column
        categorical_columns: List of categorical feature columns
        model_info: Dictionary with model metadata
    """
    
    def __init__(self):
        """Initialize an OutcomePredictor instance."""
        self.model = None
        self.feature_columns = []
        self.target_column = "outcome"
        self.categorical_columns = []
        self.model_info = {}
    
    def train(
        self,
        event_log: EventLog,
        model_type: str = "random_forest",
        test_size: float = 0.2,
        random_state: int = 42,
        case_attributes: Optional[List[str]] = None,
        include_time_features: bool = True,
        include_flow_features: bool = True,
        include_resource_features: bool = True,
        outcome_activity: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train a predictive model for process outcomes.
        
        Args:
            event_log: EventLog containing process data
            model_type: Type of model ('random_forest', 'logistic_regression', 'svm', 'xgboost')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            case_attributes: List of case attributes to include as features
            include_time_features: Whether to include time-based features
            include_flow_features: Whether to include process flow features
            include_resource_features: Whether to include resource-based features
            outcome_activity: Specific activity to predict (binary outcome)
            model_params: Additional parameters for the model
        
        Returns:
            Dictionary with training results and evaluation metrics
        
        Raises:
            ValueError: If the model type is not supported
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
        
        # Create target variable
        if outcome_activity:
            target_type = "outcome"
            target_col = "has_outcome"
        else:
            target_type = "outcome"
            target_col = "outcome"
        
        targets_df = create_target_variable(
            event_log,
            target_type=target_type,
            outcome_activity=outcome_activity
        )
        
        # Combine features and target
        data_df = combine_features_and_target(
            features_df,
            targets_df,
            case_id_column=event_log.case_id_column
        )
        
        # Identify categorical columns
        categorical_columns = []
        for col in data_df.columns:
            if data_df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data_df[col]):
                categorical_columns.append(col)
        
        # Remove case_id and target columns from features
        feature_columns = [col for col in data_df.columns 
                          if col != event_log.case_id_column and col != target_col]
        
        # Handle categorical features
        if categorical_columns:
            categorical_features = [col for col in categorical_columns 
                                   if col in feature_columns]
            if categorical_features:
                data_df = encode_categorical_features(
                    data_df, 
                    categorical_features,
                    encoding_method="onehot"
                )
        
        # Update feature columns after encoding
        feature_columns = [col for col in data_df.columns 
                          if col != event_log.case_id_column and col != target_col]
        
        # Split data into training and testing sets
        X = data_df[feature_columns]
        y = data_df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None
        )
        
        # Create and train the model
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model_class = RandomForestClassifier
            default_params = {"n_estimators": 100, "random_state": random_state}
        
        elif model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            model_class = LogisticRegression
            default_params = {"random_state": random_state, "max_iter": 1000}
        
        elif model_type == "svm":
            from sklearn.svm import SVC
            model_class = SVC
            default_params = {"random_state": random_state, "probability": True}
        
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                model_class = XGBClassifier
                default_params = {"random_state": random_state}
            except ImportError:
                raise ValueError("XGBoost is not installed. Install with 'pip install xgboost'")
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Combine default and user-provided parameters
        params = {**default_params, **(model_params or {})}
        
        # Create model pipeline with preprocessing
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model_class(**params))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate evaluation metrics
        if len(np.unique(y)) > 2:  # Multiclass
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {
                "accuracy": accuracy,
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }
        else:  # Binary
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {
                "accuracy": accuracy,
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
        
        # Get detailed classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Get feature importances if available
        feature_importances = {}
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importances = pipeline.named_steps['model'].feature_importances_
            feature_importances = dict(zip(X.columns, importances))
            # Sort by importance
            feature_importances = {k: v for k, v in sorted(
                feature_importances.items(), key=lambda item: item[1], reverse=True
            )}
        
        # Store model and metadata
        self.model = pipeline
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_col
        
        # Store model information
        self.model_info = {
            "model_type": model_type,
            "target_type": target_type,
            "outcome_activity": outcome_activity,
            "feature_count": len(feature_columns),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": metrics,
            "classification_report": class_report,
            "feature_importances": feature_importances
        }
        
        return self.model_info
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Args:
            features_df: DataFrame containing features for prediction
        
        Returns:
            DataFrame with case IDs and predicted outcomes
        
        Raises:
            ValueError: If the model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Ensure all required columns are present
        missing_columns = [col for col in self.feature_columns if col not in features_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")
        
        # Extract case IDs
        if "case_id" in features_df.columns:
            case_ids = features_df["case_id"].copy()
        else:
            case_ids = pd.Series(range(len(features_df)), name="case_id")
        
        # Make predictions
        X = features_df[self.feature_columns]
        y_pred = self.model.predict(X)
        
        # Get prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)
            
            # Create results DataFrame with probabilities
            if y_proba.shape[1] == 2:  # Binary classification
                results = pd.DataFrame({
                    "case_id": case_ids,
                    "predicted_outcome": y_pred,
                    "probability": y_proba[:, 1]
                })
            else:  # Multiclass classification
                results = pd.DataFrame({
                    "case_id": case_ids,
                    "predicted_outcome": y_pred
                })
                
                # Add probability for each class
                classes = self.model.classes_
                for i, cls in enumerate(classes):
                    results[f"probability_{cls}"] = y_proba[:, i]
        else:
            # Create results DataFrame without probabilities
            results = pd.DataFrame({
                "case_id": case_ids,
                "predicted_outcome": y_pred
            })
        
        return results
    
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
            "target_column": self.target_column,
            "model_info": self.model_info
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
        self.target_column = metadata["target_column"]
        self.model_info = metadata["model_info"]


class DurationPredictor:
    """
    Predicts process durations based on case features.
    
    This class builds and applies machine learning models to predict
    numeric process durations or binary duration classes.
    
    Attributes:
        model: The trained machine learning model
        feature_columns: List of feature column names
        target_column: Name of the target column
        categorical_columns: List of categorical feature columns
        model_info: Dictionary with model metadata
        is_classifier: Whether the model is a classifier (binary duration)
        scaler: The scaler used for target normalization (regression only)
    """
    
    def __init__(self):
        """Initialize a DurationPredictor instance."""
        self.model = None
        self.feature_columns = []
        self.target_column = "duration"
        self.categorical_columns = []
        self.model_info = {}
        self.is_classifier = False
        self.scaler = None
    
    def train(
        self,
        event_log: EventLog,
        model_type: str = "random_forest",
        prediction_type: str = "regression",
        duration_unit: str = "days",
        duration_threshold: Optional[float] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        case_attributes: Optional[List[str]] = None,
        include_time_features: bool = True,
        include_flow_features: bool = True,
        include_resource_features: bool = True,
        model_params: Optional[Dict[str, Any]] = None,
        normalize_target: bool = True,
    ) -> Dict[str, Any]:
        """
        Train a predictive model for process durations.
        
        Args:
            event_log: EventLog containing process data
            model_type: Type of model ('random_forest', 'linear_regression', 'svr', 'xgboost')
            prediction_type: Type of prediction ('regression', 'classification')
            duration_unit: Time unit for duration ('seconds', 'minutes', 'hours', 'days')
            duration_threshold: Threshold for binary classification (if prediction_type is 'classification')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            case_attributes: List of case attributes to include as features
            include_time_features: Whether to include time-based features
            include_flow_features: Whether to include process flow features
            include_resource_features: Whether to include resource-based features
            model_params: Additional parameters for the model
            normalize_target: Whether to normalize the target variable (regression only)
        
        Returns:
            Dictionary with training results and evaluation metrics
        
        Raises:
            ValueError: If the model type or prediction type is not supported
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
        
        # Create target variable
        if prediction_type == "classification":
            if duration_threshold is None:
                raise ValueError("duration_threshold must be provided for classification")
            
            target_type = "binary_duration"
            target_col = "duration_class"
            self.is_classifier = True
        else:
            target_type = "duration"
            target_col = "duration"
            self.is_classifier = False
        
        targets_df = create_target_variable(
            event_log,
            target_type=target_type,
            duration_unit=duration_unit,
            threshold=duration_threshold
        )
        
        # Combine features and target
        data_df = combine_features_and_target(
            features_df,
            targets_df,
            case_id_column=event_log.case_id_column
        )
        
        # Identify categorical columns
        categorical_columns = []
        for col in data_df.columns:
            if data_df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data_df[col]):
                categorical_columns.append(col)
        
        # Remove case_id and target columns from features
        feature_columns = [col for col in data_df.columns 
                          if col != event_log.case_id_column 
                          and col != target_col
                          and col != "duration"]  # Exclude duration if target is duration_class
        
        # Handle categorical features
        if categorical_columns:
            categorical_features = [col for col in categorical_columns 
                                   if col in feature_columns]
            if categorical_features:
                data_df = encode_categorical_features(
                    data_df, 
                    categorical_features,
                    encoding_method="onehot"
                )
        
        # Update feature columns after encoding
        feature_columns = [col for col in data_df.columns 
                          if col != event_log.case_id_column 
                          and col != target_col
                          and col != "duration"]  # Exclude duration if target is duration_class
        
        # Split data into training and testing sets
        X = data_df[feature_columns]
        y = data_df[target_col]
        
        # Normalize target variable for regression
        if not self.is_classifier and normalize_target:
            from sklearn.preprocessing import StandardScaler
            y_scaler = StandardScaler()
            y_array = y.values.reshape(-1, 1)
            y_scaled = y_scaler.fit_transform(y_array).flatten()
            y = pd.Series(y_scaled, index=y.index)
            self.scaler = y_scaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create and train the model
        if self.is_classifier:
            # Classification models
            if model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                model_class = RandomForestClassifier
                default_params = {"n_estimators": 100, "random_state": random_state}
            
            elif model_type == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                model_class = LogisticRegression
                default_params = {"random_state": random_state, "max_iter": 1000}
            
            elif model_type == "svm":
                from sklearn.svm import SVC
                model_class = SVC
                default_params = {"random_state": random_state, "probability": True}
            
            elif model_type == "xgboost":
                try:
                    from xgboost import XGBClassifier
                    model_class = XGBClassifier
                    default_params = {"random_state": random_state}
                except ImportError:
                    raise ValueError("XGBoost is not installed. Install with 'pip install xgboost'")
            
            else:
                raise ValueError(f"Unsupported classification model type: {model_type}")
            
            # Combine default and user-provided parameters
            params = {**default_params, **(model_params or {})}
            
            # Create model pipeline with preprocessing
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', model_class(**params))
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = pipeline.predict(X_test)
            
            # Calculate evaluation metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }
            
            # Get detailed classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
        else:
            # Regression models
            if model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                model_class = RandomForestRegressor
                default_params = {"n_estimators": 100, "random_state": random_state}
            
            elif model_type == "linear_regression":
                from sklearn.linear_model import LinearRegression
                model_class = LinearRegression
                default_params = {}
            
            elif model_type == "svr":
                from sklearn.svm import SVR
                model_class = SVR
                default_params = {}
            
            elif model_type == "xgboost":
                try:
                    from xgboost import XGBRegressor
                    model_class = XGBRegressor
                    default_params = {"random_state": random_state}
                except ImportError:
                    raise ValueError("XGBoost is not installed. Install with 'pip install xgboost'")
            
            else:
                raise ValueError(f"Unsupported regression model type: {model_type}")
            
            # Combine default and user-provided parameters
            params = {**default_params, **(model_params or {})}
            
            # Create model pipeline with preprocessing
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', model_class(**params))
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = pipeline.predict(X_test)
            
            # Calculate evaluation metrics
            if self.scaler:
                # Transform predictions and actual values back to original scale
                y_test_orig = self.scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
                y_pred_orig = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                
                metrics = {
                    "mean_squared_error": mean_squared_error(y_test_orig, y_pred_orig),
                    "root_mean_squared_error": np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
                    "mean_absolute_error": mean_absolute_error(y_test_orig, y_pred_orig),
                    "r2_score": r2_score(y_test_orig, y_pred_orig)
                }
            else:
                metrics = {
                    "mean_squared_error": mean_squared_error(y_test, y_pred),
                    "root_mean_squared_error": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mean_absolute_error": mean_absolute_error(y_test, y_pred),
                    "r2_score": r2_score(y_test, y_pred)
                }
            
            class_report = None
        
        # Get feature importances if available
        feature_importances = {}
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importances = pipeline.named_steps['model'].feature_importances_
            feature_importances = dict(zip(X.columns, importances))
            # Sort by importance
            feature_importances = {k: v for k, v in sorted(
                feature_importances.items(), key=lambda item: item[1], reverse=True
            )}
        
        # Store model and metadata
        self.model = pipeline
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_col
        
        # Store model information
        self.model_info = {
            "model_type": model_type,
            "prediction_type": prediction_type,
            "duration_unit": duration_unit,
            "duration_threshold": duration_threshold,
            "feature_count": len(feature_columns),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": metrics,
            "classification_report": class_report,
            "feature_importances": feature_importances,
            "normalized_target": normalize_target and not self.is_classifier
        }
        
        return self.model_info
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Args:
            features_df: DataFrame containing features for prediction
        
        Returns:
            DataFrame with case IDs and predicted durations
        
        Raises:
            ValueError: If the model has not been trained
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Ensure all required columns are present
        missing_columns = [col for col in self.feature_columns if col not in features_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")
        
        # Extract case IDs
        if "case_id" in features_df.columns:
            case_ids = features_df["case_id"].copy()
        else:
            case_ids = pd.Series(range(len(features_df)), name="case_id")
        
        # Make predictions
        X = features_df[self.feature_columns]
        y_pred = self.model.predict(X)
        
        # Create results DataFrame
        if self.is_classifier:
            # Classification result
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X)
                
                results = pd.DataFrame({
                    "case_id": case_ids,
                    "predicted_class": y_pred,
                    "probability": y_proba[:, 1]
                })
            else:
                results = pd.DataFrame({
                    "case_id": case_ids,
                    "predicted_class": y_pred
                })
        else:
            # Regression result
            if self.scaler:
                # Transform predictions back to original scale
                y_pred_orig = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                results = pd.DataFrame({
                    "case_id": case_ids,
                    "predicted_duration": y_pred_orig
                })
            else:
                results = pd.DataFrame({
                    "case_id": case_ids,
                    "predicted_duration": y_pred
                })
        
        return results
    
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
        
        # Save scaler if available
        if self.scaler is not None:
            joblib.dump(self.scaler, f"{filepath}_scaler.joblib")
        
        # Save metadata as JSON
        metadata = {
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "target_column": self.target_column,
            "model_info": self.model_info,
            "is_classifier": self.is_classifier,
            "has_scaler": self.scaler is not None
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
        self.target_column = metadata["target_column"]
        self.model_info = metadata["model_info"]
        self.is_classifier = metadata["is_classifier"]
        
        # Load scaler if available
        if metadata.get("has_scaler", False):
            self.scaler = joblib.load(f"{filepath}_scaler.joblib")
        else:
            self.scaler = None