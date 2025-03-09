"""
EventLog class implementation - the central data structure for Heraclitus.
"""
from __future__ import annotations

import pandas as pd
from typing import List, Union, Optional, Literal, Dict, Any


class EventLog:
    """
    EventLog class represents event data with case_id, activity, and timestamp.
    
    The EventLog is the central data structure in Heraclitus, providing a
    standardized interface for working with process mining data.
    
    Attributes:
        _df: Internal pandas DataFrame storing the event log data
        case_id_column: Column name for case identifiers
        activity_column: Column name for activities
        timestamp_column: Column name for timestamps
        attribute_columns: List of additional attribute column names
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        case_id_column: str = "case_id",
        activity_column: str = "activity",
        timestamp_column: str = "timestamp",
        duplicate_handling: Literal["error", "warn", "keep_first", "keep_last"] = "warn",
        **kwargs: Any,
    ) -> None:
        """
        Initialize an EventLog instance.
        
        Args:
            data: Data source (pandas DataFrame or file path)
            case_id_column: Column name for case identifiers
            activity_column: Column name for activities
            timestamp_column: Column name for timestamps
            duplicate_handling: How to handle duplicate events
            **kwargs: Additional parameters passed to data loading functions
        
        Raises:
            ValueError: If required columns are missing or data format is invalid
            TypeError: If the data type is not supported
        """
        self.case_id_column = case_id_column
        self.activity_column = activity_column
        self.timestamp_column = timestamp_column
        self.attribute_columns: List[str] = []
        
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            self._df = data.copy()
        elif isinstance(data, str):
            # Lazy load from file
            if data.endswith(".csv"):
                self._df = pd.read_csv(data, **kwargs)
            elif data.endswith((".xls", ".xlsx", ".xlsm")):
                self._df = pd.read_excel(data, **kwargs)
            else:
                raise ValueError(f"Unsupported file format for: {data}")
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        # Validate required columns
        for col in [case_id_column, activity_column, timestamp_column]:
            if col not in self._df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Convert timestamps to datetime
        self._df[timestamp_column] = pd.to_datetime(
            self._df[timestamp_column], errors="coerce"
        )
        
        # Check for invalid timestamps
        if self._df[timestamp_column].isna().any():
            raise ValueError("Found invalid timestamps in data")
        
        # Handle duplicates
        duplicated = self._df.duplicated(
            subset=[case_id_column, activity_column, timestamp_column], keep=False
        )
        if duplicated.any():
            if duplicate_handling == "error":
                raise ValueError("Duplicate events found in the data")
            elif duplicate_handling == "warn":
                dup_count = duplicated.sum()
                print(f"Warning: {dup_count} duplicate events found in the data")
            elif duplicate_handling == "keep_first":
                self._df = self._df.drop_duplicates(
                    subset=[case_id_column, activity_column, timestamp_column],
                    keep="first",
                )
            elif duplicate_handling == "keep_last":
                self._df = self._df.drop_duplicates(
                    subset=[case_id_column, activity_column, timestamp_column],
                    keep="last",
                )
        
        # Store attribute columns
        self.attribute_columns = [
            col for col in self._df.columns 
            if col not in [case_id_column, activity_column, timestamp_column]
        ]
        
        # Sort by case_id and timestamp
        self._df = self._df.sort_values(by=[case_id_column, timestamp_column])
    
    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, **kwargs: Any
    ) -> EventLog:
        """
        Create an EventLog instance from a pandas DataFrame.
        
        Args:
            df: Source DataFrame containing event data
            **kwargs: Additional parameters passed to the EventLog constructor
        
        Returns:
            A new EventLog instance
        """
        return cls(data=df, **kwargs)
    
    @classmethod
    def from_csv(cls, filepath: str, **kwargs: Any) -> EventLog:
        """
        Create an EventLog instance from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            **kwargs: Additional parameters passed to pd.read_csv and the EventLog constructor
        
        Returns:
            A new EventLog instance
        """
        return cls(data=filepath, **kwargs)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a copy of the internal DataFrame.
        
        Returns:
            A copy of the internal DataFrame
        """
        return self._df.copy()
    
    def filter_cases(self, case_ids: List[str]) -> EventLog:
        """
        Filter the event log to include only specified case IDs.
        
        Args:
            case_ids: List of case IDs to include
        
        Returns:
            A new EventLog instance with only the specified cases
        """
        filtered_df = self._df[self._df[self.case_id_column].isin(case_ids)]
        return EventLog.from_dataframe(
            filtered_df,
            case_id_column=self.case_id_column,
            activity_column=self.activity_column,
            timestamp_column=self.timestamp_column,
        )
    
    def filter_activities(self, activities: List[str]) -> EventLog:
        """
        Filter the event log to include only specified activities.
        
        Args:
            activities: List of activities to include
        
        Returns:
            A new EventLog instance with only the specified activities
        """
        filtered_df = self._df[self._df[self.activity_column].isin(activities)]
        return EventLog.from_dataframe(
            filtered_df,
            case_id_column=self.case_id_column,
            activity_column=self.activity_column,
            timestamp_column=self.timestamp_column,
        )
    
    def filter_time_range(
        self, start_time: Optional[pd.Timestamp] = None, end_time: Optional[pd.Timestamp] = None
    ) -> EventLog:
        """
        Filter the event log to include only events within the specified time range.
        
        Args:
            start_time: Start of the time range (inclusive)
            end_time: End of the time range (inclusive)
        
        Returns:
            A new EventLog instance with only events in the specified time range
        """
        filtered_df = self._df.copy()
        
        if start_time is not None:
            filtered_df = filtered_df[
                filtered_df[self.timestamp_column] >= start_time
            ]
        
        if end_time is not None:
            filtered_df = filtered_df[
                filtered_df[self.timestamp_column] <= end_time
            ]
        
        return EventLog.from_dataframe(
            filtered_df,
            case_id_column=self.case_id_column,
            activity_column=self.activity_column,
            timestamp_column=self.timestamp_column,
        )
    
    def get_attributes(self) -> List[str]:
        """
        Return the list of attribute column names.
        
        Returns:
            List of attribute column names
        """
        return self.attribute_columns.copy()
    
    def add_attribute(self, column_name: str, data: List[Any]) -> None:
        """
        Add a new attribute column to the event log.
        
        Args:
            column_name: Name of the new attribute column
            data: Data for the new attribute column
        
        Raises:
            ValueError: If the data length doesn't match the number of rows,
                       or if the column already exists
        """
        if column_name in self._df.columns:
            raise ValueError(f"Column '{column_name}' already exists")
        
        if len(data) != len(self._df):
            raise ValueError(
                f"Data length ({len(data)}) doesn't match number of rows ({len(self._df)})"
            )
        
        self._df[column_name] = data
        if column_name not in [
            self.case_id_column, self.activity_column, self.timestamp_column
        ]:
            self.attribute_columns.append(column_name)
    
    def __len__(self) -> int:
        """Return the number of events in the log."""
        return len(self._df)
    
    def case_count(self) -> int:
        """Return the number of unique cases in the log."""
        return self._df[self.case_id_column].nunique()
    
    def activity_count(self) -> int:
        """Return the number of unique activities in the log."""
        return self._df[self.activity_column].nunique()
    
    def get_unique_activities(self) -> List[str]:
        """Return a list of unique activities in the log."""
        return self._df[self.activity_column].unique().tolist()
    
    def get_unique_cases(self) -> List[str]:
        """Return a list of unique case IDs in the log."""
        return self._df[self.case_id_column].unique().tolist()