"""
Time-based metrics for process analysis.
"""
from typing import Optional, List, Dict, Union, Literal, Tuple
import pandas as pd
import numpy as np
from functools import lru_cache

from heraclitus.data import EventLog


@lru_cache(maxsize=32)
def calculate_cycle_time(
    event_log: EventLog,
    case_id: Optional[str] = None,
    start_activity: Optional[str] = None,
    end_activity: Optional[str] = None,
    unit: Literal["seconds", "minutes", "hours", "days"] = "seconds",
    include_stats: bool = False,
) -> Union[float, Dict[str, float]]:
    """
    Calculate cycle time between activities.
    
    Args:
        event_log: The event log to analyze
        case_id: Optional specific case ID to analyze
        start_activity: Optional specific start activity (first occurrence)
        end_activity: Optional specific end activity (last occurrence)
        unit: Time unit for the result
        include_stats: Whether to include additional statistics
    
    Returns:
        If include_stats is False, returns the mean cycle time.
        If include_stats is True, returns a dictionary with mean, median, min, max,
        and standard deviation of cycle times.
        
    Raises:
        ValueError: If case_id is provided but not found, or if activities are not found
    """
    # Get the dataframe
    df = event_log.to_dataframe()
    
    # Filter by case_id if provided
    if case_id is not None:
        if case_id not in df[event_log.case_id_column].values:
            raise ValueError(f"Case ID '{case_id}' not found in event log")
        df = df[df[event_log.case_id_column] == case_id]
    
    # Time conversion factors
    time_factors = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
        "days": 86400
    }
    factor = time_factors[unit]
    
    # Vectorized calculation
    # Group the dataframe by case_id
    case_id_col = event_log.case_id_column
    ts_col = event_log.timestamp_column
    act_col = event_log.activity_column
    
    # Sort the dataframe in one go
    df = df.sort_values([case_id_col, ts_col])
    
    # Get the first and last event for each case
    if start_activity is None and end_activity is None:
        # Simple case: first and last event of each case
        case_first_last = df.groupby(case_id_col)[ts_col].agg(['first', 'last']).reset_index()
        cycle_times = (case_first_last['last'] - case_first_last['first']).dt.total_seconds() / factor
        
    else:
        # We need custom start/end points
        cycle_times = []
        
        # This part is harder to vectorize due to custom activities
        for case_id, case_df in df.groupby(case_id_col):
            # Find start time
            if start_activity is not None:
                start_events = case_df[case_df[act_col] == start_activity]
                if len(start_events) == 0:
                    continue
                start_time = start_events.iloc[0][ts_col]
            else:
                start_time = case_df.iloc[0][ts_col]
            
            # Find end time
            if end_activity is not None:
                end_events = case_df[case_df[act_col] == end_activity]
                if len(end_events) == 0:
                    continue
                end_time = end_events.iloc[-1][ts_col]
            else:
                end_time = case_df.iloc[-1][ts_col]
            
            # Calculate duration
            duration = (end_time - start_time).total_seconds() / factor
            cycle_times.append(duration)
    
    if len(cycle_times) == 0:
        raise ValueError("No valid cases found matching the criteria")
    
    # Convert to numpy array if it's not already
    if isinstance(cycle_times, pd.Series):
        cycle_times = cycle_times.to_numpy()
    
    if include_stats:
        return {
            "mean": np.mean(cycle_times),
            "median": np.median(cycle_times),
            "min": np.min(cycle_times),
            "max": np.max(cycle_times),
            "std": np.std(cycle_times),
            "count": len(cycle_times)
        }
    
    return np.mean(cycle_times)


@lru_cache(maxsize=32)
def calculate_waiting_time(
    event_log: EventLog,
    activity: str,
    case_id: Optional[str] = None,
    unit: Literal["seconds", "minutes", "hours", "days"] = "seconds",
    include_stats: bool = False,
) -> Union[float, Dict[str, float]]:
    """
    Calculate waiting time before a specific activity.
    
    Args:
        event_log: The event log to analyze
        activity: The activity to calculate waiting time for
        case_id: Optional specific case ID to analyze
        unit: Time unit for the result
        include_stats: Whether to include additional statistics
    
    Returns:
        If include_stats is False, returns the mean waiting time.
        If include_stats is True, returns a dictionary with mean, median, min, max,
        and standard deviation of waiting times.
        
    Raises:
        ValueError: If activity is not found, or if case_id is provided but not found
    """
    # Get the dataframe
    df = event_log.to_dataframe()
    
    # Check if activity exists
    if activity not in df[event_log.activity_column].values:
        raise ValueError(f"Activity '{activity}' not found in event log")
    
    # Filter by case_id if provided
    if case_id is not None:
        if case_id not in df[event_log.case_id_column].values:
            raise ValueError(f"Case ID '{case_id}' not found in event log")
        df = df[df[event_log.case_id_column] == case_id]
    
    # Time conversion factors
    time_factors = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
        "days": 86400
    }
    factor = time_factors[unit]
    
    # Vectorized implementation
    case_id_col = event_log.case_id_column
    ts_col = event_log.timestamp_column
    act_col = event_log.activity_column
    
    # Sort all events by case ID and timestamp
    df = df.sort_values([case_id_col, ts_col])
    
    # Process each case
    waiting_times = []
    
    # This is challenging to fully vectorize due to the need to find preceding events
    # for each activity occurrence - we'll use a grouped approach with shift()
    for case_id, case_df in df.groupby(case_id_col):
        # Mark the target activities
        case_df['is_target'] = case_df[act_col] == activity
        
        # Create a column with the previous timestamp
        case_df['prev_timestamp'] = case_df[ts_col].shift(1)
        
        # Filter to only target activities that have a preceding event
        filtered = case_df[case_df['is_target'] & case_df['prev_timestamp'].notna()]
        
        if not filtered.empty:
            # Calculate all waiting times at once
            times = (filtered[ts_col] - filtered['prev_timestamp']).dt.total_seconds() / factor
            waiting_times.extend(times.tolist())
    
    if not waiting_times:
        raise ValueError(f"No waiting times could be calculated for activity '{activity}'")
    
    # Convert to numpy array
    waiting_times_array = np.array(waiting_times)
    
    if include_stats:
        return {
            "mean": np.mean(waiting_times_array),
            "median": np.median(waiting_times_array),
            "min": np.min(waiting_times_array),
            "max": np.max(waiting_times_array),
            "std": np.std(waiting_times_array),
            "count": len(waiting_times_array)
        }
    
    return np.mean(waiting_times_array)


@lru_cache(maxsize=32)
def calculate_processing_time(
    event_log: EventLog,
    activity: str,
    case_id: Optional[str] = None,
    unit: Literal["seconds", "minutes", "hours", "days"] = "seconds",
    include_stats: bool = False,
    skip_last_events: bool = True,
) -> Union[float, Dict[str, float]]:
    """
    Calculate processing time for a specific activity (time until next activity).
    
    Args:
        event_log: The event log to analyze
        activity: The activity to calculate processing time for
        case_id: Optional specific case ID to analyze
        unit: Time unit for the result
        include_stats: Whether to include additional statistics
        skip_last_events: Whether to skip activity occurrences that are the last in a case
    
    Returns:
        If include_stats is False, returns the mean processing time.
        If include_stats is True, returns a dictionary with mean, median, min, max,
        and standard deviation of processing times.
        
    Raises:
        ValueError: If activity is not found, or if case_id is provided but not found
    """
    # Get the dataframe
    df = event_log.to_dataframe()
    
    # Check if activity exists
    if activity not in df[event_log.activity_column].values:
        raise ValueError(f"Activity '{activity}' not found in event log")
    
    # Filter by case_id if provided
    if case_id is not None:
        if case_id not in df[event_log.case_id_column].values:
            raise ValueError(f"Case ID '{case_id}' not found in event log")
        df = df[df[event_log.case_id_column] == case_id]
    
    # Time conversion factors
    time_factors = {
        "seconds": 1,
        "minutes": 60,
        "hours": 3600,
        "days": 86400
    }
    factor = time_factors[unit]
    
    # Vectorized implementation
    case_id_col = event_log.case_id_column
    ts_col = event_log.timestamp_column
    act_col = event_log.activity_column
    
    # Sort all events by case ID and timestamp
    df = df.sort_values([case_id_col, ts_col])
    
    # Process each case with vectorized operations
    processing_times = []
    
    # Similar approach to waiting time but looking at next events instead of previous
    for case_id, case_df in df.groupby(case_id_col):
        # Mark the target activities
        case_df['is_target'] = case_df[act_col] == activity
        
        # Create a column with the next timestamp using shift(-1)
        case_df['next_timestamp'] = case_df[ts_col].shift(-1)
        
        # Filter to only target activities
        target_activities = case_df[case_df['is_target']]
        
        if not target_activities.empty:
            # For each target activity:
            # - If it has a next event (not the last in case), include it
            # - If it's the last in case and skip_last_events=False, don't include it
            if skip_last_events:
                # Only include activities with a valid next timestamp
                filtered = target_activities[target_activities['next_timestamp'].notna()]
            else:
                # Include all target activities with next timestamps
                filtered = target_activities[target_activities['next_timestamp'].notna()]
                # Could add custom handling for the last events here if needed
            
            if not filtered.empty:
                # Calculate all processing times at once
                times = (filtered['next_timestamp'] - filtered[ts_col]).dt.total_seconds() / factor
                processing_times.extend(times.tolist())
    
    if not processing_times:
        raise ValueError(f"No processing times could be calculated for activity '{activity}'")
    
    # Convert to numpy array
    processing_times_array = np.array(processing_times)
    
    if include_stats:
        return {
            "mean": np.mean(processing_times_array),
            "median": np.median(processing_times_array),
            "min": np.min(processing_times_array),
            "max": np.max(processing_times_array),
            "std": np.std(processing_times_array),
            "count": len(processing_times_array)
        }
    
    return np.mean(processing_times_array)