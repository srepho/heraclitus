"""
Time-based metrics for process analysis.
"""
from typing import Optional, List, Dict, Union, Literal
import pandas as pd
import numpy as np

from heraclitus.data import EventLog


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
    
    # Calculate cycle times for each case
    cycle_times = []
    
    for case_id, case_df in df.groupby(event_log.case_id_column):
        # Sort by timestamp
        case_df = case_df.sort_values(by=event_log.timestamp_column)
        
        if start_activity is not None:
            # Find first occurrence of start_activity
            start_events = case_df[case_df[event_log.activity_column] == start_activity]
            if len(start_events) == 0:
                # Skip this case if start_activity not found
                continue
            start_time = start_events.iloc[0][event_log.timestamp_column]
        else:
            # Use first event as start
            start_time = case_df.iloc[0][event_log.timestamp_column]
        
        if end_activity is not None:
            # Find last occurrence of end_activity
            end_events = case_df[case_df[event_log.activity_column] == end_activity]
            if len(end_events) == 0:
                # Skip this case if end_activity not found
                continue
            end_time = end_events.iloc[-1][event_log.timestamp_column]
        else:
            # Use last event as end
            end_time = case_df.iloc[-1][event_log.timestamp_column]
        
        # Calculate duration in seconds and convert to the desired unit
        duration = (end_time - start_time).total_seconds() / factor
        cycle_times.append(duration)
    
    if not cycle_times:
        raise ValueError("No valid cases found matching the criteria")
    
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
    
    # Calculate waiting times
    waiting_times = []
    
    for case_id, case_df in df.groupby(event_log.case_id_column):
        # Sort by timestamp
        case_df = case_df.sort_values(by=event_log.timestamp_column)
        
        # Find occurrences of the activity
        activity_events = case_df[case_df[event_log.activity_column] == activity]
        
        for idx, activity_event in activity_events.iterrows():
            activity_time = activity_event[event_log.timestamp_column]
            
            # Find the event immediately preceding this activity
            preceding_events = case_df[case_df[event_log.timestamp_column] < activity_time]
            
            if not preceding_events.empty:
                # Get the most recent preceding event
                preceding_event = preceding_events.iloc[-1]
                preceding_time = preceding_event[event_log.timestamp_column]
                
                # Calculate waiting time
                waiting_time = (activity_time - preceding_time).total_seconds() / factor
                waiting_times.append(waiting_time)
    
    if not waiting_times:
        raise ValueError(f"No waiting times could be calculated for activity '{activity}'")
    
    if include_stats:
        return {
            "mean": np.mean(waiting_times),
            "median": np.median(waiting_times),
            "min": np.min(waiting_times),
            "max": np.max(waiting_times),
            "std": np.std(waiting_times),
            "count": len(waiting_times)
        }
    
    return np.mean(waiting_times)


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
    
    # Calculate processing times
    processing_times = []
    
    for case_id, case_df in df.groupby(event_log.case_id_column):
        # Sort by timestamp
        case_df = case_df.sort_values(by=event_log.timestamp_column)
        
        # Find occurrences of the activity
        activity_events = case_df[case_df[event_log.activity_column] == activity]
        
        for idx, activity_event in activity_events.iterrows():
            activity_time = activity_event[event_log.timestamp_column]
            
            # Find the event immediately following this activity
            following_events = case_df[case_df[event_log.timestamp_column] > activity_time]
            
            if not following_events.empty:
                # Get the next event
                following_event = following_events.iloc[0]
                following_time = following_event[event_log.timestamp_column]
                
                # Calculate processing time
                processing_time = (following_time - activity_time).total_seconds() / factor
                processing_times.append(processing_time)
            elif not skip_last_events:
                # This is the last event in the case and we're not skipping last events
                # We could use current time, but for now we'll skip these
                pass
    
    if not processing_times:
        raise ValueError(f"No processing times could be calculated for activity '{activity}'")
    
    if include_stats:
        return {
            "mean": np.mean(processing_times),
            "median": np.median(processing_times),
            "min": np.min(processing_times),
            "max": np.max(processing_times),
            "std": np.std(processing_times),
            "count": len(processing_times)
        }
    
    return np.mean(processing_times)