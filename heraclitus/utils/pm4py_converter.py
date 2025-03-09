"""
Converters between Heraclitus EventLog and PM4PY formats.

This module provides functions to convert Heraclitus EventLog objects
to PM4PY EventLog objects and vice versa, enabling integration with
PM4PY's advanced process mining algorithms.
"""
from typing import Optional, Dict, Any, List, Union, Tuple
import pandas as pd
import numpy as np
import warnings

from heraclitus.data import EventLog

try:
    import pm4py
    from pm4py.objects.log.obj import EventLog as PM4PyEventLog
    from pm4py.objects.log.obj import Trace, Event
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    warnings.warn(
        "PM4PY is not installed. Install with 'pip install heraclitus[pm4py]' "
        "to use PM4PY integration features."
    )


def check_pm4py_available() -> None:
    """
    Check if PM4PY is available.
    
    Raises:
        ImportError: If PM4PY is not installed
    """
    if not PM4PY_AVAILABLE:
        raise ImportError(
            "PM4PY is not installed. Install with 'pip install heraclitus[pm4py]' "
            "to use PM4PY integration features."
        )


def to_pm4py(
    event_log: EventLog,
    additional_attributes: Optional[List[str]] = None,
) -> "PM4PyEventLog":
    """
    Convert a Heraclitus EventLog to a PM4PY EventLog.
    
    Args:
        event_log: The Heraclitus EventLog to convert
        additional_attributes: Optional list of additional attributes to include
            in the PM4PY EventLog
    
    Returns:
        A PM4PY EventLog object
    
    Raises:
        ImportError: If PM4PY is not installed
    """
    check_pm4py_available()
    
    # Convert to pandas DataFrame
    df = event_log.to_dataframe()
    
    # PM4PY expects specific column names
    rename_dict = {
        event_log.case_id_column: 'case:concept:name',
        event_log.activity_column: 'concept:name',
        event_log.timestamp_column: 'time:timestamp'
    }
    
    df_renamed = df.rename(columns=rename_dict)
    
    # Add additional attributes if provided
    if additional_attributes:
        for attr in additional_attributes:
            if attr in df.columns and attr not in rename_dict.keys():
                # Keep the attribute as is
                pass
            elif attr in rename_dict.keys():
                # Skip attributes that are already renamed
                pass
            else:
                # Attribute not found
                warnings.warn(f"Attribute '{attr}' not found in EventLog")
    
    # Use PM4PY's converter to create a PM4PY EventLog
    pm4py_log = pm4py.convert_to_event_log(df_renamed)
    
    return pm4py_log


def from_pm4py(
    pm4py_log: "PM4PyEventLog",
    case_id_key: str = 'case:concept:name',
    activity_key: str = 'concept:name',
    timestamp_key: str = 'time:timestamp',
    additional_attributes: Optional[List[str]] = None
) -> EventLog:
    """
    Convert a PM4PY EventLog to a Heraclitus EventLog.
    
    Args:
        pm4py_log: The PM4PY EventLog to convert
        case_id_key: The key for case ID in the PM4PY EventLog
        activity_key: The key for activity in the PM4PY EventLog
        timestamp_key: The key for timestamp in the PM4PY EventLog
        additional_attributes: Optional list of additional attributes to include
    
    Returns:
        A Heraclitus EventLog object
    
    Raises:
        ImportError: If PM4PY is not installed
        ValueError: If required keys are not found in the PM4PY EventLog
    """
    check_pm4py_available()
    
    # Convert PM4PY log to DataFrame
    df = pm4py.convert_to_dataframe(pm4py_log)
    
    # Check if required keys exist
    for key, name in [
        (case_id_key, 'case ID'),
        (activity_key, 'activity'),
        (timestamp_key, 'timestamp')
    ]:
        if key not in df.columns:
            raise ValueError(f"Required key '{key}' for {name} not found in PM4PY EventLog")
    
    # Rename columns to match Heraclitus format
    rename_dict = {
        case_id_key: 'case_id',
        activity_key: 'activity',
        timestamp_key: 'timestamp'
    }
    
    # Add additional attributes if requested
    if additional_attributes:
        for attr in additional_attributes:
            if attr in df.columns and attr not in rename_dict.keys():
                # Keep the attribute as is
                pass
            else:
                # Attribute not found or already renamed
                warnings.warn(f"Attribute '{attr}' not found in PM4PY EventLog or already included")
    
    # Rename columns
    df_renamed = df.rename(columns=rename_dict)
    
    # Create Heraclitus EventLog
    heraclitus_log = EventLog(
        df_renamed,
        case_id_column='case_id',
        activity_column='activity',
        timestamp_column='timestamp'
    )
    
    return heraclitus_log


def apply_pm4py_algorithm(
    event_log: EventLog,
    algorithm: callable,
    additional_attributes: Optional[List[str]] = None,
    **kwargs
) -> Any:
    """
    Apply a PM4PY algorithm to a Heraclitus EventLog.
    
    Args:
        event_log: The Heraclitus EventLog to analyze
        algorithm: The PM4PY algorithm function to apply
        additional_attributes: Optional list of additional attributes to include
        **kwargs: Additional keyword arguments to pass to the algorithm
    
    Returns:
        The result of applying the PM4PY algorithm
    
    Examples:
        >>> from pm4py.algo.discovery.alpha import algorithm as alpha_miner
        >>> net, im, fm = apply_pm4py_algorithm(event_log, alpha_miner)
    """
    check_pm4py_available()
    
    # Convert to PM4PY format
    pm4py_log = to_pm4py(event_log, additional_attributes)
    
    # Apply the algorithm
    result = algorithm(pm4py_log, **kwargs)
    
    return result