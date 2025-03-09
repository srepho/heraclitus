"""
Tests for metrics module.
"""
import pytest
import pandas as pd
from datetime import datetime

from heraclitus.data import EventLog
from heraclitus.metrics import (
    calculate_cycle_time,
    calculate_waiting_time,
    calculate_processing_time,
)


@pytest.fixture
def sample_event_log():
    """Create a sample EventLog for testing."""
    df = pd.DataFrame({
        "case_id": ["case1", "case1", "case1", "case2", "case2", "case2"],
        "activity": ["start", "process", "end", "start", "process", "end"],
        "timestamp": [
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 1, 10, 15),
            datetime(2023, 1, 1, 10, 30),
            datetime(2023, 1, 1, 11, 0),
            datetime(2023, 1, 1, 11, 30),
            datetime(2023, 1, 1, 12, 0)
        ]
    })
    return EventLog(df)


def test_calculate_cycle_time(sample_event_log):
    """Test calculating cycle time."""
    # Test average cycle time across all cases
    cycle_time = calculate_cycle_time(sample_event_log, unit="minutes")
    assert cycle_time == 45.0  # (30 min for case1 + 60 min for case2) / 2
    
    # Test cycle time for specific case
    case1_time = calculate_cycle_time(sample_event_log, case_id="case1", unit="minutes")
    assert case1_time == 30.0
    
    # Test cycle time between specific activities
    activity_time = calculate_cycle_time(
        sample_event_log, 
        start_activity="start", 
        end_activity="end", 
        unit="minutes"
    )
    assert activity_time == 45.0
    
    # Test with detailed stats
    stats = calculate_cycle_time(sample_event_log, unit="minutes", include_stats=True)
    assert isinstance(stats, dict)
    assert "mean" in stats
    assert "median" in stats
    assert "min" in stats
    assert "max" in stats
    assert stats["min"] == 30.0
    assert stats["max"] == 60.0


def test_calculate_waiting_time(sample_event_log):
    """Test calculating waiting time."""
    # Test waiting time for 'process' activity
    wait_time = calculate_waiting_time(sample_event_log, "process", unit="minutes")
    assert wait_time == 22.5  # (15 min for case1 + 30 min for case2) / 2
    
    # Test waiting time for specific case
    case2_wait = calculate_waiting_time(
        sample_event_log, "process", case_id="case2", unit="minutes"
    )
    assert case2_wait == 30.0
    
    # Test with detailed stats
    stats = calculate_waiting_time(
        sample_event_log, "process", unit="minutes", include_stats=True
    )
    assert isinstance(stats, dict)
    assert stats["min"] == 15.0
    assert stats["max"] == 30.0


def test_calculate_processing_time(sample_event_log):
    """Test calculating processing time."""
    # Test processing time for 'process' activity
    proc_time = calculate_processing_time(sample_event_log, "process", unit="minutes")
    assert proc_time == 22.5  # (15 min for case1 + 30 min for case2) / 2
    
    # Test processing time for specific case
    case1_proc = calculate_processing_time(
        sample_event_log, "process", case_id="case1", unit="minutes"
    )
    assert case1_proc == 15.0
    
    # Test with detailed stats
    stats = calculate_processing_time(
        sample_event_log, "process", unit="minutes", include_stats=True
    )
    assert isinstance(stats, dict)
    assert stats["min"] == 15.0
    assert stats["max"] == 30.0


def test_error_handling():
    """Test error handling in metrics functions."""
    # Create an empty event log
    empty_df = pd.DataFrame({
        "case_id": [],
        "activity": [],
        "timestamp": []
    })
    empty_log = EventLog(empty_df)
    
    # Test with non-existent case_id
    with pytest.raises(ValueError):
        calculate_cycle_time(sample_event_log, case_id="non_existent")
    
    # Test with non-existent activity
    with pytest.raises(ValueError):
        calculate_waiting_time(sample_event_log, "non_existent")