"""
Tests for the EventLog class.
"""
import pytest
import pandas as pd
from datetime import datetime

from heraclitus.data import EventLog


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "case_id": ["case1", "case1", "case1", "case2", "case2"],
        "activity": ["start", "process", "end", "start", "end"],
        "timestamp": [
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 1, 10, 15),
            datetime(2023, 1, 1, 10, 30),
            datetime(2023, 1, 1, 11, 0),
            datetime(2023, 1, 1, 11, 15)
        ],
        "resource": ["system", "user1", "user2", "system", "user2"],
        "cost": [0, 100, 50, 0, 50]
    })


def test_event_log_creation(sample_df):
    """Test creating an EventLog from a DataFrame."""
    event_log = EventLog(sample_df)
    
    assert event_log.case_id_column == "case_id"
    assert event_log.activity_column == "activity"
    assert event_log.timestamp_column == "timestamp"
    assert set(event_log.attribute_columns) == {"resource", "cost"}
    assert len(event_log) == 5
    assert event_log.case_count() == 2
    assert event_log.activity_count() == 3


def test_event_log_filter_cases(sample_df):
    """Test filtering EventLog by case IDs."""
    event_log = EventLog(sample_df)
    filtered_log = event_log.filter_cases(["case1"])
    
    assert len(filtered_log) == 3
    assert filtered_log.case_count() == 1
    assert "case2" not in filtered_log.to_dataframe()[filtered_log.case_id_column].values


def test_event_log_filter_activities(sample_df):
    """Test filtering EventLog by activities."""
    event_log = EventLog(sample_df)
    filtered_log = event_log.filter_activities(["start", "end"])
    
    assert len(filtered_log) == 4
    assert "process" not in filtered_log.to_dataframe()[filtered_log.activity_column].values


def test_event_log_filter_time_range(sample_df):
    """Test filtering EventLog by time range."""
    event_log = EventLog(sample_df)
    
    # Filter from 10:10 to 11:10
    start_time = pd.Timestamp("2023-01-01 10:10:00")
    end_time = pd.Timestamp("2023-01-01 11:10:00")
    
    filtered_log = event_log.filter_time_range(start_time, end_time)
    
    assert len(filtered_log) == 3
    assert filtered_log.to_dataframe()[filtered_log.timestamp_column].min() >= start_time
    assert filtered_log.to_dataframe()[filtered_log.timestamp_column].max() <= end_time


def test_event_log_add_attribute(sample_df):
    """Test adding a new attribute."""
    event_log = EventLog(sample_df)
    
    # Add a priority attribute
    priorities = [1, 2, 3, 1, 2]
    event_log.add_attribute("priority", priorities)
    
    assert "priority" in event_log.get_attributes()
    assert "priority" in event_log.to_dataframe().columns
    assert list(event_log.to_dataframe()["priority"]) == priorities


def test_event_log_add_attribute_error(sample_df):
    """Test error handling when adding an attribute with incorrect length."""
    event_log = EventLog(sample_df)
    
    # Try to add an attribute with wrong length
    priorities = [1, 2, 3]  # Only 3 values for 5 rows
    
    with pytest.raises(ValueError):
        event_log.add_attribute("priority", priorities)


def test_event_log_get_unique_values(sample_df):
    """Test getting unique values from EventLog."""
    event_log = EventLog(sample_df)
    
    activities = event_log.get_unique_activities()
    cases = event_log.get_unique_cases()
    
    assert set(activities) == {"start", "process", "end"}
    assert set(cases) == {"case1", "case2"}