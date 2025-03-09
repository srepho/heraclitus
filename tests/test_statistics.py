"""
Tests for statistics module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from heraclitus.data import EventLog
from heraclitus.statistics import compare_cycle_times, bottleneck_analysis, fit_distribution


@pytest.fixture
def sample_event_log_with_groups():
    """Create a sample EventLog with group attributes for testing."""
    # Generate sample data with two groups: "fast" and "slow"
    np.random.seed(42)  # For reproducibility
    
    case_ids = []
    activities = []
    timestamps = []
    groups = []
    resources = []
    
    # Create 20 "fast" cases - shorter cycle times
    base_time = datetime(2023, 1, 1, 10, 0)
    for i in range(1, 21):
        case_id = f"fast{i}"
        
        # Start activity
        case_ids.append(case_id)
        activities.append("start")
        timestamps.append(base_time + timedelta(hours=i))
        groups.append("fast")
        resources.append("system")
        
        # Process activity
        case_ids.append(case_id)
        activities.append("process")
        timestamps.append(base_time + timedelta(hours=i, minutes=np.random.randint(10, 20)))
        groups.append("fast")
        resources.append("user1")
        
        # End activity
        case_ids.append(case_id)
        activities.append("end")
        timestamps.append(base_time + timedelta(hours=i, minutes=np.random.randint(25, 35)))
        groups.append("fast")
        resources.append("user2")
    
    # Create 20 "slow" cases - longer cycle times
    for i in range(1, 21):
        case_id = f"slow{i}"
        
        # Start activity
        case_ids.append(case_id)
        activities.append("start")
        timestamps.append(base_time + timedelta(days=1, hours=i))
        groups.append("slow")
        resources.append("system")
        
        # Process activity
        case_ids.append(case_id)
        activities.append("process")
        timestamps.append(base_time + timedelta(days=1, hours=i, minutes=np.random.randint(20, 40)))
        groups.append("slow")
        resources.append("user3")
        
        # End activity
        case_ids.append(case_id)
        activities.append("end")
        timestamps.append(base_time + timedelta(days=1, hours=i, minutes=np.random.randint(45, 75)))
        groups.append("slow")
        resources.append("user4")
    
    df = pd.DataFrame({
        "case_id": case_ids,
        "activity": activities,
        "timestamp": timestamps,
        "group": groups,
        "resource": resources
    })
    
    return EventLog(df)


def test_compare_cycle_times(sample_event_log_with_groups):
    """Test comparing cycle times between groups."""
    # Test parametric comparison
    result = compare_cycle_times(
        sample_event_log_with_groups,
        group_by_attribute="group",
        unit="minutes",
        test_type="parametric"
    )
    
    # Check basic structure of the result
    assert "groups" in result
    assert "sample_sizes" in result
    assert "means" in result
    assert "medians" in result
    assert "test_name" in result
    assert "statistic" in result
    assert "p_value" in result
    assert "significant" in result
    
    # Check that we have two groups
    assert len(result["groups"]) == 2
    assert "fast" in result["groups"]
    assert "slow" in result["groups"]
    
    # The slow group should have higher mean cycle time
    assert result["means"]["slow"] > result["means"]["fast"]
    
    # The difference should be significant
    assert result["significant"] is True
    assert result["p_value"] < 0.05
    
    # Test non-parametric comparison
    nonparam_result = compare_cycle_times(
        sample_event_log_with_groups,
        group_by_attribute="group",
        unit="minutes",
        test_type="non_parametric"
    )
    
    # Should still show significant difference
    assert nonparam_result["significant"] is True
    assert nonparam_result["p_value"] < 0.05
    
    # Test with specific activities
    activity_result = compare_cycle_times(
        sample_event_log_with_groups,
        group_by_attribute="group",
        start_activity="process",
        end_activity="end",
        unit="minutes"
    )
    
    # Should still show significant difference
    assert activity_result["significant"] is True


def test_bottleneck_analysis(sample_event_log_with_groups):
    """Test bottleneck analysis."""
    # Test waiting time method
    waiting_result = bottleneck_analysis(
        sample_event_log_with_groups,
        method="waiting_time",
        unit="minutes"
    )
    
    # Check basic structure
    assert "method" in waiting_result
    assert "bottlenecks" in waiting_result
    assert "metrics" in waiting_result
    assert waiting_result["method"] == "waiting_time"
    
    # Check that we have bottlenecks identified
    assert len(waiting_result["bottlenecks"]) > 0
    
    # Test processing time method
    proc_result = bottleneck_analysis(
        sample_event_log_with_groups,
        method="processing_time",
        unit="minutes"
    )
    
    assert proc_result["method"] == "processing_time"
    assert len(proc_result["bottlenecks"]) > 0
    
    # Test frequency method
    freq_result = bottleneck_analysis(
        sample_event_log_with_groups,
        method="frequency"
    )
    
    assert freq_result["method"] == "frequency"
    assert len(freq_result["bottlenecks"]) > 0
    
    # Activities should appear in all three results
    for activity in ["start", "process", "end"]:
        assert activity in freq_result["metrics"]


def test_fit_distribution(sample_event_log_with_groups):
    """Test distribution fitting."""
    # Test exponential distribution fitting for cycle times
    exp_result = fit_distribution(
        sample_event_log_with_groups,
        data_type="cycle_time",
        distribution="exponential",
        unit="minutes"
    )
    
    # Check basic structure
    assert "distribution" in exp_result
    assert "params" in exp_result
    assert "sse" in exp_result
    assert "data" in exp_result
    assert "data_summary" in exp_result
    assert exp_result["distribution"] == "exponential"
    
    # Test normal distribution fitting
    norm_result = fit_distribution(
        sample_event_log_with_groups,
        data_type="cycle_time",
        distribution="normal",
        unit="minutes"
    )
    
    assert norm_result["distribution"] == "normal"
    assert "mean" in norm_result["params"]
    assert "std" in norm_result["params"]
    
    # Test fitting waiting time distribution
    wait_result = fit_distribution(
        sample_event_log_with_groups,
        data_type="waiting_time",
        activity="process",
        distribution="lognormal",
        unit="minutes"
    )
    
    assert wait_result["distribution"] == "lognormal"
    assert len(wait_result["data"]) > 0
    
    # Test with different activity
    proc_result = fit_distribution(
        sample_event_log_with_groups,
        data_type="processing_time",
        activity="process",
        distribution="gamma",
        unit="minutes"
    )
    
    assert proc_result["distribution"] == "gamma"
    assert len(proc_result["data"]) > 0


def test_statistics_error_handling():
    """Test error handling in statistics functions."""
    # Create a simple event log with no group attribute
    df = pd.DataFrame({
        "case_id": ["case1", "case1", "case2", "case2"],
        "activity": ["start", "end", "start", "end"],
        "timestamp": [
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 1, 11, 0),
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 1, 13, 0)
        ]
    })
    event_log = EventLog(df)
    
    # Test error when attribute doesn't exist
    with pytest.raises(ValueError):
        compare_cycle_times(event_log, group_by_attribute="non_existent")
    
    # Test error when activity doesn't exist
    with pytest.raises(ValueError):
        fit_distribution(
            event_log,
            data_type="waiting_time",
            activity="non_existent"
        )
    
    # Test error when missing required parameter
    with pytest.raises(ValueError):
        fit_distribution(
            event_log,
            data_type="waiting_time",
            distribution="normal"
            # Missing activity parameter
        )