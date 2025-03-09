"""
Statistical comparison functions for process analysis.
"""
from typing import Dict, List, Optional, Union, Literal, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats

from heraclitus.data import EventLog
from heraclitus.metrics import calculate_cycle_time


def compare_cycle_times(
    event_log: EventLog,
    group_by_attribute: str,
    start_activity: Optional[str] = None,
    end_activity: Optional[str] = None,
    unit: Literal["seconds", "minutes", "hours", "days"] = "seconds",
    test_type: Literal["parametric", "non_parametric"] = "parametric",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compares cycle times between groups defined by an attribute.
    
    Args:
        event_log: The event log to analyze
        group_by_attribute: Attribute to group cases by
        start_activity: Optional specific start activity
        end_activity: Optional specific end activity
        unit: Time unit for the result
        test_type: Type of statistical test to use
        alpha: Significance level
    
    Returns:
        A dictionary containing test results, including:
        - groups: List of group names
        - sample_sizes: Sample size for each group
        - means: Mean cycle time for each group
        - medians: Median cycle time for each group
        - test_name: Name of the statistical test used
        - statistic: Test statistic
        - p_value: P-value of the test
        - significant: Whether the difference is significant
    
    Raises:
        ValueError: If group_by_attribute doesn't exist or has fewer than 2 groups
    """
    # Get the dataframe
    df = event_log.to_dataframe()
    
    # Check if group_by_attribute exists
    if group_by_attribute not in df.columns:
        raise ValueError(f"Attribute '{group_by_attribute}' not found in event log")
    
    # Get unique groups
    groups = df[group_by_attribute].unique()
    
    if len(groups) < 2:
        raise ValueError(
            f"Need at least 2 groups for comparison, but '{group_by_attribute}' has only {len(groups)}"
        )
    
    # Calculate cycle times for each case
    case_data = []
    for case_id in df[event_log.case_id_column].unique():
        case_df = df[df[event_log.case_id_column] == case_id]
        
        # Get the group for this case
        if group_by_attribute in case_df.columns:
            group = case_df.iloc[0][group_by_attribute]
        else:
            # Skip cases without group information
            continue
        
        try:
            # Calculate cycle time for this case
            cycle_time = calculate_cycle_time(
                EventLog(case_df, 
                         case_id_column=event_log.case_id_column,
                         activity_column=event_log.activity_column,
                         timestamp_column=event_log.timestamp_column),
                start_activity=start_activity,
                end_activity=end_activity,
                unit=unit
            )
            
            case_data.append({
                "case_id": case_id,
                "group": group,
                "cycle_time": cycle_time
            })
        except ValueError:
            # Skip cases where cycle time couldn't be calculated
            continue
    
    if not case_data:
        raise ValueError("No valid cycle times could be calculated")
    
    # Create a DataFrame of case data
    case_df = pd.DataFrame(case_data)
    
    # Check if we have enough data
    group_counts = case_df.groupby("group").size()
    valid_groups = group_counts[group_counts >= 2].index.tolist()
    
    if len(valid_groups) < 2:
        raise ValueError(
            f"Need at least 2 groups with sufficient data, but only found {len(valid_groups)}"
        )
    
    # Filter to valid groups
    case_df = case_df[case_df["group"].isin(valid_groups)]
    
    # Calculate statistics for each group
    group_stats = {}
    for group in valid_groups:
        group_cycle_times = case_df[case_df["group"] == group]["cycle_time"]
        group_stats[group] = {
            "n": len(group_cycle_times),
            "mean": group_cycle_times.mean(),
            "median": group_cycle_times.median(),
            "std": group_cycle_times.std()
        }
    
    # Perform statistical test
    if len(valid_groups) == 2:
        # Two groups: t-test or Mann-Whitney U test
        group1, group2 = valid_groups[0], valid_groups[1]
        data1 = case_df[case_df["group"] == group1]["cycle_time"]
        data2 = case_df[case_df["group"] == group2]["cycle_time"]
        
        if test_type == "parametric":
            # Independent t-test
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            test_name = "Independent t-test (Welch's t-test)"
            
            # Calculate effect size (Cohen's d)
            mean1, mean2 = data1.mean(), data2.mean()
            std1, std2 = data1.std(), data2.std()
            n1, n2 = len(data1), len(data2)
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            effect_size = abs(mean1 - mean2) / pooled_std
            effect_size_name = "Cohen's d"
            
        else:
            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(data1, data2)
            test_name = "Mann-Whitney U test"
            
            # Calculate effect size (r = Z / sqrt(N))
            n_total = len(data1) + len(data2)
            z_score = stats.norm.ppf(1 - p_value / 2)  # Two-tailed p-value to z-score
            effect_size = abs(z_score) / np.sqrt(n_total)
            effect_size_name = "r (Z / sqrt(N))"
            
    else:
        # More than two groups: ANOVA or Kruskal-Wallis
        if test_type == "parametric":
            # One-way ANOVA
            anova_data = [case_df[case_df["group"] == group]["cycle_time"] for group in valid_groups]
            stat, p_value = stats.f_oneway(*anova_data)
            test_name = "One-way ANOVA"
            
            # Calculate effect size (Eta-squared)
            # This is a simplified calculation
            ss_between = sum((group_data.mean() - case_df["cycle_time"].mean())**2 * len(group_data) 
                           for group_data in anova_data)
            ss_total = sum((x - case_df["cycle_time"].mean())**2 for x in case_df["cycle_time"])
            effect_size = ss_between / ss_total if ss_total != 0 else 0
            effect_size_name = "Eta-squared"
            
        else:
            # Kruskal-Wallis H test
            kruskal_data = [case_df[case_df["group"] == group]["cycle_time"] for group in valid_groups]
            stat, p_value = stats.kruskal(*kruskal_data)
            test_name = "Kruskal-Wallis H test"
            
            # Calculate effect size (Eta-squared)
            # Similar approach as for ANOVA
            n_total = len(case_df)
            effect_size = (stat - len(valid_groups) + 1) / (n_total - len(valid_groups))
            effect_size_name = "Eta-squared (H)"
    
    # Interpret effect size
    if test_type == "parametric" and len(valid_groups) == 2:
        # Cohen's d interpretation
        if effect_size < 0.2:
            effect_interpretation = "Negligible effect"
        elif effect_size < 0.5:
            effect_interpretation = "Small effect"
        elif effect_size < 0.8:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"
    else:
        # Eta-squared or r interpretation
        if effect_size < 0.01:
            effect_interpretation = "Negligible effect"
        elif effect_size < 0.06:
            effect_interpretation = "Small effect"
        elif effect_size < 0.14:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"
    
    # Prepare result dictionary
    result = {
        "groups": valid_groups,
        "sample_sizes": {group: stats["n"] for group, stats in group_stats.items()},
        "means": {group: stats["mean"] for group, stats in group_stats.items()},
        "medians": {group: stats["median"] for group, stats in group_stats.items()},
        "std_devs": {group: stats["std"] for group, stats in group_stats.items()},
        "test_name": test_name,
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "effect_size": effect_size,
        "effect_size_name": effect_size_name,
        "effect_interpretation": effect_interpretation
    }
    
    return result


def bottleneck_analysis(
    event_log: EventLog,
    method: Literal["waiting_time", "processing_time", "frequency"] = "waiting_time",
    unit: Literal["seconds", "minutes", "hours", "days"] = "seconds",
    top_n: int = 3,
) -> Dict[str, Any]:
    """
    Performs bottleneck analysis to identify process bottlenecks.
    
    Args:
        event_log: The event log to analyze
        method: Method to identify bottlenecks
        unit: Time unit for time-based methods
        top_n: Number of top bottlenecks to identify
    
    Returns:
        A dictionary containing bottleneck analysis results:
        - method: Method used for analysis
        - bottlenecks: List of bottleneck activities
        - metrics: Metrics for each activity
    
    Raises:
        ValueError: If an invalid method is specified or no activities are found
    """
    # Get the dataframe
    df = event_log.to_dataframe()
    
    # Get unique activities
    activities = df[event_log.activity_column].unique()
    
    if len(activities) == 0:
        raise ValueError("No activities found in event log")
    
    # Calculate metrics for each activity
    activity_metrics = {}
    
    if method == "frequency":
        # Count frequency of each activity
        activity_counts = df[event_log.activity_column].value_counts()
        total_events = len(df)
        
        for activity, count in activity_counts.items():
            activity_metrics[activity] = {
                "count": count,
                "percentage": count / total_events * 100
            }
        
        # Sort by frequency (higher frequency may indicate bottlenecks)
        sorted_activities = sorted(
            activity_metrics.keys(),
            key=lambda x: activity_metrics[x]["count"],
            reverse=True
        )
        
        # Determine bottlenecks (top N by frequency)
        bottlenecks = sorted_activities[:top_n]
        
    elif method == "waiting_time" or method == "processing_time":
        # Import functions here to avoid circular imports
        from heraclitus.metrics import calculate_waiting_time, calculate_processing_time
        
        # Calculate waiting time or processing time for each activity
        for activity in activities:
            try:
                if method == "waiting_time":
                    time_stats = calculate_waiting_time(
                        event_log, activity, unit=unit, include_stats=True
                    )
                else:  # processing_time
                    time_stats = calculate_processing_time(
                        event_log, activity, unit=unit, include_stats=True
                    )
                
                activity_metrics[activity] = time_stats
                
            except ValueError:
                # Skip activities where metrics couldn't be calculated
                continue
        
        # Sort by mean time (higher time indicates bottlenecks)
        sorted_activities = sorted(
            activity_metrics.keys(),
            key=lambda x: activity_metrics[x]["mean"],
            reverse=True
        )
        
        # Determine bottlenecks (top N by time)
        bottlenecks = sorted_activities[:top_n]
        
    else:
        raise ValueError(f"Invalid method: {method}. Use 'waiting_time', 'processing_time', or 'frequency'")
    
    return {
        "method": method,
        "bottlenecks": bottlenecks,
        "metrics": activity_metrics
    }


def fit_distribution(
    event_log: EventLog,
    data_type: Literal["cycle_time", "waiting_time", "processing_time"] = "cycle_time",
    distribution: Literal["exponential", "normal", "lognormal", "gamma", "weibull"] = "exponential",
    activity: Optional[str] = None,
    unit: Literal["seconds", "minutes", "hours", "days"] = "seconds",
) -> Dict[str, Any]:
    """
    Fits a statistical distribution to process data.
    
    Args:
        event_log: The event log to analyze
        data_type: Type of data to fit distribution to
        distribution: Type of distribution to fit
        activity: Required for waiting_time and processing_time
        unit: Time unit for the analysis
    
    Returns:
        A dictionary containing distribution fitting results:
        - distribution: Name of the distribution
        - params: Estimated distribution parameters
        - sse: Sum of squared errors (goodness of fit)
        - data: Original data points
    
    Raises:
        ValueError: If invalid parameters are provided
    """
    # Import required functions
    from heraclitus.metrics import (
        calculate_cycle_time, 
        calculate_waiting_time, 
        calculate_processing_time
    )
    
    # Get data based on data_type
    if data_type == "cycle_time":
        # Collect cycle times for all cases
        data = []
        df = event_log.to_dataframe()
        for case_id in df[event_log.case_id_column].unique():
            case_df = df[df[event_log.case_id_column] == case_id]
            try:
                cycle_time = calculate_cycle_time(
                    EventLog(case_df, 
                            case_id_column=event_log.case_id_column,
                            activity_column=event_log.activity_column,
                            timestamp_column=event_log.timestamp_column),
                    unit=unit
                )
                data.append(cycle_time)
            except ValueError:
                continue
    
    elif data_type == "waiting_time":
        if activity is None:
            raise ValueError("Activity must be specified for waiting_time analysis")
        
        # Calculate waiting times
        try:
            time_stats = calculate_waiting_time(
                event_log, activity, unit=unit, include_stats=True
            )
            # We need individual data points, not just summary statistics
            # This is a limitation of the current API
            # For now, we'll collect data points manually
            data = []
            df = event_log.to_dataframe()
            df = df.sort_values(by=[event_log.case_id_column, event_log.timestamp_column])
            
            for case_id, case_df in df.groupby(event_log.case_id_column):
                # Find occurrences of the activity
                activity_events = case_df[case_df[event_log.activity_column] == activity]
                
                for _, activity_event in activity_events.iterrows():
                    activity_time = activity_event[event_log.timestamp_column]
                    
                    # Find the event immediately preceding this activity
                    preceding_events = case_df[case_df[event_log.timestamp_column] < activity_time]
                    
                    if not preceding_events.empty:
                        # Get the most recent preceding event
                        preceding_event = preceding_events.iloc[-1]
                        preceding_time = preceding_event[event_log.timestamp_column]
                        
                        # Calculate waiting time
                        time_factors = {
                            "seconds": 1,
                            "minutes": 60,
                            "hours": 3600,
                            "days": 86400
                        }
                        factor = time_factors[unit]
                        waiting_time = (activity_time - preceding_time).total_seconds() / factor
                        data.append(waiting_time)
            
        except ValueError:
            raise ValueError(f"Could not calculate waiting times for activity '{activity}'")
    
    elif data_type == "processing_time":
        if activity is None:
            raise ValueError("Activity must be specified for processing_time analysis")
        
        # Calculate processing times using a similar approach as waiting_time
        data = []
        df = event_log.to_dataframe()
        df = df.sort_values(by=[event_log.case_id_column, event_log.timestamp_column])
        
        for case_id, case_df in df.groupby(event_log.case_id_column):
            # Find occurrences of the activity
            activity_events = case_df[case_df[event_log.activity_column] == activity]
            
            for _, activity_event in activity_events.iterrows():
                activity_time = activity_event[event_log.timestamp_column]
                
                # Find the event immediately following this activity
                following_events = case_df[case_df[event_log.timestamp_column] > activity_time]
                
                if not following_events.empty:
                    # Get the next event
                    following_event = following_events.iloc[0]
                    following_time = following_event[event_log.timestamp_column]
                    
                    # Calculate processing time
                    time_factors = {
                        "seconds": 1,
                        "minutes": 60,
                        "hours": 3600,
                        "days": 86400
                    }
                    factor = time_factors[unit]
                    processing_time = (following_time - activity_time).total_seconds() / factor
                    data.append(processing_time)
    
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    
    if not data:
        raise ValueError(f"No valid data points found for {data_type}")
    
    # Fit the distribution
    if distribution == "exponential":
        # Exponential distribution has one parameter (scale)
        loc, scale = stats.expon.fit(data)
        dist = stats.expon(loc=loc, scale=scale)
        params = {"loc": loc, "scale": scale}
        
    elif distribution == "normal":
        # Normal distribution has two parameters (loc, scale)
        loc, scale = stats.norm.fit(data)
        dist = stats.norm(loc=loc, scale=scale)
        params = {"mean": loc, "std": scale}
        
    elif distribution == "lognormal":
        # Lognormal distribution has two parameters (s, scale)
        s, loc, scale = stats.lognorm.fit(data)
        dist = stats.lognorm(s=s, loc=loc, scale=scale)
        params = {"s": s, "loc": loc, "scale": scale}
        
    elif distribution == "gamma":
        # Gamma distribution has two parameters (a, scale)
        a, loc, scale = stats.gamma.fit(data)
        dist = stats.gamma(a=a, loc=loc, scale=scale)
        params = {"a": a, "loc": loc, "scale": scale}
        
    elif distribution == "weibull":
        # Weibull distribution has two parameters (c, scale)
        c, loc, scale = stats.weibull_min.fit(data)
        dist = stats.weibull_min(c=c, loc=loc, scale=scale)
        params = {"c": c, "loc": loc, "scale": scale}
        
    else:
        raise ValueError(f"Invalid distribution: {distribution}")
    
    # Calculate goodness of fit
    hist, bin_edges = np.histogram(data, bins="auto", density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pdf = dist.pdf(bin_centers)
    sse = np.sum((hist - pdf)**2)
    
    # Calculate Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.kstest(data, dist.cdf)
    
    return {
        "distribution": distribution,
        "params": params,
        "sse": sse,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "data": data,
        "data_summary": {
            "mean": np.mean(data),
            "median": np.median(data),
            "min": np.min(data),
            "max": np.max(data),
            "std": np.std(data),
            "count": len(data)
        }
    }