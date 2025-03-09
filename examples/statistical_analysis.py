"""
Example demonstrating statistical analysis features in Heraclitus.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from heraclitus.data import EventLog
from heraclitus.metrics import calculate_cycle_time
from heraclitus.statistics import (
    compare_cycle_times,
    bottleneck_analysis,
    fit_distribution,
)


def generate_sample_data():
    """Generate sample data with two process variants."""
    np.random.seed(42)  # For reproducibility
    
    # Generate data for two process variants: "standard" and "express"
    case_ids = []
    activities = []
    timestamps = []
    variants = []
    priorities = []
    costs = []
    
    base_time = datetime(2023, 1, 1, 8, 0)
    
    # Standard process: Register -> Validate -> Process -> Review -> Complete
    for i in range(1, 31):
        case_id = f"std{i}"
        variant = "standard"
        priority = np.random.choice(["low", "medium", "high"])
        
        # Register
        case_ids.append(case_id)
        activities.append("Register")
        timestamps.append(base_time + timedelta(hours=i))
        variants.append(variant)
        priorities.append(priority)
        costs.append(10)
        
        # Validate (takes 30-60 minutes)
        case_ids.append(case_id)
        activities.append("Validate")
        timestamps.append(base_time + timedelta(hours=i, minutes=np.random.randint(30, 60)))
        variants.append(variant)
        priorities.append(priority)
        costs.append(20)
        
        # Process (takes 60-120 minutes after Validate)
        case_ids.append(case_id)
        activities.append("Process")
        last_time = timestamps[-1]
        timestamps.append(last_time + timedelta(minutes=np.random.randint(60, 120)))
        variants.append(variant)
        priorities.append(priority)
        costs.append(50)
        
        # Review (takes 30-90 minutes after Process)
        case_ids.append(case_id)
        activities.append("Review")
        last_time = timestamps[-1]
        timestamps.append(last_time + timedelta(minutes=np.random.randint(30, 90)))
        variants.append(variant)
        priorities.append(priority)
        costs.append(30)
        
        # Complete (takes 15-30 minutes after Review)
        case_ids.append(case_id)
        activities.append("Complete")
        last_time = timestamps[-1]
        timestamps.append(last_time + timedelta(minutes=np.random.randint(15, 30)))
        variants.append(variant)
        priorities.append(priority)
        costs.append(10)
    
    # Express process: Register -> Process -> Complete (faster)
    for i in range(1, 31):
        case_id = f"exp{i}"
        variant = "express"
        priority = np.random.choice(["low", "medium", "high"])
        
        # Register
        case_ids.append(case_id)
        activities.append("Register")
        timestamps.append(base_time + timedelta(days=3, hours=i))  # Later timeframe
        variants.append(variant)
        priorities.append(priority)
        costs.append(20)  # Higher initial cost
        
        # Process (takes 30-60 minutes, faster than standard)
        case_ids.append(case_id)
        activities.append("Process")
        last_time = timestamps[-1]
        timestamps.append(last_time + timedelta(minutes=np.random.randint(30, 60)))
        variants.append(variant)
        priorities.append(priority)
        costs.append(80)  # Higher processing cost
        
        # Complete (takes 10-20 minutes, faster than standard)
        case_ids.append(case_id)
        activities.append("Complete")
        last_time = timestamps[-1]
        timestamps.append(last_time + timedelta(minutes=np.random.randint(10, 20)))
        variants.append(variant)
        priorities.append(priority)
        costs.append(20)
    
    # Create DataFrame
    df = pd.DataFrame({
        "case_id": case_ids,
        "activity": activities,
        "timestamp": timestamps,
        "variant": variants,
        "priority": priorities,
        "cost": costs
    })
    
    return df


def analyze_process_variants(event_log):
    """Compare cycle times between process variants."""
    print("Analyzing Process Variants")
    print("==========================")
    
    # Compare cycle times between variants
    result = compare_cycle_times(
        event_log,
        group_by_attribute="variant",
        unit="minutes",
        test_type="parametric"
    )
    
    print(f"Test: {result['test_name']}")
    print(f"p-value: {result['p_value']:.4f}")
    
    if result['significant']:
        print("Result: Significant difference between variants")
    else:
        print("Result: No significant difference between variants")
    
    print("\nCycle Time Statistics (minutes):")
    for group in result['groups']:
        print(f"  {group.capitalize()}:")
        print(f"    Mean: {result['means'][group]:.1f}")
        print(f"    Median: {result['medians'][group]:.1f}")
        print(f"    Sample size: {result['sample_sizes'][group]}")
    
    print(f"\nEffect Size ({result['effect_size_name']}): {result['effect_size']:.4f}")
    print(f"Interpretation: {result['effect_interpretation']}")
    
    # Plot cycle time distributions
    plt.figure(figsize=(10, 6))
    
    # Get cycle times for each variant
    df = event_log.to_dataframe()
    cycle_times = {}
    
    for variant in df['variant'].unique():
        variant_log = event_log.to_dataframe()
        variant_log = variant_log[variant_log['variant'] == variant]
        variant_log = EventLog(variant_log)
        
        times = []
        for case_id in variant_log.get_unique_cases():
            case_log = variant_log.filter_cases([case_id])
            try:
                time = calculate_cycle_time(case_log, unit="minutes")
                times.append(time)
            except ValueError:
                continue
        
        cycle_times[variant] = times
    
    # Create histogram
    plt.hist(
        [cycle_times['standard'], cycle_times['express']], 
        bins=15, 
        alpha=0.7, 
        label=['Standard Process', 'Express Process']
    )
    
    plt.xlabel('Cycle Time (minutes)')
    plt.ylabel('Frequency')
    plt.title('Cycle Time Distribution by Process Variant')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('variant_comparison.png')
    print("Saved plot to 'variant_comparison.png'")


def analyze_bottlenecks(event_log):
    """Identify process bottlenecks."""
    print("\nBottleneck Analysis")
    print("==================")
    
    # Use waiting time method
    wait_result = bottleneck_analysis(
        event_log,
        method="waiting_time",
        unit="minutes",
        top_n=3
    )
    
    print("Top bottlenecks by waiting time:")
    for i, activity in enumerate(wait_result['bottlenecks'], 1):
        stats = wait_result['metrics'][activity]
        print(f"  {i}. {activity}:")
        print(f"     Mean waiting time: {stats['mean']:.1f} minutes")
        print(f"     Median waiting time: {stats['median']:.1f} minutes")
    
    # Use processing time method
    proc_result = bottleneck_analysis(
        event_log,
        method="processing_time",
        unit="minutes",
        top_n=3
    )
    
    print("\nTop bottlenecks by processing time:")
    for i, activity in enumerate(proc_result['bottlenecks'], 1):
        stats = proc_result['metrics'][activity]
        print(f"  {i}. {activity}:")
        print(f"     Mean processing time: {stats['mean']:.1f} minutes")
        print(f"     Median processing time: {stats['median']:.1f} minutes")
    
    # Create a scatter plot of waiting time vs processing time
    plt.figure(figsize=(10, 6))
    
    activities = set(wait_result['metrics'].keys()) & set(proc_result['metrics'].keys())
    x = []  # waiting times
    y = []  # processing times
    labels = []
    sizes = []
    
    for activity in activities:
        if activity in wait_result['metrics'] and activity in proc_result['metrics']:
            x.append(wait_result['metrics'][activity]['mean'])
            y.append(proc_result['metrics'][activity]['mean'])
            labels.append(activity)
            # Size based on frequency
            sizes.append(wait_result['metrics'][activity]['count'] * 20)
    
    plt.scatter(x, y, s=sizes, alpha=0.7)
    
    # Add labels to each point
    for i, label in enumerate(labels):
        plt.annotate(
            label,
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    plt.xlabel('Mean Waiting Time (minutes)')
    plt.ylabel('Mean Processing Time (minutes)')
    plt.title('Activity Analysis: Waiting Time vs Processing Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('bottleneck_analysis.png')
    print("Saved plot to 'bottleneck_analysis.png'")


def analyze_distributions(event_log):
    """Fit distributions to process data."""
    print("\nDistribution Fitting")
    print("===================")
    
    # Fit distribution to cycle times
    cycle_result = fit_distribution(
        event_log,
        data_type="cycle_time",
        distribution="lognormal",
        unit="minutes"
    )
    
    print("Cycle Time Distribution (lognormal):")
    print(f"  Sample size: {cycle_result['data_summary']['count']}")
    print(f"  Mean: {cycle_result['data_summary']['mean']:.1f} minutes")
    print(f"  Parameters: {cycle_result['params']}")
    print(f"  Goodness of fit (KS test p-value): {cycle_result['ks_pvalue']:.4f}")
    
    # Fit distribution to waiting times for Process activity
    wait_result = fit_distribution(
        event_log,
        data_type="waiting_time",
        activity="Process",
        distribution="gamma",
        unit="minutes"
    )
    
    print("\nWaiting Time Distribution for 'Process' (gamma):")
    print(f"  Sample size: {wait_result['data_summary']['count']}")
    print(f"  Mean: {wait_result['data_summary']['mean']:.1f} minutes")
    print(f"  Parameters: {wait_result['params']}")
    print(f"  Goodness of fit (KS test p-value): {wait_result['ks_pvalue']:.4f}")
    
    # Plot cycle time distribution
    plt.figure(figsize=(10, 6))
    
    # Get fitted distribution
    data = cycle_result['data']
    distribution = cycle_result['distribution']
    params = cycle_result['params']
    
    # Plot histogram
    plt.hist(data, bins=20, density=True, alpha=0.7, label='Observed Data')
    
    # Plot fitted distribution
    x = np.linspace(min(data), max(data), 1000)
    if distribution == 'lognormal':
        y = stats.lognorm.pdf(x, s=params['s'], loc=params['loc'], scale=params['scale'])
    elif distribution == 'normal':
        y = stats.norm.pdf(x, loc=params['mean'], scale=params['std'])
    elif distribution == 'exponential':
        y = stats.expon.pdf(x, loc=params['loc'], scale=params['scale'])
    elif distribution == 'gamma':
        y = stats.gamma.pdf(x, a=params['a'], loc=params['loc'], scale=params['scale'])
    
    plt.plot(x, y, 'r-', lw=2, label=f'Fitted {distribution.capitalize()} Distribution')
    
    plt.xlabel('Cycle Time (minutes)')
    plt.ylabel('Density')
    plt.title(f'Cycle Time Distribution with Fitted {distribution.capitalize()} Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('distribution_fit.png')
    print("Saved plot to 'distribution_fit.png'")


if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data()
    print(f"Generated sample data with {len(df)} events")
    
    # Create EventLog
    event_log = EventLog(df)
    print(f"Event log contains {event_log.case_count()} cases")
    
    # Analyze process variants
    analyze_process_variants(event_log)
    
    # Analyze bottlenecks
    analyze_bottlenecks(event_log)
    
    # Analyze distributions
    try:
        from scipy import stats
        analyze_distributions(event_log)
    except ImportError:
        print("\nDistribution analysis requires scipy")
        print("Install scipy with: pip install scipy")