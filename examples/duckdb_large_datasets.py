"""
Example demonstrating DuckDB integration for large datasets.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from heraclitus.data import EventLog, DuckDBConnector, eventlog_to_duckdb
from heraclitus.visualization import create_interactive_process_map, plot_activity_frequency


def generate_large_dataset(num_cases=1000, output_file="large_dataset.csv"):
    """
    Generate a large process dataset and save it to CSV.
    
    Args:
        num_cases: Number of cases to generate
        output_file: Path to save the CSV file
    """
    print(f"Generating dataset with {num_cases} cases...")
    np.random.seed(42)  # For reproducibility
    
    # Define workflow stages
    stages = [
        "Receive",
        "Validate",
        "Assess",
        "Approve",
        "Prepare",
        "Execute",
        "Verify",
        "Close"
    ]
    
    # Define process paths with probabilities
    paths = [
        {"path": stages, "probability": 0.6},  # Full path
        {"path": ["Receive", "Validate", "Assess", "Reject"], "probability": 0.15},  # Rejection after assessment
        {"path": ["Receive", "Validate", "Assess", "Approve", "Prepare", "Execute", "Rework", "Execute", "Verify", "Close"], 
         "probability": 0.15},  # Rework path
        {"path": ["Receive", "Validate", "Expedite", "Execute", "Verify", "Close"], 
         "probability": 0.1}  # Expedited path
    ]
    
    # Generate data
    case_ids = []
    activities = []
    timestamps = []
    resources = []
    costs = []
    categories = []
    priorities = []
    
    base_time = datetime(2023, 1, 1)
    
    for case_num in range(1, num_cases + 1):
        case_id = f"C{case_num:06d}"
        
        # Select path based on probability
        path_rand = np.random.random()
        cum_prob = 0
        selected_path = None
        
        for path in paths:
            cum_prob += path["probability"]
            if path_rand <= cum_prob:
                selected_path = path["path"]
                break
        
        # Assign case category and priority
        category = np.random.choice(["Type A", "Type B", "Type C", "Type D"], p=[0.3, 0.3, 0.3, 0.1])
        priority = np.random.choice(["Low", "Medium", "High", "Critical"], p=[0.2, 0.5, 0.2, 0.1])
        
        # Process the case
        current_time = base_time + timedelta(hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))
        
        for activity in selected_path:
            case_ids.append(case_id)
            activities.append(activity)
            timestamps.append(current_time)
            
            # Resource assignment
            resource = f"User{np.random.randint(1, 21):02d}"
            resources.append(resource)
            
            # Cost assignment based on activity
            if activity in ["Receive", "Validate", "Close"]:
                cost = np.random.randint(10, 50)
            elif activity in ["Assess", "Verify"]:
                cost = np.random.randint(50, 150)
            elif activity in ["Approve", "Reject"]:
                cost = np.random.randint(100, 200)
            elif activity in ["Prepare", "Execute", "Rework"]:
                cost = np.random.randint(200, 500)
            elif activity == "Expedite":
                cost = np.random.randint(300, 600)
            else:
                cost = np.random.randint(20, 100)
            
            costs.append(cost)
            categories.append(category)
            priorities.append(priority)
            
            # Calculate next activity time
            if priority == "Critical":
                time_factor = 0.5  # Faster for critical priority
            elif priority == "High":
                time_factor = 0.8
            elif priority == "Medium":
                time_factor = 1.0
            else:  # Low
                time_factor = 1.5  # Slower for low priority
            
            # Base times for activities
            if activity == "Receive":
                base_minutes = np.random.randint(10, 30)
            elif activity == "Validate":
                base_minutes = np.random.randint(30, 90)
            elif activity == "Assess":
                base_minutes = np.random.randint(60, 180)
            elif activity in ["Approve", "Reject"]:
                base_minutes = np.random.randint(30, 120)
            elif activity == "Prepare":
                base_minutes = np.random.randint(60, 240)
            elif activity == "Execute":
                base_minutes = np.random.randint(120, 360)
            elif activity == "Rework":
                base_minutes = np.random.randint(60, 180)
            elif activity == "Verify":
                base_minutes = np.random.randint(30, 90)
            elif activity == "Close":
                base_minutes = np.random.randint(15, 45)
            elif activity == "Expedite":
                base_minutes = np.random.randint(30, 60)
            else:
                base_minutes = np.random.randint(30, 120)
            
            # Calculate next time
            wait_minutes = int(base_minutes * time_factor)
            current_time += timedelta(minutes=wait_minutes)
    
    # Create DataFrame
    df = pd.DataFrame({
        "case_id": case_ids,
        "activity": activities,
        "timestamp": timestamps,
        "resource": resources,
        "cost": costs,
        "category": categories,
        "priority": priorities
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved dataset with {len(df)} events to {output_file}")
    
    return df


def main():
    """Run the DuckDB integration example."""
    # Create large dataset if it doesn't exist
    dataset_file = "large_dataset.csv"
    
    if not os.path.exists(dataset_file):
        df = generate_large_dataset(num_cases=5000, output_file=dataset_file)
    else:
        print(f"Using existing dataset: {dataset_file}")
    
    # Initialize DuckDB connector
    print("\nInitializing DuckDB connector...")
    db_connector = DuckDBConnector("process_data.duckdb")
    
    # Load the CSV into DuckDB
    print("Loading data into DuckDB...")
    db_connector.load_csv(
        filepath=dataset_file,
        table_name="process_events",
        case_id_column="case_id",
        activity_column="activity",
        timestamp_column="timestamp"
    )
    
    # Get table info
    print("\nDuckDB Table Information:")
    table_info = db_connector.get_table_info("process_events")
    print(f"  Records: {table_info['record_count']}")
    print(f"  Cases: {table_info['case_count']}")
    print(f"  Activities: {table_info['activity_count']}")
    print(f"  Time Range: {table_info['time_range']['min']} to {table_info['time_range']['max']}")
    print(f"  Duration: {table_info['time_range']['duration_days']} days")
    
    # Examples of queries
    print("\nExample 1: Getting a small sample as EventLog")
    sample_log = db_connector.query_to_eventlog(
        """
        SELECT * FROM process_events 
        WHERE case_id IN (
            SELECT DISTINCT case_id FROM process_events 
            LIMIT 50
        )
        ORDER BY case_id, timestamp
        """
    )
    
    print(f"  Sample log contains {sample_log.case_count()} cases and {len(sample_log)} events")
    
    # Create process map from sample
    print("  Creating process map from sample...")
    fig = create_interactive_process_map(
        sample_log,
        title="Process Map from DuckDB Sample",
        color_by="activity"
    )
    fig.write_html("duckdb_process_map.html", auto_open=True)
    print("  Saved process map to 'duckdb_process_map.html'")
    
    # Example 2: Filtering by time and category
    print("\nExample 2: Filtering by time and category")
    start_time = datetime(2023, 1, 15)
    end_time = datetime(2023, 1, 30)
    
    filtered_log = db_connector.query_to_eventlog(
        f"""
        SELECT * FROM process_events 
        WHERE timestamp >= TIMESTAMP '{start_time}'
          AND timestamp <= TIMESTAMP '{end_time}'
          AND category = 'Type A'
        ORDER BY case_id, timestamp
        """
    )
    
    print(f"  Filtered log contains {filtered_log.case_count()} cases and {len(filtered_log)} events")
    
    # Example 3: Using the get_eventlog method with filtering
    print("\nExample 3: Using get_eventlog with filtering")
    
    # Get high priority cases
    high_priority_log = db_connector.get_eventlog(
        table_name="process_events",
        case_ids=None,  # All cases
        activities=None,  # All activities
        start_time=None,  # All times
        end_time=None
    )
    
    # Execute a direct SQL query
    print("\nExample 4: Advanced SQL query")
    result = db_connector.execute("""
        WITH case_durations AS (
            SELECT 
                case_id,
                MIN(timestamp) AS start_time,
                MAX(timestamp) AS end_time,
                category,
                priority
            FROM process_events
            GROUP BY case_id, category, priority
        )
        
        SELECT 
            priority,
            category,
            COUNT(*) AS case_count,
            AVG(EXTRACT(EPOCH FROM (end_time - start_time)) / 3600) AS avg_duration_hours
        FROM case_durations
        GROUP BY priority, category
        ORDER BY priority, category
    """).fetchdf()
    
    print("\nCase Duration Analysis by Priority and Category:")
    print(result)
    
    # Close the connection
    db_connector.close()
    print("\nDuckDB connection closed")


if __name__ == "__main__":
    main()