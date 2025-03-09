"""
Example demonstrating interactive visualizations with Plotly.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.io as pio

from heraclitus.data import EventLog
from heraclitus.visualization import (
    create_interactive_process_map,
    create_cycle_time_distribution,
    create_activity_timeline,
    create_bottleneck_dashboard,
)


def generate_sample_data(num_cases=100):
    """Generate sample process data with multiple variants."""
    np.random.seed(42)  # For reproducibility
    
    # Define process variants
    variants = [
        {
            "name": "standard",
            "activities": ["Register", "Review", "Approve", "Process", "Notify", "Complete"],
            "probability": 0.6,
        },
        {
            "name": "expedited",
            "activities": ["Register", "Approve", "Process", "Complete"],
            "probability": 0.3,
        },
        {
            "name": "rejected",
            "activities": ["Register", "Review", "Reject", "Notify", "Complete"],
            "probability": 0.1,
        }
    ]
    
    # Define resource pools
    resources = {
        "Register": ["Alice", "Bob"],
        "Review": ["Charlie", "Diana"],
        "Approve": ["Eve", "Frank"],
        "Reject": ["Eve", "Frank"],
        "Process": ["Grace", "Heidi"],
        "Notify": ["Ivan", "Julia"],
        "Complete": ["Kate", "Luke"]
    }
    
    # Define departments
    departments = {
        "Register": "Intake",
        "Review": "Evaluation",
        "Approve": "Management",
        "Reject": "Management",
        "Process": "Operations",
        "Notify": "Communications",
        "Complete": "Intake"
    }
    
    # Generate data
    case_ids = []
    activities = []
    resources_list = []
    departments_list = []
    timestamps = []
    variants_list = []
    costs = []
    priorities = []
    
    base_time = datetime(2023, 1, 1, 8, 0)
    
    for i in range(1, num_cases + 1):
        case_id = f"case-{i:04d}"
        
        # Randomly select a variant
        rand = np.random.random()
        cum_prob = 0
        selected_variant = None
        
        for variant in variants:
            cum_prob += variant["probability"]
            if rand <= cum_prob:
                selected_variant = variant
                break
        
        # Set priority
        priority = np.random.choice(["low", "medium", "high"], p=[0.2, 0.6, 0.2])
        
        # Process the case
        current_time = base_time + timedelta(days=i//10, hours=i%10)
        
        for activity in selected_variant["activities"]:
            case_ids.append(case_id)
            activities.append(activity)
            
            # Select resource
            resource = np.random.choice(resources[activity])
            resources_list.append(resource)
            
            # Add department
            departments_list.append(departments[activity])
            
            # Add timestamp
            timestamps.append(current_time)
            
            # Add variant
            variants_list.append(selected_variant["name"])
            
            # Add priority
            priorities.append(priority)
            
            # Add cost (random based on activity)
            if activity == "Register":
                cost = np.random.randint(10, 20)
            elif activity in ["Review", "Notify"]:
                cost = np.random.randint(20, 40)
            elif activity in ["Approve", "Reject"]:
                cost = np.random.randint(50, 100)
            elif activity == "Process":
                cost = np.random.randint(100, 200)
            else:  # Complete
                cost = np.random.randint(10, 30)
            
            costs.append(cost)
            
            # Advance time
            wait_time = np.random.randint(30, 120)  # minutes
            current_time += timedelta(minutes=wait_time)
    
    # Create DataFrame
    df = pd.DataFrame({
        "case_id": case_ids,
        "activity": activities,
        "timestamp": timestamps,
        "resource": resources_list,
        "department": departments_list,
        "variant": variants_list,
        "cost": costs,
        "priority": priorities
    })
    
    return df


def main():
    """Run the interactive visualization examples."""
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data(num_cases=100)
    
    # Create EventLog
    event_log = EventLog(df)
    print(f"Created event log with {event_log.case_count()} cases and {len(event_log)} events")
    
    # Create interactive process map
    print("\nCreating interactive process map...")
    fig1 = create_interactive_process_map(
        event_log,
        frequency_threshold=0.01,
        title="Interactive Process Map",
        color_by="department"
    )
    
    # Save as HTML
    pio.write_html(fig1, "interactive_process_map.html", auto_open=True)
    print("Saved process map to 'interactive_process_map.html'")
    
    # Create cycle time distribution
    print("\nCreating cycle time distribution...")
    fig2 = create_cycle_time_distribution(
        event_log,
        group_by="variant",
        unit="minutes",
        title="Cycle Time by Process Variant"
    )
    
    # Save as HTML
    pio.write_html(fig2, "cycle_time_distribution.html", auto_open=True)
    print("Saved cycle time distribution to 'cycle_time_distribution.html'")
    
    # Create activity timeline
    print("\nCreating activity timeline...")
    # Get a sample of 10 cases for clarity
    sample_cases = event_log.get_unique_cases()[:10]
    
    fig3 = create_activity_timeline(
        event_log,
        cases=sample_cases,
        title="Activity Timeline for Sample Cases"
    )
    
    # Save as HTML
    pio.write_html(fig3, "activity_timeline.html", auto_open=True)
    print("Saved activity timeline to 'activity_timeline.html'")
    
    # Create bottleneck dashboard
    print("\nCreating bottleneck dashboard...")
    fig4 = create_bottleneck_dashboard(
        event_log,
        unit="minutes",
        title="Process Bottleneck Analysis"
    )
    
    # Save as HTML
    pio.write_html(fig4, "bottleneck_dashboard.html", auto_open=True)
    print("Saved bottleneck dashboard to 'bottleneck_dashboard.html'")
    
    print("\nAll visualizations have been created and saved as HTML files.")
    print("You can open these files in a web browser to interact with the visualizations.")


if __name__ == "__main__":
    main()