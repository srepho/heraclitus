"""
Basic usage example for the heraclitus package.
"""
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Import the heraclitus package
from heraclitus.data import EventLog
from heraclitus.visualization import visualize_process_map, plot_activity_frequency
from heraclitus.metrics import calculate_cycle_time, calculate_waiting_time

# Create a sample event log
data = {
    "case_id": [
        "case1", "case1", "case1", "case1",
        "case2", "case2", "case2",
        "case3", "case3", "case3", "case3", "case3"
    ],
    "activity": [
        "A_Register", "B_Verify", "C_Process", "D_Complete",
        "A_Register", "B_Verify", "D_Complete",
        "A_Register", "B_Verify", "E_Review", "C_Process", "D_Complete"
    ],
    "timestamp": [
        datetime(2023, 1, 1, 9, 0), datetime(2023, 1, 1, 9, 30),
        datetime(2023, 1, 1, 10, 15), datetime(2023, 1, 1, 11, 0),
        
        datetime(2023, 1, 2, 10, 0), datetime(2023, 1, 2, 10, 45),
        datetime(2023, 1, 2, 11, 30),
        
        datetime(2023, 1, 3, 9, 0), datetime(2023, 1, 3, 9, 45),
        datetime(2023, 1, 3, 10, 30), datetime(2023, 1, 3, 11, 15),
        datetime(2023, 1, 3, 12, 0)
    ],
    "resource": [
        "John", "Mary", "John", "Mary",
        "Mary", "John", "Mary",
        "John", "Mary", "Supervisor", "John", "Mary"
    ]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Create an EventLog from the DataFrame
event_log = EventLog(df)

print(f"Event log contains {len(event_log)} events from {event_log.case_count()} cases")
print(f"Activities in log: {', '.join(event_log.get_unique_activities())}")

# Calculate metrics
avg_cycle_time = calculate_cycle_time(event_log, unit="minutes")
print(f"Average cycle time: {avg_cycle_time:.1f} minutes")

# Calculate waiting time for process activity
process_wait_time = calculate_waiting_time(
    event_log, "C_Process", unit="minutes", include_stats=True
)
print(f"Waiting time for C_Process: {process_wait_time['mean']:.1f} minutes")
print(f"Min: {process_wait_time['min']:.1f}, Max: {process_wait_time['max']:.1f}")

# Create visualizations
# Process map
process_fig = visualize_process_map(event_log, title="Sample Process Map")

# Activity frequency
freq_fig = plot_activity_frequency(event_log, title="Activity Frequency")

plt.tight_layout()
plt.show()

# Filter the event log to analyze a specific case
case3_log = event_log.filter_cases(["case3"])
print(f"\nAnalyzing case3 ({len(case3_log)} events):")

case3_time = calculate_cycle_time(case3_log, unit="minutes")
print(f"Cycle time for case3: {case3_time:.1f} minutes")

# Filter by time range
start_time = pd.Timestamp("2023-01-02 00:00:00")
end_time = pd.Timestamp("2023-01-03 00:00:00")
filtered_log = event_log.filter_time_range(start_time, end_time)
print(f"\nEvents between {start_time.date()} and {end_time.date()}: {len(filtered_log)}")

# Add an attribute
durations = [(df.iloc[i+1]['timestamp'] - df.iloc[i]['timestamp']).total_seconds()/60 
            if i+1 < len(df) and df.iloc[i+1]['case_id'] == df.iloc[i]['case_id'] 
            else 0 for i in range(len(df))]
event_log.add_attribute("duration_min", durations)
print("\nAdded duration attribute")

# Check the attributes
print(f"Available attributes: {', '.join(event_log.get_attributes())}")