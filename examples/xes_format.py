"""
XES Format Example

This example demonstrates how to use Heraclitus to work with XES format files,
which is the IEEE 1849-2016 standard for process mining event logs.
"""
import os
import pandas as pd
import datetime as dt
import tempfile

from heraclitus.data import EventLog
from heraclitus.data.xes_handler import import_xes, export_xes
from heraclitus.discovery.process_discovery import discover_directly_follows_graph


def create_sample_xes_file():
    """
    Create a sample XES file for demonstration purposes.
    
    Returns:
        Path to the created XES file
    """
    # Create a temporary file for the XES data
    fd, temp_path = tempfile.mkstemp(suffix=".xes")
    os.close(fd)
    
    # Create sample event data
    df = pd.DataFrame([
        # Case 1: Happy path
        {"case_id": "case_1", "activity": "Register", "timestamp": dt.datetime(2023, 1, 1, 10, 0), "resource": "Alice", "cost": 10},
        {"case_id": "case_1", "activity": "Review", "timestamp": dt.datetime(2023, 1, 1, 11, 0), "resource": "Bob", "cost": 20},
        {"case_id": "case_1", "activity": "Approve", "timestamp": dt.datetime(2023, 1, 1, 12, 0), "resource": "Charlie", "cost": 15},
        {"case_id": "case_1", "activity": "Notify", "timestamp": dt.datetime(2023, 1, 1, 13, 0), "resource": "Alice", "cost": 5},
        
        # Case 2: Rejection path
        {"case_id": "case_2", "activity": "Register", "timestamp": dt.datetime(2023, 1, 2, 9, 0), "resource": "Alice", "cost": 10},
        {"case_id": "case_2", "activity": "Review", "timestamp": dt.datetime(2023, 1, 2, 10, 0), "resource": "Bob", "cost": 20},
        {"case_id": "case_2", "activity": "Reject", "timestamp": dt.datetime(2023, 1, 2, 11, 0), "resource": "Charlie", "cost": 15},
        {"case_id": "case_2", "activity": "Notify", "timestamp": dt.datetime(2023, 1, 2, 12, 0), "resource": "Alice", "cost": 5},
        
        # Case 3: Alternative path with verification
        {"case_id": "case_3", "activity": "Register", "timestamp": dt.datetime(2023, 1, 3, 14, 0), "resource": "Bob", "cost": 10},
        {"case_id": "case_3", "activity": "Verify", "timestamp": dt.datetime(2023, 1, 3, 15, 0), "resource": "Charlie", "cost": 25},
        {"case_id": "case_3", "activity": "Review", "timestamp": dt.datetime(2023, 1, 3, 16, 0), "resource": "Alice", "cost": 20},
        {"case_id": "case_3", "activity": "Approve", "timestamp": dt.datetime(2023, 1, 3, 17, 0), "resource": "Bob", "cost": 15},
        {"case_id": "case_3", "activity": "Notify", "timestamp": dt.datetime(2023, 1, 3, 18, 0), "resource": "Charlie", "cost": 5}
    ])
    
    # Create an EventLog and export to XES
    event_log = EventLog(
        df,
        case_id_column="case_id",
        activity_column="activity",
        timestamp_column="timestamp"
    )
    
    export_xes(event_log, temp_path)
    print(f"Sample XES file created at: {temp_path}")
    
    return temp_path


def example_xes_import_export():
    """Example of importing and exporting XES files."""
    print("\n=== XES Import/Export Example ===")
    
    # Create a sample XES file
    xes_file_path = create_sample_xes_file()
    
    # Import the XES file
    print("\nImporting XES file...")
    event_log = import_xes(xes_file_path)
    
    # Display EventLog information
    print(f"EventLog created with {len(event_log.to_dataframe())} events "
          f"across {len(event_log.get_case_ids())} cases")
    
    # Show unique activities
    activities = event_log.get_activities()
    print(f"\nProcess contains {len(activities)} unique activities:")
    for activity in activities:
        print(f"- {activity}")
    
    # Export to a new XES file
    output_path = os.path.join(os.path.dirname(xes_file_path), "exported.xes")
    print(f"\nExporting EventLog to XES file: {output_path}")
    event_log.to_xes(output_path)
    
    # Re-import the exported file to verify
    print("\nVerifying export by re-importing the XES file...")
    reimported_log = EventLog.from_xes(output_path)
    
    original_df = event_log.to_dataframe()
    reimported_df = reimported_log.to_dataframe()
    
    print(f"Original event count: {len(original_df)}")
    print(f"Reimported event count: {len(reimported_df)}")
    print(f"Data integrity preserved: {len(original_df) == len(reimported_df)}")
    
    # Clean up temporary files
    try:
        os.remove(xes_file_path)
        os.remove(output_path)
        print("\nTemporary files cleaned up.")
    except Exception as e:
        print(f"\nError cleaning up temporary files: {str(e)}")
    
    return event_log


def example_xes_process_discovery(event_log):
    """Example of process discovery using an XES imported EventLog."""
    print("\n=== Process Discovery with XES Data ===")
    
    # Discover a Directly-Follows Graph
    dfg = discover_directly_follows_graph(event_log)
    
    # Print DFG information
    print("\nDirectly-Follows Graph:")
    print(f"- Nodes (activities): {len(dfg.get_nodes())}")
    print(f"- Edges (transitions): {len(dfg.get_edges())}")
    
    print("\nStarting activities:")
    for activity, count in dfg.get_starting_activities().items():
        print(f"- {activity}: {count} cases")
    
    print("\nEnding activities:")
    for activity, count in dfg.get_ending_activities().items():
        print(f"- {activity}: {count} cases")
    
    print("\nAll transitions (activity -> activity):")
    for (source, target), frequency in sorted(dfg.get_edges().items(), key=lambda x: x[1], reverse=True):
        print(f"- {source} -> {target}: {frequency} occurrences")


if __name__ == "__main__":
    print("Heraclitus XES Format Example")
    print("=============================")
    
    # Run examples
    event_log = example_xes_import_export()
    example_xes_process_discovery(event_log)
    
    print("\nExample completed!")