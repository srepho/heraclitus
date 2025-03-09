"""
Process Discovery Example

This example demonstrates the process discovery and conformance checking capabilities
of Heraclitus, both with and without PM4PY integration.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os

# Import Heraclitus
from heraclitus.data import EventLog
from heraclitus.discovery.process_discovery import (
    discover_directly_follows_graph,
    discover_process_model,
    conformance_checking
)
from heraclitus.visualization.pm4py_viz import visualize_process_model, convert_to_bpmn


# Check if PM4PY is available
try:
    import pm4py
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    warnings.warn(
        "PM4PY is not installed. Some examples will be skipped. "
        "Install with: pip install heraclitus[pm4py]"
    )


def create_sample_event_log() -> EventLog:
    """Create a sample event log for process discovery."""
    # Create sample data
    cases = []
    
    # Process model: Register -> Review -> (Approve or Reject) -> Notify
    # With some variations and deviations
    
    for i in range(1, 101):
        case_id = f"case_{i}"
        
        # Standard path (70%)
        if i <= 70:
            start_time = datetime(2023, 1, 1) + timedelta(days=i, hours=np.random.randint(0, 5))
            
            # Register
            events = [
                {
                    "case_id": case_id,
                    "activity": "Register",
                    "timestamp": start_time,
                    "resource": f"employee_{np.random.randint(1, 5)}"
                }
            ]
            
            # Review
            events.append({
                "case_id": case_id,
                "activity": "Review",
                "timestamp": start_time + timedelta(hours=np.random.randint(1, 4)),
                "resource": f"employee_{np.random.randint(1, 5)}"
            })
            
            # Approve or Reject (80% approve, 20% reject)
            if i <= 56:  # 80% of 70 cases
                events.append({
                    "case_id": case_id,
                    "activity": "Approve",
                    "timestamp": events[-1]["timestamp"] + timedelta(hours=np.random.randint(1, 3)),
                    "resource": f"employee_{np.random.randint(1, 5)}"
                })
            else:
                events.append({
                    "case_id": case_id,
                    "activity": "Reject",
                    "timestamp": events[-1]["timestamp"] + timedelta(hours=np.random.randint(1, 3)),
                    "resource": f"employee_{np.random.randint(1, 5)}"
                })
            
            # Notify
            events.append({
                "case_id": case_id,
                "activity": "Notify",
                "timestamp": events[-1]["timestamp"] + timedelta(hours=np.random.randint(1, 2)),
                "resource": f"employee_{np.random.randint(1, 5)}"
            })
        
        # Variation 1: Register -> Verify -> Review -> (Approve or Reject) -> Notify (20%)
        elif i <= 90:
            start_time = datetime(2023, 1, 1) + timedelta(days=i, hours=np.random.randint(0, 5))
            
            # Register
            events = [
                {
                    "case_id": case_id,
                    "activity": "Register",
                    "timestamp": start_time,
                    "resource": f"employee_{np.random.randint(1, 5)}"
                }
            ]
            
            # Verify (additional step)
            events.append({
                "case_id": case_id,
                "activity": "Verify",
                "timestamp": start_time + timedelta(hours=np.random.randint(1, 3)),
                "resource": f"employee_{np.random.randint(1, 5)}"
            })
            
            # Review
            events.append({
                "case_id": case_id,
                "activity": "Review",
                "timestamp": events[-1]["timestamp"] + timedelta(hours=np.random.randint(1, 4)),
                "resource": f"employee_{np.random.randint(1, 5)}"
            })
            
            # Approve or Reject (50% each)
            if i <= 80:  # 50% of 20 cases
                events.append({
                    "case_id": case_id,
                    "activity": "Approve",
                    "timestamp": events[-1]["timestamp"] + timedelta(hours=np.random.randint(1, 3)),
                    "resource": f"employee_{np.random.randint(1, 5)}"
                })
            else:
                events.append({
                    "case_id": case_id,
                    "activity": "Reject",
                    "timestamp": events[-1]["timestamp"] + timedelta(hours=np.random.randint(1, 3)),
                    "resource": f"employee_{np.random.randint(1, 5)}"
                })
            
            # Notify
            events.append({
                "case_id": case_id,
                "activity": "Notify",
                "timestamp": events[-1]["timestamp"] + timedelta(hours=np.random.randint(1, 2)),
                "resource": f"employee_{np.random.randint(1, 5)}"
            })
        
        # Variation 2 (Deviation): Register -> Review -> Approve -> Modify -> Notify (10%)
        else:
            start_time = datetime(2023, 1, 1) + timedelta(days=i, hours=np.random.randint(0, 5))
            
            # Register
            events = [
                {
                    "case_id": case_id,
                    "activity": "Register",
                    "timestamp": start_time,
                    "resource": f"employee_{np.random.randint(1, 5)}"
                }
            ]
            
            # Review
            events.append({
                "case_id": case_id,
                "activity": "Review",
                "timestamp": start_time + timedelta(hours=np.random.randint(1, 4)),
                "resource": f"employee_{np.random.randint(1, 5)}"
            })
            
            # Approve
            events.append({
                "case_id": case_id,
                "activity": "Approve",
                "timestamp": events[-1]["timestamp"] + timedelta(hours=np.random.randint(1, 3)),
                "resource": f"employee_{np.random.randint(1, 5)}"
            })
            
            # Modify (deviation from standard process)
            events.append({
                "case_id": case_id,
                "activity": "Modify",
                "timestamp": events[-1]["timestamp"] + timedelta(hours=np.random.randint(1, 3)),
                "resource": f"employee_{np.random.randint(1, 5)}"
            })
            
            # Notify
            events.append({
                "case_id": case_id,
                "activity": "Notify",
                "timestamp": events[-1]["timestamp"] + timedelta(hours=np.random.randint(1, 2)),
                "resource": f"employee_{np.random.randint(1, 5)}"
            })
        
        cases.extend(events)
    
    # Convert to DataFrame and create EventLog
    df = pd.DataFrame(cases)
    event_log = EventLog(df, case_id_column="case_id", activity_column="activity", timestamp_column="timestamp")
    
    return event_log


def example_directly_follows_graph():
    """Example of creating a directly-follows graph."""
    print("\n=== Directly-Follows Graph Example ===")
    
    # Create sample event log
    event_log = create_sample_event_log()
    print(f"Event log created with {len(event_log.to_dataframe())} events "
          f"across {len(event_log.get_case_ids())} cases")
    
    # Discover DFG using native implementation
    dfg = discover_directly_follows_graph(event_log, threshold=0.05)
    
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
    
    print("\nTop transitions (activity -> activity):")
    sorted_edges = sorted(dfg.get_edges().items(), key=lambda x: x[1], reverse=True)
    for (source, target), frequency in sorted_edges[:5]:
        print(f"- {source} -> {target}: {frequency} occurrences")


def example_pm4py_integration():
    """Example of PM4PY integration for process discovery."""
    if not PM4PY_AVAILABLE:
        print("\n=== PM4PY Integration Example (SKIPPED) ===")
        print("PM4PY is not installed. Install with: pip install heraclitus[pm4py]")
        return
    
    print("\n=== PM4PY Integration Example ===")
    
    # Create sample event log
    event_log = create_sample_event_log()
    
    # Discover process models using different algorithms
    print("\nDiscovering process models using PM4PY algorithms...")
    
    # Alpha Miner
    alpha_model = discover_process_model(event_log, algorithm="alpha")
    print("- Alpha Miner: Process model discovered")
    
    # Inductive Miner
    inductive_model = discover_process_model(event_log, algorithm="inductive")
    print("- Inductive Miner: Process model discovered")
    
    # Heuristics Miner
    heuristics_model = discover_process_model(event_log, algorithm="heuristics")
    print("- Heuristics Miner: Process model discovered")
    
    # Visualize the models
    print("\nVisualizing process models...")
    
    # Save in the current directory
    output_dir = os.getcwd()
    
    # Visualize Inductive Miner result
    inductive_path = visualize_process_model(
        event_log, 
        algorithm="inductive",
        output_path=os.path.join(output_dir, "inductive_model.png")
    )
    print(f"- Inductive Miner visualization saved to: {inductive_path}")
    
    # Visualize Directly-Follows Graph
    dfg_path = visualize_process_model(
        event_log, 
        algorithm="dfg",
        show_performance=True,
        output_path=os.path.join(output_dir, "dfg_model.png")
    )
    print(f"- DFG visualization saved to: {dfg_path}")
    
    # Create BPMN diagram
    bpmn_path = convert_to_bpmn(
        event_log,
        output_path=os.path.join(output_dir, "bpmn_model.png")
    )
    print(f"- BPMN diagram saved to: {bpmn_path}")


def example_conformance_checking():
    """Example of conformance checking."""
    if not PM4PY_AVAILABLE:
        print("\n=== Conformance Checking Example (SKIPPED) ===")
        print("PM4PY is not installed. Install with: pip install heraclitus[pm4py]")
        return
    
    print("\n=== Conformance Checking Example ===")
    
    # Create sample event log
    event_log = create_sample_event_log()
    
    # Discover a process model
    print("\nDiscovering process model with Inductive Miner...")
    model = discover_process_model(event_log, algorithm="inductive")
    
    # Convert to Petri net for conformance checking
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(model)
    
    # Perform conformance checking with token replay
    print("\nPerforming conformance checking with token replay...")
    token_results = conformance_checking(
        event_log,
        (net, initial_marking, final_marking),
        algorithm="token_replay"
    )
    
    # Print results
    print(f"- Overall fitness: {token_results['aggregate']['fitness']:.4f}")
    print(f"- Produced tokens: {token_results['aggregate']['produced_tokens']}")
    print(f"- Consumed tokens: {token_results['aggregate']['consumed_tokens']}")
    print(f"- Missing tokens: {token_results['aggregate']['missing_tokens']}")
    print(f"- Remaining tokens: {token_results['aggregate']['remaining_tokens']}")
    
    # Alignments (can be computationally expensive)
    print("\nPerforming conformance checking with alignments (may take some time)...")
    try:
        alignment_results = conformance_checking(
            event_log,
            (net, initial_marking, final_marking),
            algorithm="alignments"
        )
        
        # Print results
        print(f"- Overall fitness: {alignment_results['aggregate']['fitness']:.4f}")
        print(f"- Min fitness: {alignment_results['aggregate']['min_fitness']:.4f}")
        print(f"- Max fitness: {alignment_results['aggregate']['max_fitness']:.4f}")
    except Exception as e:
        print(f"Error during alignment calculation: {str(e)}")
        print("Alignments can be computationally expensive and may timeout.")


if __name__ == "__main__":
    print("Heraclitus Process Discovery Example")
    print("====================================")
    
    # Run examples
    example_directly_follows_graph()
    example_pm4py_integration()
    example_conformance_checking()
    
    print("\nExample completed!")