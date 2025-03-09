"""
Process map visualization functions.
"""
from typing import Optional, Dict, Any, Union, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from matplotlib.figure import Figure

from heraclitus.data import EventLog


def visualize_process_map(
    event_log: EventLog,
    frequency_threshold: float = 0.0,
    performance_metric: Optional[str] = None,
    custom_node_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
    custom_edge_attributes: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    title: str = "Process Map",
    figsize: Tuple[int, int] = (10, 8),
) -> Figure:
    """
    Create a process map visualization from an event log.
    
    Args:
        event_log: The event log to visualize
        frequency_threshold: Minimum frequency for edges (as a proportion)
        performance_metric: Optional performance metric to display on edges
        custom_node_attributes: Custom attributes for nodes
        custom_edge_attributes: Custom attributes for edges
        title: Title for the visualization
        figsize: Figure size (width, height) in inches
    
    Returns:
        A matplotlib Figure object containing the process map
    """
    # Get the dataframe
    df = event_log.to_dataframe()
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Sort by case_id and timestamp
    df = df.sort_values(by=[event_log.case_id_column, event_log.timestamp_column])
    
    # Count transitions between activities
    transitions = {}
    total_transitions = 0
    
    # Group by case_id
    for case_id, case_df in df.groupby(event_log.case_id_column):
        # Get the sequence of activities
        activities = case_df[event_log.activity_column].tolist()
        
        # Count transitions
        for i in range(len(activities) - 1):
            source = activities[i]
            target = activities[i + 1]
            transition = (source, target)
            
            if transition in transitions:
                transitions[transition] += 1
            else:
                transitions[transition] = 1
            
            total_transitions += 1
    
    # Add nodes for all activities
    unique_activities = df[event_log.activity_column].unique()
    for activity in unique_activities:
        G.add_node(activity)
        
        # Count occurrences of each activity for node size
        activity_count = df[df[event_log.activity_column] == activity].shape[0]
        G.nodes[activity]["weight"] = activity_count
    
    # Add edges based on frequency threshold
    for (source, target), count in transitions.items():
        # Calculate frequency as a proportion
        frequency = count / total_transitions
        
        if frequency >= frequency_threshold:
            G.add_edge(source, target, weight=count, frequency=frequency)
    
    # Apply custom node attributes if provided
    if custom_node_attributes:
        for node, attrs in custom_node_attributes.items():
            if node in G.nodes:
                for attr, value in attrs.items():
                    G.nodes[node][attr] = value
    
    # Apply custom edge attributes if provided
    if custom_edge_attributes:
        for edge, attrs in custom_edge_attributes.items():
            if edge in G.edges:
                for attr, value in attrs.items():
                    G.edges[edge][attr] = value
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate node sizes based on activity frequency
    node_sizes = [G.nodes[node]["weight"] * 100 for node in G.nodes]
    
    # Calculate edge widths based on transition frequency
    edge_widths = [G.edges[edge]["weight"] / 10 for edge in G.edges]
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color="skyblue",
        alpha=0.8,
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        width=edge_widths,
        edge_color="gray",
        alpha=0.6,
        arrowsize=15,
        arrowstyle="->",
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10,
        font_family="sans-serif",
        ax=ax
    )
    
    # Add edge labels if performance metric is provided
    if performance_metric:
        # This is a placeholder - actual implementation would use metrics module
        edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels,
            font_size=8,
            ax=ax
        )
    
    # Set title and remove axis
    ax.set_title(title)
    ax.axis("off")
    
    return fig


def plot_activity_frequency(
    event_log: EventLog,
    title: str = "Activity Frequency",
    figsize: Tuple[int, int] = (10, 6),
    sort_by: str = "frequency",
) -> Figure:
    """
    Create a bar chart of activity frequencies.
    
    Args:
        event_log: The event log to visualize
        title: Title for the visualization
        figsize: Figure size (width, height) in inches
        sort_by: How to sort activities ('frequency' or 'alphabetical')
    
    Returns:
        A matplotlib Figure object containing the bar chart
    """
    # Get the dataframe
    df = event_log.to_dataframe()
    
    # Count activity frequencies
    activity_counts = df[event_log.activity_column].value_counts()
    
    # Sort based on user preference
    if sort_by == "alphabetical":
        activity_counts = activity_counts.sort_index()
    # Default is already sorted by frequency
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    activity_counts.plot(kind="bar", ax=ax)
    
    # Set labels and title
    ax.set_xlabel("Activity")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig