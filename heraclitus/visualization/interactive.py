"""
Interactive visualization module using Plotly.
"""
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from heraclitus.data import EventLog


def create_interactive_process_map(
    event_log: EventLog,
    frequency_threshold: float = 0.0,
    title: str = "Interactive Process Map",
    color_by: Optional[str] = None,
    node_color: str = "#6175c1",
    edge_color: str = "#a8b3d6",
) -> go.Figure:
    """
    Create an interactive process map visualization using Plotly.
    
    Args:
        event_log: The event log to visualize
        frequency_threshold: Minimum frequency for edges (as a proportion)
        title: Title for the visualization
        color_by: Optional attribute to color nodes by
        node_color: Default color for nodes
        edge_color: Default color for edges
    
    Returns:
        A Plotly Figure object containing the interactive process map
    """
    # Get the dataframe
    df = event_log.to_dataframe()
    
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
    
    # Calculate node frequencies
    node_freq = df[event_log.activity_column].value_counts().to_dict()
    
    # Create node lists
    nodes = list(node_freq.keys())
    node_sizes = [node_freq[node] for node in nodes]
    node_colors = [node_color] * len(nodes)
    
    # Create edge lists
    edges = [(source, target) for (source, target), count in transitions.items()
             if count / total_transitions >= frequency_threshold]
    edge_weights = [transitions[edge] for edge in edges]
    
    # Map node names to indices
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Create edge index lists
    edge_source_indices = [node_indices[source] for source, _ in edges]
    edge_target_indices = [node_indices[target] for _, target in edges]
    
    # Handle node coloring by attribute
    node_text = []
    
    if color_by and color_by in df.columns:
        # Create a mapping of activities to attribute values
        activity_attrs = {}
        for _, row in df.iterrows():
            activity = row[event_log.activity_column]
            attr_value = row[color_by]
            
            if activity not in activity_attrs:
                activity_attrs[activity] = []
            
            activity_attrs[activity].append(attr_value)
        
        # Determine most frequent value for each activity
        for i, node in enumerate(nodes):
            if node in activity_attrs:
                # Convert values to strings for consistent handling
                values = [str(val) for val in activity_attrs[node]]
                most_common = max(set(values), key=values.count)
                node_text.append(f"{node}<br>{color_by}: {most_common}")
            else:
                node_text.append(node)
        
        # Create colorscale
        unique_values = sorted(list(set(df[color_by].astype(str))))
        colorscale = px.colors.qualitative.Plotly[:len(unique_values)]
        
        # Map values to colors
        value_colors = {val: colorscale[i % len(colorscale)] 
                        for i, val in enumerate(unique_values)}
        
        # Assign colors to nodes
        for i, node in enumerate(nodes):
            if node in activity_attrs:
                values = [str(val) for val in activity_attrs[node]]
                most_common = max(set(values), key=values.count)
                node_colors[i] = value_colors[most_common]
    else:
        node_text = nodes
    
    # Create network graph using Plotly
    fig = go.Figure()
    
    # Add edges as scatter traces
    for i in range(len(edges)):
        source_idx = edge_source_indices[i]
        target_idx = edge_target_indices[i]
        weight = edge_weights[i]
        
        # Normalize edge weight for visual scaling
        norm_weight = 1 + (weight / max(edge_weights) * 5)
        
        # Add edge
        fig.add_trace(
            go.Scatter(
                x=[nodes[source_idx], nodes[target_idx]],
                y=[node_sizes[source_idx], node_sizes[target_idx]],
                mode='lines',
                line=dict(
                    width=norm_weight,
                    color=edge_color,
                    shape='spline'
                ),
                hoverinfo='text',
                text=f"{edges[i][0]} → {edges[i][1]}<br>Count: {weight}",
                opacity=0.7,
                showlegend=False
            )
        )
    
    # Add nodes as scatter trace
    fig.add_trace(
        go.Scatter(
            x=nodes,
            y=node_sizes,
            mode='markers+text',
            marker=dict(
                size=[10 + (size / max(node_sizes) * 40) for size in node_sizes],
                color=node_colors,
                line=dict(width=1, color='rgba(0,0,0,0.5)')
            ),
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            hovertext=[f"{node}<br>Frequency: {freq}" for node, freq in zip(nodes, node_sizes)],
            showlegend=False
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        width=900,
        annotations=[
            dict(
                text="Activity size represents frequency",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.01,
                align="left"
            )
        ]
    )
    
    return fig


def create_cycle_time_distribution(
    event_log: EventLog,
    group_by: Optional[str] = None,
    unit: str = "minutes",
    title: str = "Cycle Time Distribution",
) -> go.Figure:
    """
    Create an interactive cycle time distribution visualization using Plotly.
    
    Args:
        event_log: The event log to visualize
        group_by: Optional attribute to group cases by
        unit: Time unit for analysis
        title: Title for the visualization
    
    Returns:
        A Plotly Figure object containing the interactive visualization
    """
    from heraclitus.metrics import calculate_cycle_time
    
    # Get the dataframe
    df = event_log.to_dataframe()
    
    # Calculate cycle times
    if group_by and group_by in df.columns:
        # Group by the specified attribute
        groups = df[group_by].unique()
        
        # Create a figure with a histogram for each group
        fig = go.Figure()
        
        for group in groups:
            # Filter to this group
            group_df = df[df[group_by] == group]
            group_event_log = EventLog(
                group_df,
                case_id_column=event_log.case_id_column,
                activity_column=event_log.activity_column,
                timestamp_column=event_log.timestamp_column
            )
            
            # Calculate cycle times for each case in this group
            cycle_times = []
            for case_id in group_event_log.get_unique_cases():
                case_log = group_event_log.filter_cases([case_id])
                try:
                    time = calculate_cycle_time(case_log, unit=unit)
                    cycle_times.append(time)
                except ValueError:
                    continue
            
            # Add histogram for this group
            fig.add_trace(
                go.Histogram(
                    x=cycle_times,
                    name=str(group),
                    opacity=0.7,
                    nbinsx=20,
                    hovertemplate="Cycle Time: %{x:.1f} " + unit + "<br>Count: %{y}"
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=f"Cycle Time ({unit})",
            yaxis_title="Frequency",
            barmode='overlay',
            legend_title=group_by
        )
        
    else:
        # Calculate cycle times for all cases
        cycle_times = []
        for case_id in event_log.get_unique_cases():
            case_log = event_log.filter_cases([case_id])
            try:
                time = calculate_cycle_time(case_log, unit=unit)
                cycle_times.append(time)
            except ValueError:
                continue
        
        # Create histogram
        fig = go.Figure(
            go.Histogram(
                x=cycle_times,
                opacity=0.7,
                nbinsx=20,
                marker_color='#6175c1',
                hovertemplate="Cycle Time: %{x:.1f} " + unit + "<br>Count: %{y}"
            )
        )
        
        # Add median and mean lines
        if cycle_times:
            median = np.median(cycle_times)
            mean = np.mean(cycle_times)
            
            fig.add_vline(
                x=median,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Median: {median:.1f}",
                annotation_position="top right"
            )
            
            fig.add_vline(
                x=mean,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Mean: {mean:.1f}",
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=f"Cycle Time ({unit})",
            yaxis_title="Frequency"
        )
    
    # Common layout settings
    fig.update_layout(
        height=500,
        width=800,
        template="plotly_white"
    )
    
    return fig


def create_activity_timeline(
    event_log: EventLog,
    cases: Optional[List[str]] = None,
    title: str = "Activity Timeline",
) -> go.Figure:
    """
    Create an interactive timeline of activities using Plotly.
    
    Args:
        event_log: The event log to visualize
        cases: Optional list of case IDs to include (defaults to all)
        title: Title for the visualization
    
    Returns:
        A Plotly Figure object containing the timeline visualization
    """
    # Get the dataframe
    df = event_log.to_dataframe()
    
    # Filter to specified cases if provided
    if cases:
        df = df[df[event_log.case_id_column].isin(cases)]
    
    # Sort by timestamp
    df = df.sort_values(by=[event_log.case_id_column, event_log.timestamp_column])
    
    # Create a Gantt chart
    fig = go.Figure()
    
    # Get unique activities for color mapping
    activities = df[event_log.activity_column].unique()
    colors = px.colors.qualitative.Plotly[:len(activities)]
    activity_colors = {activity: colors[i % len(colors)] for i, activity in enumerate(activities)}
    
    # Process each case
    for case_id, case_df in df.groupby(event_log.case_id_column):
        for i in range(len(case_df) - 1):
            # Get current and next activity
            current_row = case_df.iloc[i]
            next_row = case_df.iloc[i+1]
            
            current_activity = current_row[event_log.activity_column]
            current_time = current_row[event_log.timestamp_column]
            next_time = next_row[event_log.timestamp_column]
            
            # Add bar for this activity
            fig.add_trace(
                go.Bar(
                    x=[next_time - current_time],
                    y=[case_id],
                    orientation='h',
                    base=current_time,
                    marker_color=activity_colors[current_activity],
                    name=current_activity,
                    text=current_activity,
                    hovertemplate=(
                        f"Case: {case_id}<br>" +
                        f"Activity: {current_activity}<br>" +
                        f"Start: %{{base|%Y-%m-%d %H:%M:%S}}<br>" +
                        f"End: %{{x|%Y-%m-%d %H:%M:%S}}<br>" +
                        f"Duration: %{{x}}"
                    ),
                    showlegend=True,
                    legendgroup=current_activity
                )
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Case ID",
        barmode='stack',
        height=500 + (15 * len(df[event_log.case_id_column].unique())),  # Dynamic height based on cases
        width=900,
        template="plotly_white",
        legend_title="Activity",
        # Deduplicate legend entries
        legend=dict(
            tracegroupgap=0,
            itemsizing='constant'
        )
    )
    
    # Update y-axis to show case IDs
    fig.update_yaxes(
        categoryorder='category ascending'
    )
    
    # Update x-axis to format dates
    fig.update_xaxes(
        type='date',
        tickformat='%Y-%m-%d %H:%M:%S'
    )
    
    return fig


def create_bottleneck_dashboard(
    event_log: EventLog,
    unit: str = "minutes",
    title: str = "Process Bottleneck Analysis",
) -> go.Figure:
    """
    Create an interactive dashboard for bottleneck analysis using Plotly.
    
    Args:
        event_log: The event log to analyze
        unit: Time unit for analysis
        title: Title for the visualization
    
    Returns:
        A Plotly Figure object containing the dashboard
    """
    from heraclitus.statistics import bottleneck_analysis
    
    # Get bottleneck analysis results
    wait_result = bottleneck_analysis(
        event_log,
        method="waiting_time",
        unit=unit
    )
    
    proc_result = bottleneck_analysis(
        event_log,
        method="processing_time",
        unit=unit
    )
    
    freq_result = bottleneck_analysis(
        event_log,
        method="frequency"
    )
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Waiting Time by Activity",
            "Processing Time by Activity",
            "Activity Frequency",
            "Waiting vs Processing Time"
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # Prepare data for bars
    wait_activities = list(wait_result['metrics'].keys())
    wait_times = [wait_result['metrics'][act]['mean'] for act in wait_activities]
    wait_errors = [wait_result['metrics'][act]['std'] for act in wait_activities]
    
    proc_activities = list(proc_result['metrics'].keys())
    proc_times = [proc_result['metrics'][act]['mean'] for act in proc_activities]
    proc_errors = [proc_result['metrics'][act]['std'] for act in proc_activities]
    
    freq_activities = list(freq_result['metrics'].keys())
    freq_counts = [freq_result['metrics'][act]['count'] for act in freq_activities]
    
    # Sort data by values
    wait_sorted = sorted(zip(wait_activities, wait_times, wait_errors),
                          key=lambda x: x[1], reverse=True)
    proc_sorted = sorted(zip(proc_activities, proc_times, proc_errors),
                           key=lambda x: x[1], reverse=True)
    freq_sorted = sorted(zip(freq_activities, freq_counts),
                          key=lambda x: x[1], reverse=True)
    
    # Top activities for each metric
    top_wait = [x[0] for x in wait_sorted[:10]]
    top_wait_times = [x[1] for x in wait_sorted[:10]]
    top_wait_errors = [x[2] for x in wait_sorted[:10]]
    
    top_proc = [x[0] for x in proc_sorted[:10]]
    top_proc_times = [x[1] for x in proc_sorted[:10]]
    top_proc_errors = [x[2] for x in proc_sorted[:10]]
    
    top_freq = [x[0] for x in freq_sorted[:10]]
    top_freq_counts = [x[1] for x in freq_sorted[:10]]
    
    # Add waiting time bar chart
    fig.add_trace(
        go.Bar(
            x=top_wait,
            y=top_wait_times,
            error_y=dict(
                type='data',
                array=top_wait_errors,
                visible=True
            ),
            marker_color='#6175c1',
            name="Waiting Time",
            hovertemplate="Activity: %{x}<br>Waiting Time: %{y:.1f} " + unit
        ),
        row=1, col=1
    )
    
    # Add processing time bar chart
    fig.add_trace(
        go.Bar(
            x=top_proc,
            y=top_proc_times,
            error_y=dict(
                type='data',
                array=top_proc_errors,
                visible=True
            ),
            marker_color='#c16161',
            name="Processing Time",
            hovertemplate="Activity: %{x}<br>Processing Time: %{y:.1f} " + unit
        ),
        row=1, col=2
    )
    
    # Add frequency bar chart
    fig.add_trace(
        go.Bar(
            x=top_freq,
            y=top_freq_counts,
            marker_color='#61c17e',
            name="Frequency",
            hovertemplate="Activity: %{x}<br>Count: %{y}"
        ),
        row=2, col=1
    )
    
    # Add scatter plot of waiting vs processing time
    scatter_data = []
    
    # Find activities present in both waiting and processing results
    common_activities = set(wait_result['metrics'].keys()) & set(proc_result['metrics'].keys())
    
    for activity in common_activities:
        scatter_data.append({
            'activity': activity,
            'waiting_time': wait_result['metrics'][activity]['mean'],
            'processing_time': proc_result['metrics'][activity]['mean'],
            'frequency': freq_result['metrics'][activity]['count'] if activity in freq_result['metrics'] else 0
        })
    
    # Create scatter plot
    if scatter_data:
        activities = [item['activity'] for item in scatter_data]
        waiting_times = [item['waiting_time'] for item in scatter_data]
        processing_times = [item['processing_time'] for item in scatter_data]
        frequencies = [item['frequency'] for item in scatter_data]
        
        # Scale marker sizes based on frequency
        max_freq = max(frequencies) if frequencies else 1
        marker_sizes = [10 + (freq / max_freq * 50) for freq in frequencies]
        
        fig.add_trace(
            go.Scatter(
                x=waiting_times,
                y=processing_times,
                mode='markers+text',
                marker=dict(
                    size=marker_sizes,
                    color=frequencies,
                    colorscale='Viridis',
                    colorbar=dict(
                        title="Frequency",
                        x=1.05,
                        y=0.25,
                        len=0.5
                    ),
                    showscale=True
                ),
                text=activities,
                textposition='top center',
                hovertemplate=(
                    "Activity: %{text}<br>" +
                    f"Waiting Time: %{{x:.1f}} {unit}<br>" +
                    f"Processing Time: %{{y:.1f}} {unit}<br>" +
                    "Frequency: %{marker.size}"
                ),
                name="Waiting vs Processing"
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        width=1000,
        template="plotly_white",
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text=f"Waiting Time ({unit})", row=2, col=2)
    fig.update_yaxes(title_text=f"Processing Time ({unit})", row=2, col=2)
    
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    
    fig.update_yaxes(title_text=f"Time ({unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Time ({unit})", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    return fig