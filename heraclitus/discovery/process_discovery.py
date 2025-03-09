"""
Process discovery algorithms and utilities.

This module provides functions to discover process models from event logs,
including directly implemented algorithms and wrappers around PM4PY algorithms.
"""
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
import pandas as pd
import numpy as np
import warnings

from heraclitus.data import EventLog
from heraclitus.utils.pm4py_converter import check_pm4py_available, apply_pm4py_algorithm, to_pm4py

try:
    import pm4py
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False


class DirectlyFollowsGraph:
    """
    A simple implementation of a Directly-Follows Graph (DFG) for process discovery.
    
    This class provides a lightweight alternative to PM4PY-based DFGs and is always
    available even when PM4PY is not installed.
    """
    
    def __init__(self, event_log: EventLog, threshold: float = 0.0):
        """
        Initialize a DFG from an event log.
        
        Args:
            event_log: The event log to analyze
            threshold: Minimum frequency threshold (0.0-1.0) to include an edge
        """
        self.event_log = event_log
        self.threshold = threshold
        self.nodes = set()  # Activities
        self.edges = {}     # (source, target) -> frequency
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the directly-follows graph from the event log."""
        df = self.event_log.to_dataframe()
        case_id_col = self.event_log.case_id_column
        act_col = self.event_log.activity_column
        ts_col = self.event_log.timestamp_column
        
        # Sort by case ID and timestamp
        df = df.sort_values([case_id_col, ts_col])
        
        # For each case, find directly-follows relationships
        for case_id, case_df in df.groupby(case_id_col):
            activities = case_df[act_col].tolist()
            
            # Record all activities
            self.nodes.update(activities)
            
            # Record all directly-follows relationships
            for i in range(len(activities) - 1):
                source = activities[i]
                target = activities[i + 1]
                edge = (source, target)
                
                if edge in self.edges:
                    self.edges[edge] += 1
                else:
                    self.edges[edge] = 1
        
        # Apply threshold filtering
        if self.threshold > 0:
            max_freq = max(self.edges.values()) if self.edges else 0
            self.edges = {edge: freq for edge, freq in self.edges.items() 
                         if freq / max_freq >= self.threshold}
    
    def get_edges(self) -> Dict[Tuple[str, str], int]:
        """
        Get all edges in the DFG.
        
        Returns:
            Dictionary mapping (source, target) tuples to frequencies
        """
        return self.edges
    
    def get_nodes(self) -> set:
        """
        Get all nodes in the DFG.
        
        Returns:
            Set of activity names
        """
        return self.nodes
    
    def get_starting_activities(self) -> Dict[str, int]:
        """
        Get activities that start a case.
        
        Returns:
            Dictionary mapping activity names to frequencies
        """
        df = self.event_log.to_dataframe()
        case_id_col = self.event_log.case_id_column
        act_col = self.event_log.activity_column
        ts_col = self.event_log.timestamp_column
        
        # Group by case ID and get the first activity
        first_activities = df.sort_values([case_id_col, ts_col]) \
            .groupby(case_id_col).first()[act_col]
        
        # Count occurrences of each starting activity
        return dict(first_activities.value_counts())
    
    def get_ending_activities(self) -> Dict[str, int]:
        """
        Get activities that end a case.
        
        Returns:
            Dictionary mapping activity names to frequencies
        """
        df = self.event_log.to_dataframe()
        case_id_col = self.event_log.case_id_column
        act_col = self.event_log.activity_column
        ts_col = self.event_log.timestamp_column
        
        # Group by case ID and get the last activity
        last_activities = df.sort_values([case_id_col, ts_col]) \
            .groupby(case_id_col).last()[act_col]
        
        # Count occurrences of each ending activity
        return dict(last_activities.value_counts())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DFG to a dictionary format.
        
        Returns:
            Dictionary representation of the DFG with nodes, edges,
            starting activities, and ending activities
        """
        return {
            'nodes': list(self.nodes),
            'edges': self.edges,
            'start_activities': self.get_starting_activities(),
            'end_activities': self.get_ending_activities()
        }


def discover_directly_follows_graph(
    event_log: EventLog,
    threshold: float = 0.0,
    use_pm4py: bool = False
) -> Union[DirectlyFollowsGraph, Any]:
    """
    Discover a directly-follows graph from an event log.
    
    Args:
        event_log: The event log to analyze
        threshold: Minimum frequency threshold (0.0-1.0) to include an edge
        use_pm4py: Whether to use PM4PY for DFG discovery (if available)
    
    Returns:
        A DirectlyFollowsGraph object if use_pm4py=False,
        or a PM4PY DFG object if use_pm4py=True
    """
    if use_pm4py and PM4PY_AVAILABLE:
        # Use PM4PY to create the DFG
        pm4py_log = to_pm4py(event_log)
        
        # Apply threshold
        parameters = {}
        if threshold > 0:
            parameters["threshold"] = threshold
            
        return dfg_discovery.apply(pm4py_log, parameters=parameters)
    else:
        # Use native implementation
        return DirectlyFollowsGraph(event_log, threshold)


def discover_process_model(
    event_log: EventLog,
    algorithm: str = "dfg",
    parameters: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Discover a process model from an event log.
    
    Args:
        event_log: The event log to analyze
        algorithm: The algorithm to use for process discovery
            Options: 
                - "dfg": Directly-Follows Graph (native or PM4PY)
                - "alpha": Alpha Miner (requires PM4PY)
                - "inductive": Inductive Miner (requires PM4PY)
                - "heuristics": Heuristics Miner (requires PM4PY)
        parameters: Parameters for the discovery algorithm
    
    Returns:
        A process model object appropriate for the chosen algorithm
    
    Raises:
        ImportError: If PM4PY is required but not installed
        ValueError: If an unknown algorithm is specified
    """
    parameters = parameters or {}
    
    if algorithm == "dfg":
        use_pm4py = parameters.pop("use_pm4py", False)
        threshold = parameters.pop("threshold", 0.0)
        return discover_directly_follows_graph(event_log, threshold, use_pm4py)
    
    elif algorithm == "alpha":
        check_pm4py_available()
        return apply_pm4py_algorithm(event_log, alpha_miner.apply, **parameters)
    
    elif algorithm == "inductive":
        check_pm4py_available()
        return apply_pm4py_algorithm(event_log, inductive_miner.apply_tree, **parameters)
    
    elif algorithm == "heuristics":
        check_pm4py_available()
        return apply_pm4py_algorithm(event_log, heuristics_miner.apply_heu, **parameters)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                         "Available options: dfg, alpha, inductive, heuristics")


def conformance_checking(
    event_log: EventLog,
    process_model: Any,
    algorithm: str = "token_replay",
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform conformance checking between an event log and a process model.
    
    Args:
        event_log: The event log to check
        process_model: The process model to check against
        algorithm: The algorithm to use for conformance checking
            Options: "token_replay", "alignments"
        parameters: Parameters for the conformance checking algorithm
    
    Returns:
        A dictionary with conformance checking results
    
    Raises:
        ImportError: If PM4PY is not installed
        ValueError: If an unknown algorithm is specified
    """
    check_pm4py_available()
    
    parameters = parameters or {}
    pm4py_log = to_pm4py(event_log)
    
    if algorithm == "token_replay":
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
        
        # Process models can be of different types depending on the discovery algorithm
        if hasattr(process_model, "nodes") and not hasattr(process_model, "transitions"):
            # This is probably a DFG or similar - convert to Petri net
            net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(pm4py_log)
        else:
            # Assume it's already a compatible model
            net, initial_marking, final_marking = process_model
            
        # Apply token replay
        results = token_replay.apply(pm4py_log, net, initial_marking, final_marking, parameters=parameters)
        
        # Process results
        return {
            "fitness": token_replay.get_diagnostics_dataframe(pm4py_log, net, initial_marking, final_marking, results),
            "aggregate": {
                "fitness": sum(r["trace_fitness"] for r in results) / len(results) if results else 0,
                "produced_tokens": sum(r["produced_tokens"] for r in results),
                "consumed_tokens": sum(r["consumed_tokens"] for r in results),
                "missing_tokens": sum(r["missing_tokens"] for r in results),
                "remaining_tokens": sum(r["remaining_tokens"] for r in results)
            }
        }
        
    elif algorithm == "alignments":
        from pm4py.algo.conformance.alignments import algorithm as alignments
        
        # Convert to Petri net if needed
        if hasattr(process_model, "nodes") and not hasattr(process_model, "transitions"):
            net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(pm4py_log)
        else:
            net, initial_marking, final_marking = process_model
            
        # Apply alignments
        aligned_traces = alignments.apply_log(pm4py_log, net, initial_marking, final_marking, parameters=parameters)
        
        # Process results
        fitness_values = [aligned["fitness"] for aligned in aligned_traces]
        
        return {
            "alignments": aligned_traces,
            "aggregate": {
                "fitness": sum(fitness_values) / len(fitness_values) if fitness_values else 0,
                "min_fitness": min(fitness_values) if fitness_values else 0,
                "max_fitness": max(fitness_values) if fitness_values else 0
            }
        }
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                         "Available options: token_replay, alignments")