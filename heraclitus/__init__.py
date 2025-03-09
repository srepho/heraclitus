"""
Heraclitus - A library for Process Mining and analysis.

A library aimed at making Process Mining accessible to new users,
working well with PM4PY but adding additional features.
"""

# Expose main classes
from heraclitus.data.event_log import EventLog

# Imports for process discovery features
try:
    from heraclitus.discovery.process_discovery import (
        discover_directly_follows_graph,
        discover_process_model,
        conformance_checking
    )
except ImportError:
    pass

__version__ = "0.3.0"