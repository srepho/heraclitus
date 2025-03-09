"""Data module for handling event logs and related data structures."""

from heraclitus.data.event_log import EventLog
from heraclitus.data.duckdb_connector import DuckDBConnector, eventlog_to_duckdb

# Import XES handler for XES format support
try:
    from heraclitus.data.xes_handler import import_xes, export_xes
    __all__ = ["EventLog", "DuckDBConnector", "eventlog_to_duckdb", "import_xes", "export_xes"]
except ImportError:
    __all__ = ["EventLog", "DuckDBConnector", "eventlog_to_duckdb"]