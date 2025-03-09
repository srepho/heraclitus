"""Data module for handling event logs and related data structures."""

from heraclitus.data.event_log import EventLog
from heraclitus.data.duckdb_connector import DuckDBConnector, eventlog_to_duckdb

__all__ = ["EventLog", "DuckDBConnector", "eventlog_to_duckdb"]