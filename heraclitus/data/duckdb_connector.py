"""
DuckDB connector for handling large datasets in Heraclitus.
"""
from typing import Optional, Union, List, Dict, Any, Tuple
import os
import pandas as pd
import duckdb
from pathlib import Path

from heraclitus.data import EventLog


class DuckDBConnector:
    """
    DuckDB connector for efficient handling of large event logs.
    
    This class provides utilities for loading large datasets into DuckDB,
    performing efficient queries, and converting results to EventLog objects.
    
    Attributes:
        conn: DuckDB connection
        db_path: Path to the database file (optional)
        table_registry: Dictionary of registered tables
    """
    
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize a DuckDB connector.
        
        Args:
            db_path: Optional path to the database file. If None, an in-memory database is used.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path)
        self.table_registry = {}
    
    def load_csv(
        self,
        filepath: Union[str, Path],
        table_name: str,
        case_id_column: str = "case_id",
        activity_column: str = "activity",
        timestamp_column: str = "timestamp",
        timestamp_format: Optional[str] = None,
        delimiter: str = ",",
        header: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Load a CSV file into DuckDB.
        
        Args:
            filepath: Path to the CSV file
            table_name: Name for the table in DuckDB
            case_id_column: Name of the case ID column
            activity_column: Name of the activity column
            timestamp_column: Name of the timestamp column
            timestamp_format: Optional format string for parsing timestamps
            delimiter: CSV delimiter character
            header: Whether the CSV file has a header row
            **kwargs: Additional parameters passed to duckdb.read_csv
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Register table metadata
        self.table_registry[table_name] = {
            "source": filepath,
            "type": "csv",
            "case_id_column": case_id_column,
            "activity_column": activity_column,
            "timestamp_column": timestamp_column,
        }
        
        # Create timestamp parsing SQL if format is provided
        timestamp_sql = ""
        if timestamp_format:
            timestamp_sql = f", {timestamp_column} as VARCHAR"
        
        # Load the CSV file into DuckDB
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS 
            SELECT * {timestamp_sql}
            FROM read_csv_auto(
                '{filepath}', 
                delim='{delimiter}',
                header={str(header).lower()},
                all_varchar={str(kwargs.get('all_varchar', False)).lower()}
            )
            """
        )
        
        # Process timestamp column if format is provided
        if timestamp_format:
            self.conn.execute(
                f"""
                ALTER TABLE {table_name} ALTER {timestamp_column} 
                TYPE TIMESTAMP 
                USING strptime({timestamp_column}, '{timestamp_format}')
                """
            )
        
        # Create indices for efficient querying
        self.conn.execute(f"CREATE INDEX idx_{table_name}_{case_id_column} ON {table_name}({case_id_column})")
        self.conn.execute(f"CREATE INDEX idx_{table_name}_{timestamp_column} ON {table_name}({timestamp_column})")
        self.conn.execute(f"CREATE INDEX idx_{table_name}_{activity_column} ON {table_name}({activity_column})")
    
    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        case_id_column: str = "case_id",
        activity_column: str = "activity",
        timestamp_column: str = "timestamp",
    ) -> None:
        """
        Load a pandas DataFrame into DuckDB.
        
        Args:
            df: Source DataFrame
            table_name: Name for the table in DuckDB
            case_id_column: Name of the case ID column
            activity_column: Name of the activity column
            timestamp_column: Name of the timestamp column
        """
        # Register table metadata
        self.table_registry[table_name] = {
            "source": "dataframe",
            "type": "dataframe",
            "case_id_column": case_id_column,
            "activity_column": activity_column,
            "timestamp_column": timestamp_column,
        }
        
        # Ensure timestamp column is datetime
        if timestamp_column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Load the DataFrame into DuckDB
        self.conn.register("temp_df", df)
        self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM temp_df")
        self.conn.unregister("temp_df")
        
        # Create indices for efficient querying
        self.conn.execute(f"CREATE INDEX idx_{table_name}_{case_id_column} ON {table_name}({case_id_column})")
        self.conn.execute(f"CREATE INDEX idx_{table_name}_{timestamp_column} ON {table_name}({timestamp_column})")
        self.conn.execute(f"CREATE INDEX idx_{table_name}_{activity_column} ON {table_name}({activity_column})")
    
    def query_to_eventlog(
        self,
        query: str,
        table_name: Optional[str] = None,
        case_id_column: Optional[str] = None,
        activity_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
    ) -> EventLog:
        """
        Execute a SQL query and convert the result to an EventLog.
        
        Args:
            query: SQL query to execute
            table_name: Optional table name to get metadata from
            case_id_column: Optional case ID column name (overrides table metadata)
            activity_column: Optional activity column name (overrides table metadata)
            timestamp_column: Optional timestamp column name (overrides table metadata)
        
        Returns:
            An EventLog instance containing the query results
        
        Raises:
            ValueError: If required column information is not provided
        """
        # Get metadata from registry if not explicitly provided
        if table_name and table_name in self.table_registry:
            metadata = self.table_registry[table_name]
            case_id_column = case_id_column or metadata["case_id_column"]
            activity_column = activity_column or metadata["activity_column"]
            timestamp_column = timestamp_column or metadata["timestamp_column"]
        
        # Ensure we have all required column names
        if not all([case_id_column, activity_column, timestamp_column]):
            raise ValueError(
                "Column names must be provided either directly or through a registered table name"
            )
        
        # Execute the query and convert to DataFrame
        result_df = self.conn.execute(query).fetchdf()
        
        # Create EventLog
        return EventLog(
            result_df,
            case_id_column=case_id_column,
            activity_column=activity_column,
            timestamp_column=timestamp_column
        )
    
    def get_eventlog(
        self,
        table_name: str,
        case_ids: Optional[List[str]] = None,
        activities: Optional[List[str]] = None,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
    ) -> EventLog:
        """
        Get an EventLog from a DuckDB table with optional filtering.
        
        Args:
            table_name: Name of the table
            case_ids: Optional list of case IDs to filter by
            activities: Optional list of activities to filter by
            start_time: Optional start timestamp for filtering
            end_time: Optional end timestamp for filtering
            limit: Optional row limit for the query
        
        Returns:
            An EventLog instance containing the filtered data
        
        Raises:
            ValueError: If the table is not registered
        """
        if table_name not in self.table_registry:
            raise ValueError(f"Table '{table_name}' is not registered")
        
        metadata = self.table_registry[table_name]
        case_id_column = metadata["case_id_column"]
        activity_column = metadata["activity_column"]
        timestamp_column = metadata["timestamp_column"]
        
        # Build query with filters
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        
        if case_ids:
            case_ids_str = ", ".join([f"'{case_id}'" for case_id in case_ids])
            query += f" AND {case_id_column} IN ({case_ids_str})"
        
        if activities:
            activities_str = ", ".join([f"'{activity}'" for activity in activities])
            query += f" AND {activity_column} IN ({activities_str})"
        
        if start_time:
            query += f" AND {timestamp_column} >= TIMESTAMP '{start_time}'"
        
        if end_time:
            query += f" AND {timestamp_column} <= TIMESTAMP '{end_time}'"
        
        # Add sorting
        query += f" ORDER BY {case_id_column}, {timestamp_column}"
        
        # Add limit if specified
        if limit:
            query += f" LIMIT {limit}"
        
        # Execute the query
        return self.query_to_eventlog(
            query,
            table_name=table_name
        )
    
    def get_unique_values(
        self,
        table_name: str,
        column_name: str,
        limit: Optional[int] = None,
    ) -> List[Any]:
        """
        Get unique values for a column in a DuckDB table.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            limit: Optional limit for the number of unique values
        
        Returns:
            A list of unique values
        
        Raises:
            ValueError: If the table is not registered
        """
        if table_name not in self.table_registry:
            raise ValueError(f"Table '{table_name}' is not registered")
        
        query = f"SELECT DISTINCT {column_name} FROM {table_name}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = self.conn.execute(query).fetchall()
        return [row[0] for row in result]
    
    def get_case_count(self, table_name: str) -> int:
        """
        Get the number of unique cases in a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            The number of unique cases
        """
        if table_name not in self.table_registry:
            raise ValueError(f"Table '{table_name}' is not registered")
        
        metadata = self.table_registry[table_name]
        case_id_column = metadata["case_id_column"]
        
        result = self.conn.execute(
            f"SELECT COUNT(DISTINCT {case_id_column}) FROM {table_name}"
        ).fetchone()
        
        return result[0]
    
    def get_activity_count(self, table_name: str) -> int:
        """
        Get the number of unique activities in a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            The number of unique activities
        """
        if table_name not in self.table_registry:
            raise ValueError(f"Table '{table_name}' is not registered")
        
        metadata = self.table_registry[table_name]
        activity_column = metadata["activity_column"]
        
        result = self.conn.execute(
            f"SELECT COUNT(DISTINCT {activity_column}) FROM {table_name}"
        ).fetchone()
        
        return result[0]
    
    def get_time_range(self, table_name: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the minimum and maximum timestamp in a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            A tuple of (min_timestamp, max_timestamp)
        """
        if table_name not in self.table_registry:
            raise ValueError(f"Table '{table_name}' is not registered")
        
        metadata = self.table_registry[table_name]
        timestamp_column = metadata["timestamp_column"]
        
        result = self.conn.execute(
            f"SELECT MIN({timestamp_column}), MAX({timestamp_column}) FROM {table_name}"
        ).fetchone()
        
        return pd.Timestamp(result[0]), pd.Timestamp(result[1])
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a registered table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            A dictionary containing table information
        """
        if table_name not in self.table_registry:
            raise ValueError(f"Table '{table_name}' is not registered")
        
        info = self.table_registry[table_name].copy()
        
        # Add record count
        record_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        info["record_count"] = record_count
        
        # Add case count
        info["case_count"] = self.get_case_count(table_name)
        
        # Add activity count
        info["activity_count"] = self.get_activity_count(table_name)
        
        # Add time range
        min_time, max_time = self.get_time_range(table_name)
        info["time_range"] = {
            "min": min_time,
            "max": max_time,
            "duration_days": (max_time - min_time).days
        }
        
        return info
    
    def execute(self, query: str) -> duckdb.DuckDBPyRelation:
        """
        Execute a SQL query directly.
        
        Args:
            query: SQL query to execute
        
        Returns:
            DuckDB query result
        """
        return self.conn.execute(query)
    
    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
    
    def __del__(self) -> None:
        """Clean up resources when the object is deleted."""
        try:
            self.close()
        except:
            pass


def eventlog_to_duckdb(
    event_log: EventLog,
    connector: DuckDBConnector,
    table_name: str,
) -> None:
    """
    Save an EventLog to a DuckDB database.
    
    Args:
        event_log: The EventLog to save
        connector: DuckDBConnector instance
        table_name: Name for the table in DuckDB
    """
    df = event_log.to_dataframe()
    
    connector.load_dataframe(
        df,
        table_name=table_name,
        case_id_column=event_log.case_id_column,
        activity_column=event_log.activity_column,
        timestamp_column=event_log.timestamp_column,
    )