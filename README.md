# Heraclitus

<p align="center">
  <img src="heraclitus_logo.webp" alt="Heraclitus Logo" width="250">
</p>

A Python library aimed at making Process Mining accessible to new users. Works well with PM4PY but adds additional features for analysis and visualization.

## Features

- **EventLog Management**: Easy to use interface for loading, filtering, and manipulating process event data
- **Process Visualization**: 
  - Static process maps and activity frequency charts
  - **NEW**: Interactive visualizations with Plotly
  - **NEW**: Timeline visualizations and bottleneck dashboards
- **Time Metrics**: Calculate cycle times, waiting times, and processing times
- **Statistical Analysis**: 
  - Compare process variants and identify bottlenecks
  - Distribution fitting and hypothesis testing
- **Large Dataset Support**:
  - **NEW**: DuckDB integration for handling datasets larger than memory
  - Efficient querying and filtering of large process logs

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/heraclitus.git
cd heraclitus
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"

# For machine learning features (coming soon)
pip install -e ".[ml]"
```

## Quick Start

```python
import pandas as pd
from heraclitus.data import EventLog
from heraclitus.visualization import create_interactive_process_map
from heraclitus.metrics import calculate_cycle_time

# Create an event log from a DataFrame
df = pd.read_csv("your_data.csv")
event_log = EventLog(df)

# Analyze cycle time
avg_time = calculate_cycle_time(event_log, unit="hours")
print(f"Average cycle time: {avg_time:.2f} hours")

# Create an interactive visualization
fig = create_interactive_process_map(event_log)
fig.write_html("process_map.html")  # Interactive HTML file
```

### Working with Large Datasets

```python
from heraclitus.data import DuckDBConnector

# Initialize DuckDB connector
db = DuckDBConnector("process_data.duckdb")

# Load large CSV file
db.load_csv("large_event_log.csv", table_name="events")

# Query and convert to EventLog
filtered_log = db.query_to_eventlog("""
    SELECT * FROM events 
    WHERE timestamp >= '2023-01-01' 
    AND activity = 'Process Application'
""")

# Analyze the filtered log
print(f"Filtered log contains {filtered_log.case_count()} cases")
```

## Requirements

- Python 3.10-3.12
- pandas
- matplotlib
- numpy
- scipy
- plotly
- duckdb

## Examples

Check out the examples directory for detailed usage examples:
- `basic_usage.py`: Simple EventLog operations
- `statistical_analysis.py`: Statistical comparisons and bottleneck analysis
- `interactive_visualization.py`: Interactive Plotly visualizations
- `duckdb_large_datasets.py`: Working with large datasets

## Development

See the [Developer Guide](DEV_GUIDE.md) for detailed information on the architecture and design principles.

Run tests:

```bash
pytest tests/
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and improvements.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License