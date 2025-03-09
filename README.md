# Heraclitus

<p align="center">
  <img src="heraclitus_logo.webp" alt="Heraclitus Logo" width="250">
</p>

<p align="center">
  <a href="https://pypi.org/project/heraclitus/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/heraclitus.svg"></a>
  <a href="https://pypi.org/project/heraclitus/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/heraclitus.svg"></a>
  <a href="https://github.com/yourusername/heraclitus/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/yourusername/heraclitus.svg"></a>
  <a href="https://github.com/yourusername/heraclitus/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/yourusername/heraclitus.svg"></a>
</p>

A Python library aimed at making Process Mining accessible to new users. Works well with PM4PY but adds additional features for analysis and visualization.

## Features

- **EventLog Management**: Easy to use interface for loading, filtering, and manipulating process event data
- **Process Visualization**: 
  - Static process maps and activity frequency charts
  - Interactive visualizations with Plotly
  - Timeline visualizations and bottleneck dashboards
- **Time Metrics**: Calculate cycle times, waiting times, and processing times
- **Statistical Analysis**: 
  - Compare process variants and identify bottlenecks
  - Distribution fitting and hypothesis testing
- **Large Dataset Support**:
  - DuckDB integration for handling datasets larger than memory
  - Efficient querying and filtering of large process logs
- **Machine Learning**:
  - Predictive models for process outcomes
  - Duration prediction with regression and classification
  - Feature engineering tools for process data
- **Anomaly Detection**:
  - Identify unusual process behaviors
  - Detect outlier cases and process variants
  - Visualize anomalies for investigation
- **Process Discovery** (NEW in v0.2.0):
  - Custom process discovery algorithms
  - Conformance checking capabilities
  - BPMN export functionality
- **PM4PY Integration** (NEW in v0.2.0):
  - Seamless conversion between Heraclitus and PM4PY formats
  - Use PM4PY algorithms with Heraclitus EventLogs
  - Enhanced visualizations for process models
- **Performance Optimization** (NEW in v0.2.0):
  - Vectorized operations for metrics
  - Caching for repeated calculations

## Installation

```bash
# Install from PyPI
pip install heraclitus

# Install with PM4PY integration
pip install heraclitus[pm4py]

# Install with machine learning features
pip install heraclitus[ml]

# Install with all optional dependencies
pip install heraclitus[pm4py,ml,dev]
```

Or install from source:

```bash
git clone https://github.com/yourusername/heraclitus.git
cd heraclitus
pip install -e .
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

### Machine Learning and Anomaly Detection

```python
from heraclitus.ml import OutcomePredictor, ProcessAnomalyDetector, FeatureExtractor

# Extract features from event log
feature_extractor = FeatureExtractor(event_log)
features_df = feature_extractor.extract_case_features()

# Train a prediction model
predictor = OutcomePredictor()
model_info = predictor.train(event_log, model_type="random_forest")
predictions = predictor.predict(features_df)

# Detect process anomalies
anomaly_detector = ProcessAnomalyDetector()
anomaly_detector.train(event_log, method="isolation_forest")
anomalies = anomaly_detector.detect_anomalies(features_df)
```

## Requirements

- Python 3.10-3.12
- pandas
- matplotlib
- numpy
- scipy
- plotly
- duckdb
- scikit-learn (for ML features)

## Examples

Check out the examples directory for detailed usage examples:
- `basic_usage.py`: Simple EventLog operations
- `statistical_analysis.py`: Statistical comparisons and bottleneck analysis
- `interactive_visualization.py`: Interactive Plotly visualizations
- `duckdb_large_datasets.py`: Working with large datasets
- `machine_learning.py`: Predictive modeling and anomaly detection
- `process_discovery.py`: Process discovery and conformance checking (NEW)

## Documentation

See the [Documentation](docs/index.md) for comprehensive guides and tutorials.

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