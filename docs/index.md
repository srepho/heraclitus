# Heraclitus Documentation

Welcome to the Heraclitus documentation! This guide will help you understand and use the Heraclitus library for process mining and analysis.

## Overview

Heraclitus is a Python library aimed at making Process Mining accessible to new users. It provides tools for:

- EventLog management and manipulation
- Process visualization (static and interactive)
- Time-based metrics calculation
- Statistical analysis of process data
- Predictive modeling and anomaly detection
- Handling large datasets with DuckDB integration

## Getting Started

### Installation

```bash
# Basic installation
pip install heraclitus

# With development dependencies
pip install heraclitus[dev]

# With machine learning capabilities
pip install heraclitus[ml]
```

### Basic Usage

```python
import pandas as pd
from heraclitus.data import EventLog
from heraclitus.visualization import create_interactive_process_map

# Create an event log from a DataFrame
df = pd.read_csv("your_data.csv")
event_log = EventLog(df)

# Create an interactive visualization
fig = create_interactive_process_map(event_log)
fig.write_html("process_map.html")
```

## Core Components

Heraclitus is organized into several modules:

1. **[Data Module](data_guide.md)**: Core data structures and utilities
   - EventLog class
   - DuckDB integration for large datasets

2. **[Visualization Module](visualization_guide.md)**: Process visualization tools
   - Static visualizations with Matplotlib
   - Interactive visualizations with Plotly

3. **[Metrics Module](metrics_guide.md)**: Process performance metrics
   - Cycle time calculation
   - Waiting and processing time analysis

4. **[Statistics Module](statistics_guide.md)**: Statistical analysis tools
   - Comparison between process variants
   - Bottleneck analysis
   - Distribution fitting

5. **[Machine Learning Module](machine_learning_guide.md)**: Predictive analytics
   - Feature engineering
   - Outcome and duration prediction
   - Anomaly detection

## Guides and Tutorials

- [EventLog Management](tutorials/eventlog_management.md)
- [Process Visualization](tutorials/process_visualization.md)
- [Statistical Analysis](tutorials/statistical_analysis.md)
- [Working with Large Datasets](tutorials/large_datasets.md)
- [Machine Learning and Anomaly Detection](tutorials/machine_learning.md)

## API Reference

For detailed information about specific classes and functions, refer to the [API Reference](api_reference.md).

## Contributing

Contributions are welcome! See the [Contributing Guide](../CONTRIBUTING.md) for details on how to contribute to Heraclitus.