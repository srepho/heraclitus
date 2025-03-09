# Heraclitus

<p align="center">
  <img src="heraclitus_logo.webp" alt="Heraclitus Logo" width="250">
</p>

A Python library aimed at making Process Mining accessible to new users. Works well with PM4PY but adds additional features for analysis and visualization.

## Features

- **EventLog Management**: Easy to use interface for loading, filtering, and manipulating process event data
- **Process Visualization**: Generate process maps and activity frequency charts
- **Time Metrics**: Calculate cycle times, waiting times, and processing times
- **Statistical Analysis**: Compare process variants and identify bottlenecks
- **Machine Learning Integration**: Predictive models for process outcomes and durations (coming soon)

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/heraclitus.git
cd heraclitus
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from heraclitus.data import EventLog
from heraclitus.visualization import visualize_process_map
from heraclitus.metrics import calculate_cycle_time

# Create an event log from a DataFrame
df = pd.read_csv("your_data.csv")
event_log = EventLog(df)

# Analyze cycle time
avg_time = calculate_cycle_time(event_log, unit="hours")
print(f"Average cycle time: {avg_time:.2f} hours")

# Visualize the process
fig = visualize_process_map(event_log)
fig.savefig("process_map.png")
```

## Requirements

- Python 3.10-3.12
- pandas
- matplotlib
- numpy
- pm4py (optional for advanced process mining features)

## Development

See the [Developer Guide](DEV_GUIDE.md) for detailed information on the architecture and design principles.

Run tests:

```bash
pytest tests/
```

## License

MIT License