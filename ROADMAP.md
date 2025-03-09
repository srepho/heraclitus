# Heraclitus Project Roadmap

This document outlines the planned features and improvements for future releases of the Heraclitus project.

## Current Version (0.1.0)

- ✅ Core EventLog class with filtering capabilities
- ✅ Basic visualizations (process map, activity frequency)
- ✅ Time-based metrics (cycle time, waiting time, processing time)
- ✅ Initial test suite

## Short-term Goals (0.2.0)

- [ ] **Statistics Module**
  - [ ] Compare cycle times between groups
  - [ ] Bottleneck analysis
  - [ ] Distribution fitting
  - [ ] Statistical tests and p-value calculations

- [ ] **Enhanced Visualizations**
  - [ ] Interactive process maps (using Plotly)
  - [ ] Timeline visualizations
  - [ ] Cases over time charts
  - [ ] Duration histograms

- [ ] **Database Support**
  - [ ] SQL database connectors
  - [ ] EventLog.from_sql() factory method
  - [ ] Efficient querying for large datasets

- [ ] **Performance Optimization**
  - [ ] Vectorized operations for metrics
  - [ ] Caching for repeated calculations
  - [ ] Optional Dask integration for large datasets

## Medium-term Goals (0.3.0)

- [ ] **Machine Learning Module**
  - [ ] Outcome prediction
  - [ ] Duration prediction
  - [ ] Feature engineering utilities
  - [ ] Model evaluation tools

- [ ] **Anomaly Detection**
  - [ ] Process variant analysis
  - [ ] Outlier detection algorithms
  - [ ] Anomaly visualization

- [ ] **PM4PY Integration**
  - [ ] Convert between Heraclitus and PM4PY formats
  - [ ] Use PM4PY algorithms with Heraclitus EventLogs
  - [ ] Extend PM4PY visualizations

- [ ] **Documentation**
  - [ ] Complete API documentation
  - [ ] Tutorials with real-world examples
  - [ ] Jupyter notebooks showcasing features

## Long-term Vision

- [ ] **Advanced Process Discovery**
  - [ ] Custom process discovery algorithms
  - [ ] Conformance checking
  - [ ] BPMN export

- [ ] **Web Interface**
  - [ ] Interactive dashboard for process analysis
  - [ ] Shareable reports
  - [ ] Real-time monitoring

- [ ] **Integration with Other Tools**
  - [ ] XES format import/export
  - [ ] Integration with process mining standards
  - [ ] APIs for third-party extensions

## Contributing

If you're interested in working on any of these features, please check out our [CONTRIBUTING.md](CONTRIBUTING.md) guide and feel free to submit issues or pull requests.