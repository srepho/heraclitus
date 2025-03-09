# Claims Analysis Package - heraclitus

## Introduction - Developer Guide

## Overview & Goals

This document outlines the architecture, design considerations, and development tasks for the heraclitus Python package.  The goal is to create a robust, efficient, and extensible tool for analyzing claims-like data flowing through a series of statuses.  We're aiming for a balance between process-mining capabilities and user-friendly statistical/ML analysis.

**Target Audience:** Developers working on the heraclitus package.

## Core Design Principles

*   **Modularity:**  Separate concerns into well-defined modules (data, visualization, metrics, statistics, ml, utils).
*   **Extensibility:**  Design for easy addition of new metrics, visualizations, data sources, and ML models.
*   **Performance:**  Prioritize speed and efficiency, especially for large datasets.  Consider optimization techniques.
*   **Testability:**  Write comprehensive unit and integration tests for all components.
*   **Maintainability:**  Adhere to clean coding practices (PEP 8), use clear naming conventions, and document thoroughly.
*   **Data-Centric:** The `EventLog` class is the central data structure; all modules should interact with it.

## Module Breakdown and Development Tasks

### 1. `data` Module

*   **`EventLog` Class:**
    *   **Data Representation:**
        *   Internally, store data as a Pandas DataFrame. This provides efficient data manipulation and integration with other libraries.
        *   *Mandatory Columns:*  `case_id`, `activity`, `timestamp`.  These should be configurable via parameters to the constructor.  Enforce type checking (case_id: typically string/int, activity: string, timestamp: datetime).
        *   *Optional Columns:* Allow arbitrary additional columns (attributes).  Store a list of attribute column names.
    *   **Constructor (`__init__`) Implementation:**
        *   Accept various input types: Pandas DataFrame, file paths (CSV, Excel), SQL query + connection string.  Use `isinstance` to check input type.
        *   Implement lazy loading for file paths and SQL queries (only load data when needed).
        *   *Validation:*
            *   Check for required columns. Raise `ValueError` if missing.
            *   Ensure timestamps are valid datetime objects. Use `pd.to_datetime`, handling errors gracefully (e.g., provide options for parsing different date formats).
            *   Check for duplicate events (same case ID, activity, and timestamp).  Provide options: raise error, warn, or deduplicate (keeping the first or last occurrence).
    *   **Class Methods (Factory Methods):**
        *   `from_csv(filepath, ...)`: Use `pd.read_csv`. Handle common CSV parsing issues (delimiters, encodings, date parsing).  Pass keyword arguments through to `pd.read_csv`.
        *   `from_dataframe(df, ...)`: Directly use the provided DataFrame.  Perform validation checks.
        *   `from_sql(connection_string, query, ...)`: Use a database library (e.g., `psycopg2` for PostgreSQL, `sqlite3` for SQLite, `sqlalchemy` for general database support).  Handle connection errors.  Consider using connection pooling for efficiency.  Pass keyword arguments to the database query function.
    *   **Filtering Methods:**
        *   `filter_cases(case_ids)`:  Use boolean indexing (`df[df['case_id'].isin(case_ids)]`). Return a *new* `EventLog` instance.
        *   `filter_activities(activities)`: Similar to `filter_cases`. Return a *new* `EventLog` instance.
        *   `filter_time_range(start_time, end_time)`: Filter based on timestamp. Return a *new* `EventLog` instance.
    *   **Attribute Management:**
        *   `get_attributes()`: Return a list of attribute column names.
        *   `add_attribute(column_name, data)`: Add a new attribute column. Validate that the data length matches the number of rows.
    *   **`to_dataframe()`:** Return a *copy* of the internal DataFrame.
    *   **Development Tasks:**
        *   Implement thorough input validation and error handling.
        *   Write unit tests for all methods, covering different input types and edge cases.
        *   Profile performance with large datasets (e.g., using `cProfile`).  Identify bottlenecks.
        *   Consider using Dask for out-of-memory datasets (optional, advanced feature).

### 2. `visualization` Module

*   **`visualize_process_map(event_log, ...)`:**
    *   **Technology Choice:**
        *   *Primary Recommendation:*  `pm4py`. It's designed for process mining and offers good visualizations out of the box.
        *   *Alternative (for more control):*  Graphviz (via `graphviz` Python package) for static graphs.  Requires manual graph construction.
        *   *Alternative (for web-based interactivity):* mpld3 (combines matplotlib and D3.js).  More complex but allows for interactive exploration.
        *   *Decision:* Weigh the trade-offs between ease of use, customization, and interactivity. Start with `pm4py`.
    *   **Implementation (using `pm4py`):**
        *   Convert the `EventLog` to a `pm4py` event log object.
        *   Use `pm4py`'s discovery algorithms (e.g., Directly-Follows Graph, Heuristics Miner, Alpha Miner) to generate the process map.
        *   Customize appearance (colors, labels, edge thickness) using `pm4py`'s visualization functions.
    *   **Implementation (using Graphviz):**
        *   Manually create a `graphviz.Digraph` object.
        *   Iterate through the `EventLog` to count transitions between activities.
        *   Add nodes and edges to the graph, setting attributes (e.g., label, color, penwidth) based on frequency or performance metrics.
    *   **Parameters:**
        *   `frequency_threshold`: Filter out low-frequency edges.  Use this to simplify complex graphs.
        *   `performance_metric`:  Color edges based on a metric (e.g., average duration between activities).  Requires calling functions from the `metrics` module.
        *   `custom_node_attributes`, `custom_edge_attributes`: Allow users to specify custom attributes (e.g., node colors, edge styles) via dictionaries.
    *   **Development Tasks:**
        *   Implement both `pm4py` and Graphviz versions (if time allows).
        *   Write unit tests to ensure correct graph generation.
        *   Explore options for interactive graph manipulation (e.g., zooming, panning, highlighting paths).

*   **`plot_duration_histogram(event_log, activity=None, ...)`:**
    *   **Technology:** Matplotlib.
    *   **Implementation:**
        *   If `activity` is None, calculate overall cycle times (see `metrics` module).
        *   If `activity` is specified, calculate durations for that activity.
        *   Use `matplotlib.pyplot.hist` to create the histogram.
        *   Provide options for customizing bins, labels, titles, etc.
    *   **Development Tasks:**
        *   Handle cases where no data exists for the specified activity.
        *   Provide options for different time units (seconds, minutes, hours, days).

*   **`plot_activity_frequency(event_log, ...)`:**
    *   **Technology:** Matplotlib.
    *   **Implementation:**
        *   Use `event_log.to_dataframe()['activity'].value_counts()` to get frequencies.
        *   Use `matplotlib.pyplot.bar` to create the bar chart.
    *   **Development Tasks:**
        *   Sort bars by frequency or alphabetically.
        *   Allow customization of colors, labels, etc.

*   **`plot_cases_over_time(event_log, ...)`:**
    *   **Technology:** Matplotlib.
    *   **Implementation:**
        *   Group the `EventLog` by timestamp and count the number of active cases.
        *   Use `matplotlib.pyplot.plot` to create the line chart.
    *   **Development Tasks:**
        *   Handle different time granularities (daily, weekly, monthly).

*   **General Visualization Tasks:**
    *   Establish a consistent style guide for all plots (colors, fonts, etc.).
    *   Consider using Seaborn for enhanced aesthetics.
    *   Provide options for saving plots to different file formats (PNG, SVG, PDF).
    *   Explore Plotly for interactive visualizations (especially for web deployment).

### 3. `metrics` Module

*   **`calculate_cycle_time(event_log, case_id=None, start_activity=None, end_activity=None)`:**
    *   **Implementation:**
        *   If `case_id` is specified, filter the `EventLog` to that case.
        *   If `start_activity` and `end_activity` are specified, find the first occurrence of `start_activity` and the last occurrence of `end_activity` for each case.
        *   If not specified, use the first and last events for each case.
        *   Calculate the time difference (using `pd.Timedelta`).
        *   If `case_id` is None, return the average cycle time across all cases.  Consider returning other statistics (median, percentiles) as options.
        * Handle missing values and edge cases properly (e.g if a case doesn't have a start or end activity).
    *   **Development Tasks:**
        *   Implement robust handling of missing data (e.g., cases that never reach the `end_activity`).
        *   Provide options for different time units.
        *   Optimize performance (vectorized operations with Pandas).

*   **`calculate_waiting_time(event_log, activity, case_id=None)`:**
    *   **Implementation:**
        *   For each case (or the specified `case_id`), find the timestamp of the event *before* the `activity` occurs.
        *   Calculate the time difference between the preceding event and the `activity` event.
    *   **Development Tasks:**
        *   Handle cases where the `activity` is the first event in a case (no waiting time).

*   **`calculate_processing_time(event_log, activity, case_id=None)`:**
    *   **Implementation:**
        *   For each case (or the specified `case_id`), find the timestamp of the event *after* the `activity` occurs.
        *   Calculate the time difference between the `activity` event and the subsequent event.
    *   **Development Tasks:**
       *   Handle cases where the activity is the *last* event (open cases). Provide an option to either exclude these or use a current timestamp.

*   **`calculate_throughput(event_log, start_time=None, end_time=None)`:**
    *   **Implementation:**
        *   Filter the `EventLog` to the specified time range (if provided).
        *   Count the number of completed cases (cases that reach a designated "end" activity).
        *   Divide by the time duration (in the chosen unit).
    *   **Development Tasks:**
        *   Allow users to define what constitutes a "completed" case (e.g., a specific end activity).

*   **General Metrics Tasks:**
    *   Consider using Numba or Cython to optimize performance-critical calculations.
    *   Provide options for calculating different aggregation statistics (mean, median, standard deviation, percentiles).
    *   Implement a caching mechanism (e.g., using `functools.lru_cache`) to avoid redundant calculations.

### 4. `statistics` Module

*   **`compare_cycle_times(event_log, group_by_attribute, ...)`:**
    *   **Implementation:**
        *   Use `event_log.to_dataframe().groupby(group_by_attribute)` to group the data.
        *   Calculate cycle times for each group (using `calculate_cycle_time`).
        *   Use `scipy.stats` for statistical tests:
            *   For two groups: `scipy.stats.ttest_ind` (independent t-test) or `scipy.stats.mannwhitneyu` (Mann-Whitney U test, non-parametric).
            *   For more than two groups: `scipy.stats.f_oneway` (ANOVA) or `scipy.stats.kruskal` (Kruskal-Wallis test, non-parametric).
        *   Return a dictionary containing test statistics, p-values, and effect sizes (e.g., Cohen's d).
    *   **Development Tasks:**
        *   Allow users to choose the statistical test.
        *   Provide clear and informative output.
        *   Handle cases where a group has too few observations.
*   **`fit_distribution(data, distribution_type='exponential', ...)`:**
    *    *Purpose*: Allows for common distributions to be fitted to the data, with optional parameter and return information.
    *   **Implementation:**
        *   Use `scipy.stats` to fit various distributions (e.g., exponential, normal, Weibull, gamma).
        *   Allow users to specify the distribution type.
        *   Return the fitted distribution parameters.
    *   **Development Tasks:**
        *   Provide options for visualizing the fitted distribution (e.g., using a probability plot).

*   **`bottleneck_analysis(event_log, method='waiting_time', ...)`:**
    *   **Implementation:**
        *   If `method='waiting_time'`: Calculate average waiting times for each activity.
        *   If `method='queue_length'`:  Calculate the average number of cases in each status at any given time. (More complex implementation).
        *   Identify activities with the highest waiting times or queue lengths.
    *   **Development Tasks:**
        *   Implement different bottleneck analysis methods.
        *   Provide options for visualizing bottlenecks (e.g., highlighting them on the process map).

### 5. `ml` Module

*   **`predict_outcome(event_log, features, target, model_type='logistic_regression', ...)`:**
    *   **Implementation:**
        *   **Feature Engineering:**  This is crucial.  Create features from the `EventLog`:
            *   Durations of previous statuses.
            *   Number of times a case has visited a particular status.
            *   Values of attributes.
            *   Time-based features (e.g., day of the week, month).
        *   **Data Preparation:**  Split the data into training and testing sets (using `sklearn.model_selection.train_test_split`).
        *   **Model Training:**  Use scikit-learn models:
            *   `sklearn.linear_model.LogisticRegression`
            *   `sklearn.ensemble.RandomForestClassifier`
            *   `sklearn.svm.SVC`
            *   ...and others as needed.
        *   **Evaluation:**  Use appropriate metrics (accuracy, precision, recall, F1-score, ROC AUC).
        *   Return the trained model and evaluation metrics.
    *   **Development Tasks:**
        *   Implement robust feature engineering.
        *   Handle categorical features (one-hot encoding, label encoding).
        *   Provide options for hyperparameter tuning (e.g., using `sklearn.model_selection.GridSearchCV`).
        *   Allow users to save and load trained models.
        * Consider the use of `Pipeline` objects.

*   **`predict_duration(event_log, features, target_activity, model_type='linear_regression', ...)`:**
    *   Similar to `predict_outcome`, but use regression models (e.g., `sklearn.linear_model.LinearRegression`, `sklearn.ensemble.RandomForestRegressor`).
    *   Use appropriate evaluation metrics (MSE, RMSE, MAE, R-squared).

*   **`detect_anomalies(event_log, ...)`:**
    *   **Implementation:**
        *   *Option 1 (Clustering):*  Use `sklearn.cluster.KMeans` or `sklearn.cluster.DBSCAN` to cluster cases based on features.  Anomalies are cases that don't belong to any large cluster.
        *   *Option 2 (Outlier Detection):* Use `sklearn.ensemble.IsolationForest` or `sklearn.neighbors.LocalOutlierFactor`.
        *   *Option 3 (Time Series):* If timestamps are evenly spaced, use time series anomaly detection techniques (e.g., ARIMA, Prophet).
    *   **Development Tasks:**
        *   Implement multiple anomaly detection methods.
        *   Provide options for visualizing anomalies (e.g., highlighting them on the process map).

### 6. `utils` Module

*   **`load_config(filepath)`:**
    *   **Implementation:**
        *   Support YAML or JSON format. Use `yaml.safe_load` (for YAML) or `json.load` (for JSON).
        *   Define a schema for the configuration file (using a library like `schema` or `voluptuous` for validation - highly recommended).
        *   Handle file not found errors gracefully.
        *   Allow for nested configurations (e.g., sections for data mapping, visualization, database connections).
    *   **Development Tasks:**
        *   Implement robust schema validation.
        *   Provide clear error messages if the configuration file is invalid.

*   **`register_metric(name, function)`:**
    *   **Implementation:**
        *   Store custom metrics in a global dictionary (e.g., `_registered_metrics = {}`).
        *   Validate that `function` is callable.
        *   Allow overwriting existing metrics (with a warning).
    *  **Considerations**: How will the registered function access EventLog? Pass as parameter?