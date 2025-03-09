# Heraclitus Development Guide

## Build & Test Commands
```bash
# Install package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data.py

# Run tests with coverage
pytest --cov=heraclitus

# Run linting
flake8 heraclitus/
```

## Code Style Guidelines
- **PEP 8** compliant code (indentation: 4 spaces)
- Use **type hints** for all function parameters and return values
- **Imports**: Group standard library, third-party, and local imports with a blank line between groups
- **Naming**:
  - Classes: CamelCase (`EventLog`, `ProcessMap`)
  - Functions/methods: snake_case (`calculate_cycle_time()`)
  - Variables: snake_case (`event_log`, `case_id`)
- **Documentation**: Docstrings for all modules, classes, and functions (NumPy/Google style)
- **Error handling**: Use explicit exception types with informative messages
- The `EventLog` class is central - all modules should interact with this data structure
- Prioritize performance with vectorized operations when working with DataFrames
- Write comprehensive unit tests for all functionality