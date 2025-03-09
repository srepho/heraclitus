# Contributing to Heraclitus

Thank you for your interest in contributing to Heraclitus! This document provides guidelines and instructions for contributing to this project.

## Development Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heraclitus.git
   cd heraclitus
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all modules, classes, and functions (following NumPy/Google style)
- Format imports: standard library, third-party, local imports, each group separated by a blank line
- Use meaningful variable and function names that reflect their purpose

## Testing

All new code should include appropriate tests:

```bash
# Run all tests
pytest tests/

# Test with coverage
pytest --cov=heraclitus

# Run a specific test file
pytest tests/test_specific.py
```

## Pull Request Process

1. Check the issue tracker to see if your issue/feature is already being addressed
2. Fork the repository and create a feature branch
3. Make your changes with appropriate tests and documentation
4. Ensure all tests pass and code is properly formatted
5. Submit a pull request with a clear description of the changes and any relevant issue numbers

## Development Workflow

1. Choose a task from the [ROADMAP.md](ROADMAP.md) or create an issue
2. Implement the feature/fix with tests
3. Update documentation as needed
4. Submit a pull request

## Project Structure

```
heraclitus/
├── heraclitus/            # Main package
│   ├── data/              # Data handling module
│   ├── visualization/     # Visualization module
│   ├── metrics/           # Metrics module
│   ├── statistics/        # Statistics module
│   ├── ml/                # Machine learning module
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── examples/              # Example scripts
└── docs/                  # Documentation
```

## Documentation

- Add docstrings to all public functions, classes, and methods
- Update README.md with any new features or changes
- Consider adding example usage to the examples directory

## Questions?

If you have any questions or need help, please open an issue on GitHub or contact the maintainers.

Thank you for contributing to Heraclitus!