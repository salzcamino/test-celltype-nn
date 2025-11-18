# Contributing to CellType-NN

Thank you for considering contributing to CellType-NN! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/celltype-nn.git
   cd celltype-nn
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original/celltype-nn.git
   ```

## Development Setup

### Python Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### R Development

```R
# Install development dependencies
install.packages(c("devtools", "testthat", "styler", "lintr"))

# Load package in development mode
devtools::load_all()
```

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** when creating an issue
3. **Provide minimal reproducible example**
4. **Include version information** and environment details

### Suggesting Enhancements

1. **Check existing feature requests**
2. **Use the feature request template**
3. **Clearly describe the use case**
4. **Provide example usage** if possible

### Code Contributions

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Update documentation**

5. **Run tests** to ensure everything works:
   ```bash
   # Python
   pytest tests/ -v

   # R
   devtools::test()
   ```

6. **Commit your changes**:
   ```bash
   git commit -m "Add: brief description of changes"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**

## Coding Standards

### Python Code Style

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters
- **Formatter**: Black
- **Import sorter**: isort
- **Linter**: flake8
- **Type hints**: Encouraged for all functions

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check style
flake8 src/ tests/
mypy src/ --ignore-missing-imports
```

### R Code Style

We follow the **tidyverse style guide**:

- **Formatter**: styler
- **Linter**: lintr

```R
# Format code
styler::style_pkg()

# Lint code
lintr::lint_package()
```

### Documentation

#### Python Docstrings

Use **Google-style** docstrings:

```python
def my_function(arg1: int, arg2: str) -> bool:
    """Brief description of function.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid arguments provided
    """
    pass
```

#### R Documentation

Use **roxygen2** for R functions:

```R
#' Brief description
#'
#' @param arg1 Description of arg1
#' @param arg2 Description of arg2
#' @return Description of return value
#' @export
#' @examples
#' my_function(1, "test")
my_function <- function(arg1, arg2) {
  # Implementation
}
```

## Testing Guidelines

### Python Tests

- **Location**: `tests/` directory
- **Framework**: pytest
- **Coverage target**: >80%
- **Naming**: `test_*.py` files, `test_*` functions

```python
import pytest

def test_something():
    """Test description."""
    result = my_function(input_data)
    assert result == expected
```

**Running tests**:
```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=celltype_nn --cov-report=html

# Specific test file
pytest tests/test_models.py

# Specific test
pytest tests/test_models.py::test_forward_pass
```

### R Tests

- **Location**: `tests/testthat/` directory
- **Framework**: testthat
- **Naming**: `test-*.R` files

```R
test_that("description of test", {
  result <- my_function(input)
  expect_equal(result, expected)
})
```

**Running tests**:
```R
# All tests
devtools::test()

# With coverage
covr::package_coverage()

# Specific file
testthat::test_file("tests/testthat/test-models.R")
```

### Test Requirements

1. **Add tests for all new features**
2. **Add tests for bug fixes**
3. **Ensure all tests pass** before submitting PR
4. **Maintain or improve code coverage**
5. **Include both unit and integration tests** when applicable

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] CHANGELOG updated (for significant changes)

### PR Title Format

Use conventional commits format:
- `feat: add new model architecture`
- `fix: correct data loading bug`
- `docs: update README`
- `test: add integration tests`
- `refactor: simplify preprocessing code`
- `chore: update dependencies`

### Review Process

1. **Automated checks** must pass (GitHub Actions)
2. **Code review** by at least one maintainer
3. **Address feedback** promptly
4. **Squash commits** before merge (if requested)

### After Merge

- Delete your feature branch
- Update your fork:
  ```bash
  git checkout main
  git pull upstream main
  git push origin main
  ```

## Common Tasks

### Adding a New Model

1. **Create model class** in `src/celltype_nn/models/`
2. **Add to `__init__.py`** for easy import
3. **Write tests** in `tests/test_models.py`
4. **Add documentation** with examples
5. **Update README** if significant

### Adding a New Preprocessing Method

1. **Add function** to `src/celltype_nn/preprocessing/preprocess.py`
2. **Write tests** in `tests/test_preprocessing.py`
3. **Document parameters** and returns
4. **Add usage example** in docstring

### Updating Dependencies

1. **Update `requirements.txt`** or `DESCRIPTION`
2. **Test thoroughly** with new versions
3. **Update documentation** if API changed
4. **Note in CHANGELOG**

## Release Process

(For maintainers)

1. **Update version** in `setup.py` and `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Run full test suite**
4. **Create git tag**: `git tag -a v0.1.0 -m "Release v0.1.0"`
5. **Push tag**: `git push origin v0.1.0`
6. **Create GitHub release**
7. **Publish to PyPI** (Python) and CRAN (R)

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create an issue with bug report template
- **Features**: Create an issue with feature request template
- **Chat**: Join our community (if applicable)

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md
- GitHub contributors page
- Release notes

Thank you for contributing to CellType-NN! 🎉
