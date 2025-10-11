# Contributing to Medical Image Segmentation

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/medical-image-segmentation.git
   cd medical-image-segmentation
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/medical-image-segmentation.git
   ```

## Development Setup

### Using UV (Recommended)

```bash
# Install dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

### Traditional Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Making Changes

### 1. Create a Branch

Create a descriptive branch name:

```bash
# Feature branch
git checkout -b feature/add-attention-unet

# Bug fix branch
git checkout -b fix/dice-loss-edge-case

# Documentation branch
git checkout -b docs/update-architecture
```

### 2. Make Your Changes

- Write clean, documented code
- Add type hints to all functions
- Include docstrings (Google style)
- Update tests for new features
- Update documentation

### 3. Follow Code Style

We use several tools to maintain code quality:

- **Black** - Code formatting
- **isort** - Import sorting
- **Ruff** - Fast linting
- **mypy** - Type checking

Run all checks:

```bash
# Format code
uv run black src/ tests/

# Sort imports
uv run isort src/ tests/

# Lint
uv run ruff check src/ tests/ --fix

# Type check
uv run mypy src/med_seg
```

Or use pre-commit:

```bash
uv run pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=med_seg --cov-report=html

# Run specific test file
uv run pytest tests/test_models.py

# Run specific test
uv run pytest tests/test_losses.py::TestLossFunctions::test_dice_coefficient_perfect
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include both positive and negative test cases

Example:

```python
def test_dice_coefficient_perfect():
    """Test DICE coefficient with perfect prediction."""
    y_true = tf.constant([[[[1.0]], [[1.0]], [[0.0]], [[0.0]]]])
    y_pred = tf.constant([[[[1.0]], [[1.0]], [[0.0]], [[0.0]]]])

    dice = dice_coefficient(y_true, y_pred)

    assert float(dice) > 0.99  # Should be very close to 1.0
```

## Documentation

### Code Documentation

Use Google-style docstrings:

```python
def train_model(
    model: keras.Model,
    data_generator: Iterator,
    epochs: int = 100
) -> keras.callbacks.History:
    """Train a segmentation model.

    This function orchestrates the training process using the provided
    data generator and configuration.

    Args:
        model: Keras model to train
        data_generator: Generator yielding (images, masks) batches
        epochs: Number of training epochs

    Returns:
        Training history object containing loss and metrics

    Raises:
        ValueError: If model is not compiled

    Example:
        >>> from med_seg.models import UNet
        >>> model_builder = UNet(input_size=256)
        >>> model = model_builder.build()
        >>> history = train_model(model, train_gen, epochs=50)
    """
    ...
```

### Documentation Files

- Update `README.md` for user-facing changes
- Update `docs/ARCHITECTURE.md` for architectural changes
- Add examples to `notebooks/` for new features

## Submitting Changes

### 1. Commit Your Changes

Use conventional commit messages:

```bash
# Feature
git commit -m "feat: add attention U-Net architecture"

# Bug fix
git commit -m "fix: correct DICE loss calculation for edge cases"

# Documentation
git commit -m "docs: update training examples in README"

# Tests
git commit -m "test: add unit tests for data augmentation"

# Refactoring
git commit -m "refactor: simplify data loader logic"

# Performance
git commit -m "perf: optimize DICE coefficient calculation"

# CI/CD
git commit -m "ci: add Docker build to GitHub Actions"
```

### 2. Push to Your Fork

```bash
git push origin your-branch-name
```

### 3. Create Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:
   - **Title**: Clear, descriptive title
   - **Description**: What changes were made and why
   - **Related Issues**: Link any related issues
   - **Testing**: How were the changes tested
   - **Screenshots**: If applicable

### 4. PR Review Process

- Respond to review comments
- Make requested changes
- Keep the PR focused (one feature/fix per PR)
- Ensure all CI checks pass

## Code Style

### Python Style

- Follow PEP 8
- Use type hints (Python 3.10+ syntax)
- Maximum line length: 100 characters
- Use descriptive variable names
- Avoid abbreviations unless common

### Type Hints

```python
from typing import List, Dict, Optional, Tuple
import numpy as np
from tensorflow import keras

def process_images(
    images: np.ndarray,
    masks: Optional[np.ndarray] = None,
    batch_size: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Process images with optional masks."""
    ...
```

### Docstrings

- All public functions, classes, and methods must have docstrings
- Use Google style
- Include Args, Returns, Raises, and Examples sections

### Imports

Organize imports in this order:

1. Standard library
2. Third-party packages
3. Local modules

```python
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from med_seg.models import UNet
from med_seg.training import dice_loss
```

## Project-Specific Guidelines

### Adding New Models

1. Create new file in `src/med_seg/models/`
2. Implement builder class with `build()` method
3. Add comprehensive docstrings
4. Create unit tests in `tests/test_models.py`
5. Update `docs/ARCHITECTURE.md`
6. Add example to notebooks

### Adding New Loss Functions

1. Add to `src/med_seg/training/losses.py`
2. Follow signature: `(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor`
3. Include smoothing parameter if applicable
4. Add unit tests in `tests/test_losses.py`
5. Document in architecture docs

### Adding New Datasets

1. Create YAML config in `configs/`
2. Follow existing config structure
3. Document dataset-specific preprocessing
4. Add to README

## Questions?

If you have questions:

1. Check existing issues and discussions
2. Read the documentation in `docs/`
3. Open a new issue with the "question" label

## Recognition

Contributors will be acknowledged in:
- `README.md` contributors section
- Release notes
- Project documentation

Thank you for contributing to medical image segmentation! 🎉
