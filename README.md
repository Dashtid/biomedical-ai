# Medical Image Segmentation with U-Net Architectures

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![UV Package Manager](https://img.shields.io/badge/package%20manager-UV-orange.svg)](https://github.com/astral-sh/uv)

A professional, production-ready implementation of U-Net architectures for medical image segmentation. This package provides state-of-the-art deep learning models for segmenting medical images from CT, MRI, and other imaging modalities.

## Overview

This project implements state-of-the-art deep learning models for medical image segmentation, specifically designed to handle multi-expert annotations and quantify inter-rater variability. The implementation includes multiple U-Net variants with advanced features such as spatial dropout, LSTM-enhanced decoders, and deep architectures.

**Key Features:**
- Multiple U-Net architecture variants (standard, deep, LSTM-enhanced)
- Multi-expert ensemble evaluation methodology
- Comprehensive data augmentation pipeline
- DICE loss with additional metrics (IoU, precision, recall)
- Configurable training via YAML files
- Professional package structure with UV dependency management
- Extensive documentation and type hints

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/Dashtid/medical-image-segmentation.git
cd medical-image-segmentation

# Install dependencies
uv sync
```

## Quick Start

```bash
# Train a model with default configuration
uv run python scripts/train.py --config configs/brain_growth.yaml

# Train for specific expert (multi-expert datasets)
uv run python scripts/train.py --config configs/kidney.yaml --expert 1

# Evaluate trained models
uv run python scripts/evaluate.py --config configs/brain_growth.yaml --model-dir models/

# Run inference on new data
uv run python scripts/inference.py --model models/unet.h5 --input data/test/ --output results/
```

## Available Datasets

This toolkit supports various medical imaging datasets. Example configurations are provided for:

- **Brain MRI**: Brain tumor and growth segmentation
- **Kidney CT**: Renal structure segmentation
- **Prostate MRI**: Prostate gland segmentation
- **Custom datasets**: Easily adaptable via YAML configuration files

For dataset downloads, see publicly available resources:
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) - 2,633 images across 10 tasks
- [AMOS Dataset](https://amos22.grand-challenge.org/) - 500 CT + 100 MRI scans with 15 organ annotations
- [IMIS-Bench](https://github.com/IMIS-Bench/IMIS-Bench) - 6.4M images with 273.4M masks across 14 modalities

## Architecture Variants

1. **Standard U-Net** - Classic architecture with spatial dropout
2. **Deep U-Net** - Extended with 5 encoding levels
3. **U-Net with LSTM** - ConvLSTM2D-enhanced decoder
4. **Deep U-Net with LSTM** - Combines depth and LSTM

## License

MIT License - See LICENSE file for details.

## Contact

David Dashti - david.dashti@hermesmedical.com