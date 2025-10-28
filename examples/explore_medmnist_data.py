#!/usr/bin/env python
"""Explore MedMNIST PathMNIST dataset structure and visualize samples.

This script helps understand the downloaded MedMNIST data format,
splits, class distribution, and visualizes example images.

Usage:
    python examples/explore_medmnist_data.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_medmnist_data(data_dir: Path):
    """Load MedMNIST npz files.

    Args:
        data_dir: Directory containing train/val/test npz files

    Returns:
        Dictionary with train/val/test splits
    """
    data = {}

    for split in ['train', 'val', 'test']:
        npz_path = data_dir / f"{split}.npz"
        if npz_path.exists():
            print(f"[*] Loading {split} split from {npz_path}")
            loaded = np.load(npz_path)
            data[split] = {
                'images': loaded['images'],
                'labels': loaded['labels']
            }
            print(f"    Images shape: {loaded['images'].shape}")
            print(f"    Labels shape: {loaded['labels'].shape}")
        else:
            print(f"[!] {split} split not found at {npz_path}")

    return data


def analyze_dataset(data: dict):
    """Analyze dataset statistics.

    Args:
        data: Dictionary with train/val/test splits
    """
    print("\n" + "=" * 60)
    print("[+] Dataset Statistics")
    print("=" * 60)

    for split, split_data in data.items():
        images = split_data['images']
        labels = split_data['labels']

        print(f"\n{split.upper()} Split:")
        print(f"  Number of samples: {len(images)}")
        print(f"  Image shape: {images[0].shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Image value range: [{images.min()}, {images.max()}]")
        print(f"  Number of classes: {len(np.unique(labels))}")
        print(f"  Class distribution:")

        unique, counts = np.unique(labels, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = (count / len(labels)) * 100
            print(f"    Class {cls}: {count} samples ({percentage:.1f}%)")


def visualize_samples(data: dict, num_samples: int = 16):
    """Visualize random samples from each split.

    Args:
        data: Dictionary with train/val/test splits
        num_samples: Number of samples to visualize per split
    """
    print("\n[*] Creating visualizations...")

    for split, split_data in data.items():
        images = split_data['images']
        labels = split_data['labels']

        # Select random samples
        indices = np.random.choice(len(images), size=min(num_samples, len(images)), replace=False)

        # Create figure
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        fig.suptitle(f'PathMNIST {split.upper()} Split - Random Samples', fontsize=14)

        for idx, ax in enumerate(axes.flat):
            if idx < len(indices):
                img = images[indices[idx]]
                label = labels[indices[idx]][0]  # Labels are shape (1,)

                # Handle RGB or grayscale
                if img.ndim == 3 and img.shape[2] == 3:
                    ax.imshow(img)
                else:
                    ax.imshow(img.squeeze(), cmap='gray')

                ax.set_title(f'Class {label}')
                ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout()

        # Save figure
        output_path = Path('results') / f'medmnist_{split}_samples.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    Saved visualization to {output_path}")
        plt.close()


def visualize_class_examples(data: dict, samples_per_class: int = 5):
    """Visualize examples from each class.

    Args:
        data: Dictionary with train/val/test splits
        samples_per_class: Number of samples to show per class
    """
    print("\n[*] Creating class-wise visualization...")

    # Use training data for class examples
    images = data['train']['images']
    labels = data['train']['labels'].flatten()

    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    # Create figure
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(12, 2 * num_classes))
    fig.suptitle('PathMNIST - Examples per Class (Training Set)', fontsize=14)

    for class_idx, cls in enumerate(unique_classes):
        # Get samples for this class
        class_indices = np.where(labels == cls)[0]
        selected = np.random.choice(class_indices, size=min(samples_per_class, len(class_indices)), replace=False)

        for sample_idx in range(samples_per_class):
            if class_idx == 0:
                ax = axes[sample_idx] if num_classes == 1 else axes[class_idx, sample_idx]
            else:
                ax = axes[class_idx, sample_idx]

            if sample_idx < len(selected):
                img = images[selected[sample_idx]]

                # Handle RGB or grayscale
                if img.ndim == 3 and img.shape[2] == 3:
                    ax.imshow(img)
                else:
                    ax.imshow(img.squeeze(), cmap='gray')

                if sample_idx == 0:
                    ax.set_ylabel(f'Class {cls}', fontsize=10)

            ax.axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path('results') / 'medmnist_class_examples.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved class visualization to {output_path}")
    plt.close()


def main():
    """Main exploration function."""
    print("[+] MedMNIST PathMNIST Data Explorer")
    print("=" * 60)

    # Set data directory
    data_dir = Path('data/medmnist/pathmnist')

    if not data_dir.exists():
        print(f"[!] Data directory not found: {data_dir}")
        print("    Please run download script first:")
        print("    python scripts/download_data.py --dataset medmnist --task pathmnist --output data")
        return

    # Load data
    print(f"\n[*] Loading data from {data_dir}")
    data = load_medmnist_data(data_dir)

    if not data:
        print("[!] No data loaded. Please check data directory.")
        return

    # Analyze dataset
    analyze_dataset(data)

    # Visualize samples
    visualize_samples(data, num_samples=16)

    # Visualize class examples
    visualize_class_examples(data, samples_per_class=5)

    print("\n[+] Exploration complete!")
    print("    Check the 'results/' directory for visualizations")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    main()
