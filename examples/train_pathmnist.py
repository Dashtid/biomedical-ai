#!/usr/bin/env python
"""Train a classification model on Path MNIST dataset.

PathMNIST is a colon pathology dataset with 9 tissue types:
0: adipose, 1: background, 2: debris, 3: lymphocytes,
4: mucus, 5: smooth muscle, 6: normal colon mucosa,
7: cancer-associated stroma, 8: colorectal adenocarcinoma epithelium

Usage:
    python examples/train_pathmnist.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_seg.training.metrics import SegmentationMetrics


def load_pathmnist(data_path: str = "data/medmnist/pathmnist/pathmnist.npz"):
    """Load PathMNIST dataset.

    Args:
        data_path: Path to PathMNIST npz file

    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    print(f"[*] Loading PathMNIST from {data_path}")
    data = np.load(data_path)

    train_images = data['train_images']  # (89996, 28, 28, 3)
    train_labels = data['train_labels']  # (89996, 1)
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # Normalize images to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    print(f"    Train: {train_images.shape[0]} samples")
    print(f"    Val: {val_images.shape[0]} samples")
    print(f"    Test: {test_images.shape[0]} samples")
    print(f"    Image shape: {train_images[0].shape}")
    print(f"    Number of classes: {len(np.unique(train_labels))}")

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def build_cnn_classifier(input_shape=(28, 28, 3), num_classes=9):
    """Build a CNN classifier for PathMNIST.

    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of classes

    Returns:
        Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # Convolutional blocks
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='PathMNIST_Classifier')

    return model


def plot_training_history(history, save_path='results/training_history.png'):
    """Plot training history.

    Args:
        history: Keras training history object
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    # Save plot
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[*] Saved training history plot to {save_path}")
    plt.close()


def evaluate_model(model, test_images, test_labels, class_names=None):
    """Evaluate model on test set.

    Args:
        model: Trained Keras model
        test_images: Test images
        test_labels: Test labels
        class_names: Optional list of class names
    """
    print("\n" + "=" * 60)
    print("[+] Evaluating on Test Set")
    print("=" * 60)

    # Get predictions
    predictions = model.predict(test_images, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = test_labels.flatten()

    # Calculate accuracy
    accuracy = np.mean(pred_classes == true_classes)
    print(f"    Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class accuracy
    print("\n[*] Per-class Accuracy:")
    for cls in range(len(np.unique(true_classes))):
        mask = true_classes == cls
        if np.sum(mask) > 0:
            cls_acc = np.mean(pred_classes[mask] == cls)
            cls_name = class_names[cls] if class_names else f"Class {cls}"
            print(f"    {cls_name}: {cls_acc:.4f} ({cls_acc*100:.2f}%)")


def main():
    """Main training function."""
    print("[+] PathMNIST Classification Training")
    print("=" * 60)

    # Parameters
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = 9

    # Class names for PathMNIST
    CLASS_NAMES = [
        "Adipose",
        "Background",
        "Debris",
        "Lymphocytes",
        "Mucus",
        "Smooth muscle",
        "Normal colon mucosa",
        "Cancer-associated stroma",
        "Colorectal adenocarcinoma epithelium"
    ]

    # Load data
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_pathmnist()

    # Convert labels to categorical
    train_labels_cat = keras.utils.to_categorical(train_labels, NUM_CLASSES)
    val_labels_cat = keras.utils.to_categorical(val_labels, NUM_CLASSES)
    test_labels_cat = keras.utils.to_categorical(test_labels, NUM_CLASSES)

    # Build model
    print(f"\n[*] Building CNN classifier...")
    model = build_cnn_classifier(input_shape=(28, 28, 3), num_classes=NUM_CLASSES)

    print(f"    Total parameters: {model.count_params():,}")

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/pathmnist_best.keras',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    # Create checkpoint directory
    Path('checkpoints').mkdir(exist_ok=True)

    # Train model
    print(f"\n[*] Starting training for {EPOCHS} epochs...")
    print("=" * 60)

    history = model.fit(
        train_images, train_labels_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_images, val_labels_cat),
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    evaluate_model(model, test_images, test_labels, CLASS_NAMES)

    # Print final results
    print("\n" + "=" * 60)
    print("[+] Training completed!")
    print(f"    Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"    Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"    Final val loss: {history.history['val_loss'][-1]:.4f}")
    print(f"    Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"\n[+] Best model saved to: checkpoints/pathmnist_best.keras")
    print("[+] Done!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    main()
