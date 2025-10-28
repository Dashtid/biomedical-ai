#!/usr/bin/env python
"""Simple training example for medical image segmentation.

This script demonstrates how to train a U-Net model on synthetic data.
For real data, replace the synthetic data generation with actual data loading.

Usage:
    python examples/train_simple.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from med_seg.models import UNet
from med_seg.data import MedicalImagePreprocessor
from med_seg.training import ModelTrainer
from med_seg.training.losses import dice_loss, combined_loss
from med_seg.training.metrics import SegmentationMetrics


def generate_synthetic_data(
    num_samples: int = 100,
    image_size: int = 128
) -> tuple:
    """Generate synthetic segmentation data for testing.

    Args:
        num_samples: Number of samples to generate
        image_size: Size of images (square)

    Returns:
        Tuple of (images, masks)
    """
    images = []
    masks = []

    for _ in range(num_samples):
        # Create random image
        image = np.random.rand(image_size, image_size).astype(np.float32)

        # Create circular mask
        center_x = np.random.randint(image_size // 4, 3 * image_size // 4)
        center_y = np.random.randint(image_size // 4, 3 * image_size // 4)
        radius = np.random.randint(image_size // 8, image_size // 4)

        y, x = np.ogrid[:image_size, :image_size]
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2).astype(np.float32)

        # Add mask region to image for some signal
        image = image + mask * 0.5
        image = np.clip(image, 0, 1)

        images.append(image)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    # Add channel dimension
    images = np.expand_dims(images, axis=-1)
    masks = np.expand_dims(masks, axis=-1)

    return images, masks


def create_data_generator(images, masks, batch_size=4, shuffle=True):
    """Create a data generator for training.

    Args:
        images: Array of images
        masks: Array of masks
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Yields:
        Batches of (images, masks)
    """
    num_samples = len(images)
    indices = np.arange(num_samples)

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_images = images[batch_indices]
            batch_masks = masks[batch_indices]

            yield batch_images, batch_masks


def main():
    """Main training function."""
    print("[+] Medical Image Segmentation - Simple Training Example")
    print("=" * 60)

    # Parameters
    IMAGE_SIZE = 128
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 0.001

    print(f"\n[*] Generating synthetic data...")
    train_images, train_masks = generate_synthetic_data(num_samples=80, image_size=IMAGE_SIZE)
    val_images, val_masks = generate_synthetic_data(num_samples=20, image_size=IMAGE_SIZE)

    print(f"    Train samples: {len(train_images)}")
    print(f"    Val samples: {len(val_images)}")
    print(f"    Image shape: {train_images[0].shape}")

    # Create data generators
    print(f"\n[*] Creating data generators...")
    train_gen = create_data_generator(train_images, train_masks, batch_size=BATCH_SIZE)
    val_gen = create_data_generator(val_images, val_masks, batch_size=BATCH_SIZE, shuffle=False)

    steps_per_epoch = len(train_images) // BATCH_SIZE
    validation_steps = len(val_images) // BATCH_SIZE

    # Build model
    print(f"\n[*] Building U-Net model...")
    model_builder = UNet(
        input_size=IMAGE_SIZE,
        input_channels=1,
        num_classes=1,
        base_filters=32,  # Small for quick training
        depth=3,
        use_batch_norm=True,
        use_dropout=True,
        dropout_rate=0.3
    )
    model = model_builder.build()

    print(f"    Total parameters: {model.count_params():,}")
    print(f"    Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

    # Setup metrics
    metrics = SegmentationMetrics(threshold=0.5)

    # Create trainer
    print(f"\n[*] Setting up trainer...")
    trainer = ModelTrainer(
        model=model,
        loss_function=dice_loss,  # or combined_loss()
        learning_rate=LEARNING_RATE,
        metrics=[metrics.dice, metrics.iou, metrics.precision_metric]
    )

    # Compile model
    trainer.compile()

    # Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
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
            filepath='best_model.keras',
            monitor='val_dice',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    print(f"\n[*] Starting training for {EPOCHS} epochs...")
    print("=" * 60)

    history = trainer.train(
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Print final results
    print("\n" + "=" * 60)
    print("[+] Training completed!")
    print(f"    Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"    Final val loss: {history.history['val_loss'][-1]:.4f}")
    print(f"    Final val DICE: {history.history['val_dice'][-1]:.4f}")
    print(f"    Final val IoU: {history.history['val_iou'][-1]:.4f}")

    # Test prediction
    print(f"\n[*] Testing prediction...")
    test_image = val_images[0:1]
    prediction = trainer.predict(test_image)

    print(f"    Input shape: {test_image.shape}")
    print(f"    Output shape: {prediction.shape}")
    print(f"    Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")

    print(f"\n[+] Model saved to: best_model.keras")
    print("[+] Done!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    main()
