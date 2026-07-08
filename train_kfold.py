# train_kfold.py
"""
K-Fold Cross-Validation training pipeline for multi-class potato leaf disease.
Uses MobileNetV3 with transfer learning, label smoothing, and fine-tuning.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import Model, layers
from keras.applications import MobileNetV3Large, MobileNetV3Small
from sklearn.model_selection import StratifiedKFold

import config
import preprocess


def create_model(
    input_shape: Tuple[int, int, int] = config.IMG_SHAPE,
    dropout_rate: float = config.DROPOUT_RATE,
    freeze_backbone: bool = True
) -> keras.Model:
    """
    Create MobileNetV3 model for multi-class classification.
    
    Args:
        input_shape: Input image shape (H, W, C)
        dropout_rate: Dropout rate before final layer
        freeze_backbone: Whether to freeze backbone weights
    
    Returns:
        Compiled Keras model
    """
    # Load pretrained MobileNetV3 backbone based on config
    backbone_class = MobileNetV3Large if config.BACKBONE == "MobileNetV3Large" else MobileNetV3Small
    backbone = backbone_class(
        input_shape=input_shape,
        include_top=False,
        weights=config.PRETRAINED_WEIGHTS,
        include_preprocessing=False,  # We handle preprocessing in pipeline
    )
    
    # Freeze/unfreeze backbone
    backbone.trainable = not freeze_backbone
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    x = backbone(inputs, training=False if freeze_backbone else None)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    
    model = Model(inputs, outputs, name="potato_leaf_classifier")
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float,
    label_smoothing: float = config.LABEL_SMOOTHING
) -> keras.Model:
    """
    Compile model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        label_smoothing: Label smoothing factor
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=min(2, config.NUM_CLASSES), name="top2_accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )
    return model


def get_cosine_schedule(
    initial_lr: float,
    total_epochs: int,
    warmup_epochs: int = 3
) -> keras.optimizers.schedules.LearningRateSchedule:
    """
    Create cosine annealing learning rate schedule with warmup.
    
    Args:
        initial_lr: Peak learning rate after warmup
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
    
    Returns:
        Learning rate schedule
    """
    class CosineScheduleWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_lr, total_steps, warmup_steps):
            super().__init__()
            self.initial_lr = initial_lr
            self.total_steps = total_steps
            self.warmup_steps = warmup_steps
        
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            
            # Warmup phase
            warmup_lr = self.initial_lr * (step / self.warmup_steps)
            
            # Cosine decay phase
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_lr = self.initial_lr * 0.5 * (1 + tf.cos(np.pi * progress))
            
            return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
        
        def get_config(self):
            return {
                "initial_lr": self.initial_lr,
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps,
            }
    
    return CosineScheduleWithWarmup(initial_lr, total_epochs, warmup_epochs)


def get_callbacks(
    fold: int,
    stage: int,
    monitor: str = "val_accuracy"
) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks for checkpointing and early stopping.
    
    Args:
        fold: Current fold number
        stage: Training stage (1=frozen, 2=fine-tune)
        monitor: Metric to monitor for best model
    
    Returns:
        List of Keras callbacks
    """
    checkpoint_path = config.MODELS_DIR / f"fold{fold}_stage{stage}_best.keras"
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            min_delta=config.MIN_DELTA,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    
    return callbacks


def label_indices(labels: np.ndarray) -> np.ndarray:
    """Convert one-hot or integer labels to class indices."""
    labels = np.asarray(labels)
    if labels.ndim > 1:
        return np.argmax(labels, axis=1)
    return labels.astype(int)


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights to handle imbalance.
    
    Args:
        labels: Array of labels
    
    Returns:
        Dictionary mapping class index to weight
    """
    indices = label_indices(labels)
    unique, counts = np.unique(indices, return_counts=True)
    total = len(indices)
    weights = {int(cls): total / (len(unique) * count) for cls, count in zip(unique, counts)}
    return weights


def unfreeze_model(model: keras.Model, unfreeze_percent: float) -> keras.Model:
    """
    Unfreeze the last N% of backbone layers for fine-tuning.
    
    Args:
        model: Model with frozen backbone
        unfreeze_percent: Fraction of layers to unfreeze (from the end)
    
    Returns:
        Model with partially unfrozen backbone
    """
    # Find the backbone (MobileNetV3Small)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            backbone = layer
            break
    
    if backbone is None:
        print("Warning: Could not find backbone to unfreeze")
        return model
    
    # Unfreeze backbone
    backbone.trainable = True
    
    # Freeze early layers
    num_layers = len(backbone.layers)
    freeze_until = int(num_layers * (1 - unfreeze_percent))
    
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    
    trainable_count = sum(1 for layer in backbone.layers if layer.trainable)
    print(f"Unfroze {trainable_count}/{num_layers} backbone layers")
    
    return model


def train_fold(
    fold: int,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
) -> Tuple[keras.Model, Dict]:
    """
    Train a single fold with two-stage training.
    
    Args:
        fold: Fold number
        train_images: Training images
        train_labels: Training labels
        val_images: Validation images
        val_labels: Validation labels
    
    Returns:
        Trained model and training history
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{config.N_FOLDS}")
    print(f"{'='*60}")
    print(f"Train: {len(train_images)} samples")
    print(f"Val: {len(val_images)} samples")
    
    # Compute class weights
    class_weights = compute_class_weights(train_labels)
    print(f"Class weights: {class_weights}")
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    
    # Apply pipelines
    train_ds = train_ds.shuffle(len(train_images), seed=config.SEED)
    train_pipeline = preprocess.create_training_pipeline(train_ds, augment=True, use_mixup=True)
    val_pipeline = preprocess.create_validation_pipeline(val_ds)
    
    # =========================================================================
    # Stage 1: Train with frozen backbone
    # =========================================================================
    print(f"\n--- Stage 1: Frozen backbone training ---")
    
    model = create_model(freeze_backbone=True)
    model = compile_model(model, learning_rate=config.STAGE1_LR)
    
    history1 = model.fit(
        train_pipeline,
        validation_data=val_pipeline,
        epochs=config.STAGE1_EPOCHS,
        callbacks=get_callbacks(fold, stage=1),
        class_weight=class_weights,
        verbose=1,
    )
    
    # =========================================================================
    # Stage 2: Fine-tune with unfrozen backbone
    # =========================================================================
    print(f"\n--- Stage 2: Fine-tuning ---")
    
    model = unfreeze_model(model, config.FINETUNE_LAYERS_PERCENT)
    model = compile_model(model, learning_rate=config.STAGE2_LR)
    
    history2 = model.fit(
        train_pipeline,
        validation_data=val_pipeline,
        epochs=config.STAGE2_EPOCHS,
        callbacks=get_callbacks(fold, stage=2),
        class_weight=class_weights,
        verbose=1,
    )
    
    # Combine histories
    combined_history = {
        "stage1": history1.history,
        "stage2": history2.history,
    }
    
    # Save final fold model
    final_path = config.MODELS_DIR / f"fold{fold}_final.keras"
    model.save(final_path)
    print(f"Saved fold {fold + 1} model to {final_path}")
    
    return model, combined_history


def train_kfold() -> Tuple[List[keras.Model], Dict]:
    """
    Run full K-Fold cross-validation training.
    
    Returns:
        List of trained models and aggregated metrics
    """
    print("Loading data for K-Fold training...")
    images, labels = preprocess.load_for_kfold()
    label_ids = label_indices(labels)
    
    print(f"Total samples: {len(images)}")
    print(f"Class distribution: {dict(zip(*np.unique(label_ids, return_counts=True)))}")
    
    # Initialize K-Fold
    kfold = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    
    models = []
    all_histories = []
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(images, label_ids)):
        # Split data
        train_images, train_labels = images[train_idx], labels[train_idx]
        val_images, val_labels = images[val_idx], labels[val_idx]
        
        # Train fold
        model, history = train_fold(
            fold, train_images, train_labels, val_images, val_labels
        )
        
        models.append(model)
        all_histories.append(history)
        
        # Get final metrics from stage 2
        final_metrics = {
            key: float(values[-1])
            for key, values in history["stage2"].items()
            if key.startswith("val_")
        }
        fold_metrics.append(final_metrics)
        
        print(f"\nFold {fold + 1} Final Metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # Aggregate metrics across folds
    print("\n" + "="*60)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("="*60)
    
    aggregated = {}
    for key in fold_metrics[0].keys():
        values = [fm[key] for fm in fold_metrics]
        aggregated[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values,
        }
        print(f"{key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_folds": config.N_FOLDS,
        "total_samples": len(images),
        "fold_metrics": fold_metrics,
        "aggregated": {
            k: {"mean": float(v["mean"]), "std": float(v["std"])}
            for k, v in aggregated.items()
        },
    }
    
    results_path = config.OUTPUTS_DIR / "kfold_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return models, results


def train_final_model() -> keras.Model:
    """
    Train a final model on all training data (train + val) for deployment.
    Uses the best hyperparameters found during k-fold CV.
    
    Returns:
        Final trained model
    """
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL FOR DEPLOYMENT")
    print("="*60)
    
    # Load all training data
    images, labels = preprocess.load_for_kfold()
    label_ids = label_indices(labels)
    
    print(f"Training on all {len(images)} samples")
    
    # Use 10% for validation to have some feedback during training.
    rng = np.random.default_rng(config.SEED)
    indices = rng.permutation(len(images))
    val_size = min(max(1, int(len(images) * 0.1)), max(1, len(images) - 1))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    train_images, train_labels = images[train_idx], labels[train_idx]
    val_images, val_labels = images[val_idx], labels[val_idx]

    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset = dataset.shuffle(len(train_images), seed=config.SEED)
    train_pipeline = preprocess.create_training_pipeline(dataset, augment=True, use_mixup=True)
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_pipeline = preprocess.create_validation_pipeline(val_ds)
    
    class_weights = compute_class_weights(label_ids)
    
    # Stage 1
    print("\n--- Stage 1: Frozen backbone ---")
    model = create_model(freeze_backbone=True)
    model = compile_model(model, learning_rate=config.STAGE1_LR)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODELS_DIR / "final_stage1.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        ),
    ]
    
    model.fit(
        train_pipeline,
        validation_data=val_pipeline,
        epochs=config.STAGE1_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    
    # Stage 2
    print("\n--- Stage 2: Fine-tuning ---")
    model = unfreeze_model(model, config.FINETUNE_LAYERS_PERCENT)
    model = compile_model(model, learning_rate=config.STAGE2_LR)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODELS_DIR / "final_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        ),
    ]
    
    model.fit(
        train_pipeline,
        validation_data=val_pipeline,
        epochs=config.STAGE2_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    
    # Save final model
    final_path = config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")
    
    return model


if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)
    
    # Run K-Fold training
    models, results = train_kfold()
    
    # Train final deployment model
    final_model = train_final_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"K-Fold models saved in: {config.MODELS_DIR}")
    print(f"Final model: {config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME}")
    print(f"Results: {config.OUTPUTS_DIR / 'kfold_results.json'}")