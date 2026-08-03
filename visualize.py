# visualize.py
"""
Model interpretability and visualization using Grad-CAM.
Generates heatmaps showing what the model focuses on for predictions.
"""

import numpy as np
import tensorflow as tf
import keras
from keras import Model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image

import config
import preprocess


def label_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim > 1:
        return np.argmax(labels, axis=1)
    return labels.astype(int)


def get_gradcam_model(model: keras.Model) -> Tuple[Model, str]:
    """
    Create a model for Grad-CAM that outputs both predictions and last conv layer activations.
    
    Args:
        model: Trained classification model
    
    Returns:
        Tuple of (gradcam_model, last_conv_layer_name)
    """
    # Find the last convolutional layer in the backbone
    last_conv_layer = None
    last_conv_layer_name = None
    
    # Look through model layers
    for layer in reversed(model.layers):
        # Check if it's a model (backbone)
        if isinstance(layer, keras.Model):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                    last_conv_layer = sublayer
                    last_conv_layer_name = f"{layer.name}/{sublayer.name}"
                    break
            if last_conv_layer:
                break
        elif isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            last_conv_layer = layer
            last_conv_layer_name = layer.name
            break
    
    if last_conv_layer is None:
        raise ValueError("Could not find convolutional layer in model")
    
    print(f"Using layer for Grad-CAM: {last_conv_layer_name}")
    
    # For MobileNetV3, we need to get the output from the backbone
    # Find the backbone model
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            backbone = layer
            break
    
    if backbone is None:
        raise ValueError("Could not find backbone model")
    
    # Create a model that outputs both the last conv output and the final prediction
    # Get the output of the last conv layer from the backbone
    last_conv_output = backbone.get_layer(last_conv_layer.name).output
    
    # Build gradcam model
    gradcam_model = Model(
        inputs=model.input,
        outputs=[backbone.get_layer(last_conv_layer.name).output, model.output]
    )
    
    return gradcam_model, last_conv_layer.name


def compute_gradcam(
    model: keras.Model,
    image: np.ndarray,
    class_idx: Optional[int] = None
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for an image.
    
    Args:
        model: Trained model
        image: Single image (H, W, C) already preprocessed
        class_idx: Class index to visualize. If None, uses predicted class.
    
    Returns:
        Grad-CAM heatmap (H, W) normalized to [0, 1]
    """
    # Find the backbone and last conv layer
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            backbone = layer
            break
    
    if backbone is None:
        raise ValueError("Could not find backbone model")
    
    # Find last conv layer
    last_conv_layer = None
    for layer in reversed(backbone.layers):
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            last_conv_layer = layer
            break
    
    # Create gradient model
    grad_model = Model(
        inputs=model.input,
        outputs=[backbone.get_layer(last_conv_layer.name).output, model.output]
    )
    
    # Add batch dimension
    img_tensor = tf.expand_dims(image, axis=0)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[0, class_idx]
    
    # Get gradients of the loss with respect to conv outputs
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (H, W, C) in [0, 255] range
        heatmap: Grad-CAM heatmap (H', W') normalized to [0, 1]
        alpha: Blending factor for heatmap
        colormap: Matplotlib colormap name
    
    Returns:
        Superimposed image (H, W, C) in [0, 255] range
    """
    # Resize heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR
        )
    ) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Ensure image is in uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Blend
    superimposed = (1 - alpha) * image + alpha * heatmap_colored
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return superimposed


def visualize_gradcam(
    model: keras.Model,
    image: np.ndarray,
    true_label: Optional[int] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a visualization showing original image, Grad-CAM heatmap, and overlay.
    
    Args:
        model: Trained model
        image: Original image (H, W, C) in [0, 255] range
        true_label: True class label
        save_path: Path to save the figure
        show: Whether to display the figure
    
    Returns:
        Matplotlib figure
    """
    # Preprocess for model
    preprocessed = preprocess.preprocess_for_mobilenet(image)
    
    # Get prediction
    pred_probs = model.predict(np.expand_dims(preprocessed, axis=0), verbose=0)[0]
    pred_class = int(np.argmax(pred_probs))
    pred_label = config.CLASS_NAMES[pred_class]
    confidence = float(pred_probs[pred_class])
    
    # Compute Grad-CAM
    heatmap = compute_gradcam(model, preprocessed)
    
    # Create overlay
    overlay = overlay_heatmap(image, heatmap)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")
    
    # Heatmap
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(overlay)
    title = f"Prediction: {pred_label} ({confidence*100:.1f}%)"
    if true_label is not None:
        true_idx = int(np.argmax(true_label)) if np.asarray(true_label).ndim > 0 else int(true_label)
        true_name = config.CLASS_NAMES[true_idx]
        correct = "✓" if pred_class == true_idx else "✗"
        title += f"\nTrue: {true_name} {correct}"
    axes[2].set_title(title, fontsize=12)
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_batch(
    model: keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    num_samples: int = 8,
    save_dir: Optional[Path] = None,
    show: bool = False
) -> List[plt.Figure]:
    """
    Create Grad-CAM visualizations for a batch of images.
    
    Args:
        model: Trained model
        images: Array of images
        labels: Array of labels
        num_samples: Number of samples to visualize
        save_dir: Directory to save figures
        show: Whether to display figures
    
    Returns:
        List of matplotlib figures
    """
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Select samples - balance between classes and correct/incorrect predictions
    preprocessed = preprocess.preprocess_for_mobilenet(images)
    predictions = model.predict(preprocessed, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    label_ids = label_indices(labels)
    
    # Find misclassified samples
    incorrect_idx = np.where(pred_classes != label_ids)[0]
    correct_idx = np.where(pred_classes == label_ids)[0]
    
    # Prioritize showing some misclassified if available
    num_incorrect = min(len(incorrect_idx), num_samples // 2)
    num_correct = num_samples - num_incorrect
    
    selected_idx = []
    if num_incorrect > 0:
        selected_idx.extend(np.random.choice(incorrect_idx, num_incorrect, replace=False))
    if num_correct > 0:
        selected_idx.extend(np.random.choice(correct_idx, num_correct, replace=False))
    
    np.random.shuffle(selected_idx)
    
    figures = []
    for i, idx in enumerate(selected_idx):
        save_path = save_dir / f"gradcam_{i+1}.png" if save_dir else None
        fig = visualize_gradcam(
            model, images[idx], labels[idx],
            save_path=save_path, show=show
        )
        figures.append(fig)
        plt.close(fig)
    
    print(f"Generated {len(figures)} Grad-CAM visualizations")
    return figures


def create_class_activation_grid(
    model: keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    samples_per_class: int = 4,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a grid showing Grad-CAM for samples from each class.
    
    Args:
        model: Trained model
        images: Array of images
        labels: Array of labels
        samples_per_class: Number of samples per class
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(
        config.NUM_CLASSES, samples_per_class * 2,
        figsize=(samples_per_class * 6, max(4, config.NUM_CLASSES * 3))
    )
    axes = np.atleast_2d(axes)
    label_ids = label_indices(labels)
    
    for class_idx in range(config.NUM_CLASSES):
        class_mask = label_ids == class_idx
        class_images = images[class_mask]
        if len(class_images) == 0:
            continue
        
        # Select samples
        indices = np.random.choice(
            len(class_images), 
            min(samples_per_class, len(class_images)), 
            replace=False
        )
        
        for i, idx in enumerate(indices):
            img = class_images[idx]
            preprocessed = preprocess.preprocess_for_mobilenet(img)
            
            # Compute heatmap
            heatmap = compute_gradcam(model, preprocessed)
            overlay = overlay_heatmap(img, heatmap)
            
            # Get prediction
            pred_probs = model.predict(np.expand_dims(preprocessed, axis=0), verbose=0)[0]
            pred_class = int(np.argmax(pred_probs))
            pred_confidence = float(pred_probs[pred_class])
            
            # Plot original
            col = i * 2
            axes[class_idx, col].imshow(img.astype(np.uint8))
            axes[class_idx, col].axis("off")
            if i == 0:
                axes[class_idx, col].set_ylabel(
                    config.CLASS_NAMES[class_idx], 
                    fontsize=12, fontweight="bold"
                )
            
            # Plot overlay
            axes[class_idx, col + 1].imshow(overlay)
            axes[class_idx, col + 1].axis("off")
            
            # Add prediction indicator
            correct = "✓" if pred_class == class_idx else "✗"
            color = "green" if pred_class == class_idx else "red"
            axes[class_idx, col + 1].set_title(
                f"{correct} {pred_confidence*100:.0f}%",
                fontsize=10, color=color
            )
    
    plt.suptitle("Grad-CAM Visualization by Class\n(Original | Activation Overlay)", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved class activation grid to {save_path}")
    
    return fig


def analyze_model_attention(
    model_path: Optional[Path] = None,
    num_samples: int = 16,
    save_dir: Optional[Path] = None
) -> None:
    """
    Run full Grad-CAM analysis on test set.
    
    Args:
        model_path: Path to model. If None, uses default final model.
        num_samples: Number of individual samples to visualize
        save_dir: Directory to save outputs
    """
    if model_path is None:
        model_path = config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME
    
    if save_dir is None:
        save_dir = config.OUTPUTS_DIR / "gradcam"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GRAD-CAM MODEL ATTENTION ANALYSIS")
    print("="*60)
    
    # Load model and data
    print(f"\nLoading model from {model_path}")
    model = keras.models.load_model(model_path)
    
    print("Loading test data...")
    test_images, test_labels = preprocess.load_test_data()
    print(f"Test samples: {len(test_images)}")
    
    # Create class activation grid
    print("\nGenerating class activation grid...")
    grid_fig = create_class_activation_grid(
        model, test_images, test_labels,
        samples_per_class=4,
        save_path=save_dir / "class_activation_grid.png"
    )
    plt.close(grid_fig)
    
    # Generate individual visualizations
    print(f"\nGenerating {num_samples} individual Grad-CAM visualizations...")
    visualize_batch(
        model, test_images, test_labels,
        num_samples=num_samples,
        save_dir=save_dir / "individual",
        show=False
    )
    
    print(f"\nAll Grad-CAM outputs saved to {save_dir}")
    print("="*60)


if __name__ == "__main__":
    analyze_model_attention(num_samples=16)

