# convert_tflite.py
"""
Convert trained Keras model to TensorFlow Lite format with INT8 quantization
for efficient mobile deployment.
"""

import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from typing import Callable, Optional, Tuple
import json
import argparse

import config
import preprocess


def assert_field_validation_ready(force: bool = False) -> None:
    """Prevent mobile export while the field pipeline is still failing."""
    if force:
        print("WARNING: Forcing export despite field validation gate.")
        return

    results_path = config.OUTPUTS_DIR / "field_validation_results.json"
    if not results_path.exists():
        raise RuntimeError(
            f"Refusing TFLite export because {results_path} does not exist. "
            "Run field validation first, or pass --force for an experimental export."
        )

    with open(results_path, "r", encoding="utf-8") as handle:
        results = json.load(handle)
    test_results = results.get("test", {})
    macro_f1 = float(test_results.get("macro_f1", 0.0))
    uncertain_rate = float(test_results.get("diagnostics", {}).get("uncertain_rate", 1.0))
    if macro_f1 < config.MIN_FIELD_MACRO_F1_FOR_EXPORT or uncertain_rate > config.MAX_FIELD_UNCERTAIN_RATE_FOR_EXPORT:
        raise RuntimeError(
            "Refusing TFLite export because field validation is below the deployment gate: "
            f"macro_f1={macro_f1:.3f} "
            f"(required >= {config.MIN_FIELD_MACRO_F1_FOR_EXPORT:.3f}), "
            f"uncertain_rate={uncertain_rate:.3f} "
            f"(required <= {config.MAX_FIELD_UNCERTAIN_RATE_FOR_EXPORT:.3f}). "
            "Pass --force only for an experimental export."
        )


def representative_dataset_generator(
    images: np.ndarray,
    num_samples: int = 100,
    preprocessing: str = "mobilenet",
) -> Callable:
    """
    Create a representative dataset generator for quantization calibration.
    
    Args:
        images: Array of images for calibration
        num_samples: Number of samples to use for calibration
    
    Returns:
        Generator function for TFLite converter
    """
    if preprocessing == "mobilenet":
        preprocessed = preprocess.preprocess_for_mobilenet(images)
    elif preprocessing == "scale01":
        preprocessed = images.astype(np.float32) / 255.0
    else:
        preprocessed = images.astype(np.float32)
    
    # Use subset for calibration
    calibration_images = preprocessed[:min(num_samples, len(preprocessed))]
    
    def generator():
        for img in calibration_images:
            # Add batch dimension
            yield [np.expand_dims(img, axis=0).astype(np.float32)]
    
    return generator


def convert_to_tflite(
    model_path: Path,
    output_path: Path,
    quantize: bool = True,
    quantize_int8: bool = config.QUANTIZE_INT8,
    calibration_images: Optional[np.ndarray] = None,
    representative_preprocessing: str = "mobilenet",
) -> Tuple[bytes, dict]:
    """
    Convert Keras model to TFLite format.
    
    Args:
        model_path: Path to Keras model
        output_path: Path for output TFLite model
        quantize: Whether to apply any quantization
        quantize_int8: Whether to use full INT8 quantization (requires calibration data)
        calibration_images: Images for INT8 quantization calibration
    
    Returns:
        TFLite model bytes and conversion info
    """
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    conversion_info = {
        "source_model": str(model_path),
        "output_model": str(output_path),
        "quantization": "none",
    }
    
    if quantize:
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        conversion_info["quantization"] = "dynamic_range"
        
        if quantize_int8 and calibration_images is not None:
            print("Applying INT8 quantization with calibration data...")
            
            # Set representative dataset for full integer quantization
            converter.representative_dataset = representative_dataset_generator(
                calibration_images,
                preprocessing=representative_preprocessing,
            )
            
            # Full integer quantization (all ops in int8)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            
            # Set input/output types to float for ease of use
            # (they'll be quantized internally but accept float input)
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            
            conversion_info["quantization"] = "full_int8"
            conversion_info["calibration_samples"] = len(calibration_images)
    
    # Convert
    print("Converting model...")
    tflite_model = converter.convert()
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    # Calculate size reduction
    original_size = model_path.stat().st_size
    tflite_size = output_path.stat().st_size
    
    conversion_info["original_size_mb"] = round(original_size / (1024 * 1024), 2)
    conversion_info["tflite_size_mb"] = round(tflite_size / (1024 * 1024), 2)
    conversion_info["size_reduction"] = round((1 - tflite_size / original_size) * 100, 1)
    
    print(f"\nConversion complete!")
    print(f"Original size: {conversion_info['original_size_mb']:.2f} MB")
    print(f"TFLite size:   {conversion_info['tflite_size_mb']:.2f} MB")
    print(f"Size reduction: {conversion_info['size_reduction']:.1f}%")
    print(f"Saved to: {output_path}")
    
    return tflite_model, conversion_info


def validate_tflite_model(
    tflite_path: Path,
    keras_model_path: Path,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    tolerance: float = 0.05
) -> dict:
    """
    Validate TFLite model against original Keras model.
    
    Args:
        tflite_path: Path to TFLite model
        keras_model_path: Path to original Keras model
        test_images: Test images
        test_labels: Test labels
        tolerance: Maximum allowed difference in accuracy
    
    Returns:
        Validation results dictionary
    """
    print("\nValidating TFLite model...")
    
    # Load Keras model
    keras_model = keras.models.load_model(keras_model_path, compile=False)
    
    # Preprocess images
    preprocessed = preprocess.preprocess_for_mobilenet(test_images)
    
    # Keras predictions
    print("Running Keras model predictions...")
    keras_probs = keras_model.predict(preprocessed, verbose=0)
    keras_preds = np.argmax(keras_probs, axis=1)
    true_labels = np.argmax(test_labels, axis=1) if test_labels.ndim > 1 else test_labels.astype(int)
    keras_accuracy = np.mean(keras_preds == true_labels)
    
    # TFLite predictions
    print("Running TFLite model predictions...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    tflite_probs = []
    for img in preprocessed:
        input_data = np.expand_dims(img, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        tflite_probs.append(output[0])
    
    tflite_probs = np.array(tflite_probs)
    tflite_preds = np.argmax(tflite_probs, axis=1)
    tflite_accuracy = np.mean(tflite_preds == true_labels)
    
    # Compare predictions
    prediction_agreement = np.mean(keras_preds == tflite_preds)
    prob_difference = np.abs(keras_probs - tflite_probs)
    
    validation_results = {
        "keras_accuracy": float(keras_accuracy),
        "tflite_accuracy": float(tflite_accuracy),
        "accuracy_difference": float(abs(keras_accuracy - tflite_accuracy)),
        "prediction_agreement": float(prediction_agreement),
        "mean_prob_difference": float(np.mean(prob_difference)),
        "max_prob_difference": float(np.max(prob_difference)),
        "passed": abs(keras_accuracy - tflite_accuracy) <= tolerance,
    }
    
    print(f"\nValidation Results:")
    print(f"Keras Accuracy:       {keras_accuracy*100:.2f}%")
    print(f"TFLite Accuracy:      {tflite_accuracy*100:.2f}%")
    print(f"Accuracy Difference:  {abs(keras_accuracy - tflite_accuracy)*100:.2f}%")
    print(f"Prediction Agreement: {prediction_agreement*100:.2f}%")
    print(f"Mean Prob Difference: {np.mean(prob_difference):.4f}")
    print(f"Max Prob Difference:  {np.max(prob_difference):.4f}")
    print(f"Validation {'PASSED' if validation_results['passed'] else 'FAILED'}")
    
    return validation_results


def benchmark_tflite_model(
    tflite_path: Path,
    test_images: np.ndarray,
    num_runs: int = 100
) -> dict:
    """
    Benchmark TFLite model inference speed.
    
    Args:
        tflite_path: Path to TFLite model
        test_images: Test images for benchmarking
        num_runs: Number of inference runs for timing
    
    Returns:
        Benchmark results dictionary
    """
    import time
    
    print("\nBenchmarking TFLite model...")
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess one image
    img = preprocess.preprocess_for_mobilenet(test_images[0:1])
    input_data = img.astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    benchmark_results = {
        "num_runs": num_runs,
        "mean_latency_ms": float(np.mean(times)),
        "std_latency_ms": float(np.std(times)),
        "min_latency_ms": float(np.min(times)),
        "max_latency_ms": float(np.max(times)),
        "median_latency_ms": float(np.median(times)),
    }
    
    print(f"\nBenchmark Results ({num_runs} runs):")
    print(f"Mean Latency:   {benchmark_results['mean_latency_ms']:.2f} ms")
    print(f"Std Latency:    {benchmark_results['std_latency_ms']:.2f} ms")
    print(f"Min Latency:    {benchmark_results['min_latency_ms']:.2f} ms")
    print(f"Max Latency:    {benchmark_results['max_latency_ms']:.2f} ms")
    print(f"Median Latency: {benchmark_results['median_latency_ms']:.2f} ms")
    
    return benchmark_results


def run_conversion_pipeline(
    model_path: Optional[Path] = None,
    output_name: str = config.TFLITE_MODEL_NAME,
    force: bool = False,
) -> dict:
    """
    Run full conversion and validation pipeline.
    
    Args:
        model_path: Path to Keras model. If None, uses default final model.
        output_name: Output TFLite model filename
    
    Returns:
        Complete conversion results
    """
    if model_path is None:
        model_path = config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME
    assert_field_validation_ready(force=force)
    
    output_path = config.MODELS_DIR / output_name
    
    print("="*60)
    print("TFLITE MODEL CONVERSION PIPELINE")
    print("="*60)
    
    # Load calibration data (use training data for calibration)
    print("\nLoading calibration data...")
    images, _ = preprocess.load_for_kfold()
    
    # Convert model
    print("\n--- Converting to TFLite ---")
    tflite_model, conversion_info = convert_to_tflite(
        model_path=model_path,
        output_path=output_path,
        quantize=True,
        quantize_int8=config.QUANTIZE_INT8,
        calibration_images=images,
    )
    
    # Load test data for validation
    print("\nLoading test data for validation...")
    test_images, test_labels = preprocess.load_test_data()
    
    # Validate model
    print("\n--- Validating TFLite Model ---")
    validation_results = validate_tflite_model(
        tflite_path=output_path,
        keras_model_path=model_path,
        test_images=test_images,
        test_labels=test_labels,
    )
    
    # Benchmark model
    print("\n--- Benchmarking TFLite Model ---")
    benchmark_results = benchmark_tflite_model(
        tflite_path=output_path,
        test_images=test_images,
    )
    
    # Combine all results
    results = {
        "conversion": conversion_info,
        "validation": validation_results,
        "benchmark": benchmark_results,
    }
    
    # Save results
    results_path = config.OUTPUTS_DIR / "tflite_conversion_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"TFLite model: {output_path}")
    print(f"Results: {results_path}")
    
    return results


def load_segmentation_calibration_images(num_samples: int = 100) -> Optional[np.ndarray]:
    """Load raw segmentation images for segmenter INT8 calibration."""
    from segment.data import load_image

    image_dir = config.SEGMENTATION_IMAGE_DIR
    if not image_dir.exists():
        return None

    images = []
    for image_path in sorted(image_dir.glob("*"))[:num_samples]:
        try:
            images.append((load_image(str(image_path)).numpy() * 255.0).astype(np.float32))
        except Exception:
            continue
    if not images:
        return None
    return np.asarray(images)


def run_dual_conversion_pipeline(
    classifier_model_path: Optional[Path] = None,
    segmenter_model_path: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Export both classifier and segmenter models for mobile deployment."""
    assert_field_validation_ready(force=force)
    if classifier_model_path is None:
        classifier_model_path = config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME
    if segmenter_model_path is None:
        segmenter_model_path = config.MODELS_DIR / config.SEGMENTER_MODEL_NAME

    results = {}
    classifier_images, _ = preprocess.load_for_kfold()
    _, classifier_info = convert_to_tflite(
        model_path=classifier_model_path,
        output_path=config.MODELS_DIR / config.CLASSIFIER_TFLITE_MODEL_NAME,
        quantize=True,
        quantize_int8=config.QUANTIZE_INT8,
        calibration_images=classifier_images,
        representative_preprocessing="mobilenet",
    )
    results["classifier"] = classifier_info

    if segmenter_model_path.exists():
        segmenter_calibration = load_segmentation_calibration_images()
        _, segmenter_info = convert_to_tflite(
            model_path=segmenter_model_path,
            output_path=config.MODELS_DIR / config.SEGMENTER_TFLITE_MODEL_NAME,
            quantize=True,
            quantize_int8=config.QUANTIZE_INT8 and segmenter_calibration is not None,
            calibration_images=segmenter_calibration,
            representative_preprocessing="scale01",
        )
        results["segmenter"] = segmenter_info
    else:
        results["segmenter"] = {"skipped": True, "reason": f"{segmenter_model_path} not found"}

    results_path = config.OUTPUTS_DIR / "dual_tflite_conversion_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Dual conversion results saved to {results_path}")
    return results


def convert_for_android(
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Convert and package model for Android deployment.
    Creates the TFLite model with metadata for easy integration.
    
    Args:
        model_path: Path to Keras model
        output_dir: Output directory for Android assets
    
    Returns:
        Path to output TFLite model
    """
    if model_path is None:
        model_path = config.MODELS_DIR / config.CLASSIFIER_MODEL_NAME
    assert_field_validation_ready(force=force)
    
    if output_dir is None:
        output_dir = config.MODELS_DIR / "android"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_output_path = output_dir / config.CLASSIFIER_TFLITE_MODEL_NAME
    segmenter_output_path = output_dir / config.SEGMENTER_TFLITE_MODEL_NAME
    
    print("Converting model for Android deployment...")
    
    # Load calibration data
    images, _ = preprocess.load_for_kfold()
    
    # Convert with INT8 quantization
    convert_to_tflite(
        model_path=model_path,
        output_path=classifier_output_path,
        quantize=True,
        quantize_int8=True,
        calibration_images=images,
        representative_preprocessing="mobilenet",
    )

    segmenter_path = config.MODELS_DIR / config.SEGMENTER_MODEL_NAME
    segmenter_exported = segmenter_path.exists()
    if segmenter_exported:
        segmenter_calibration = load_segmentation_calibration_images()
        convert_to_tflite(
            model_path=segmenter_path,
            output_path=segmenter_output_path,
            quantize=True,
            quantize_int8=segmenter_calibration is not None,
            calibration_images=segmenter_calibration,
            representative_preprocessing="scale01",
        )
    
    # Create labels file
    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w") as f:
        for class_name in config.CLASSES:
            f.write(f"{class_name}\n")
    
    # Create model info file
    info = {
        "model_name": "Autorogue Potato Disease Detector",
        "version": "2.0",
        "pipeline": "segmenter -> per-leaf classifier -> confidence-gated aggregation",
        "classifier": {
            "file": classifier_output_path.name,
            "input_shape": [1, config.IMG_SIZE, config.IMG_SIZE, 3],
            "input_type": "float32",
            "output_shape": [1, config.NUM_CLASSES],
            "output_type": "float32",
            "labels": config.CLASSES,
            "preprocessing": "scale RGB to [-1, 1]",
        },
        "segmenter": {
            "available": segmenter_exported,
            "file": segmenter_output_path.name,
            "input_shape": [1, config.SEGMENTATION_IMG_SIZE, config.SEGMENTATION_IMG_SIZE, 3],
            "input_type": "float32",
            "output_shape": [1, config.SEGMENTATION_IMG_SIZE, config.SEGMENTATION_IMG_SIZE, 1],
            "output_type": "float32",
            "preprocessing": "scale RGB to [0, 1]",
            "threshold": config.SEGMENTATION_THRESHOLD,
        },
        "aggregation": {
            "per_leaf_confidence_threshold": config.PER_LEAF_CONFIDENCE_THRESHOLD,
            "plant_class_fraction_threshold": config.PLANT_CLASS_FRACTION_THRESHOLD,
        },
    }
    
    info_path = output_dir / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nAndroid assets created in {output_dir}:")
    print(f"  - {classifier_output_path.name}")
    if segmenter_exported:
        print(f"  - {segmenter_output_path.name}")
    print(f"  - {labels_path.name}")
    print(f"  - {info_path.name}")
    
    return classifier_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Autorogue models to TFLite.")
    parser.add_argument("--force", action="store_true", help="Bypass the field-validation export gate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dual_conversion_pipeline(force=args.force)
    convert_for_android(force=args.force)

