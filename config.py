# config.py
"""
Centralized configuration for the Autorogue segment-then-classify pipeline.

The rebuilt pipeline trains a multi-class leaf classifier and an optional leaf
segmenter, then combines per-leaf predictions into a plant-level decision.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
DATA_ROOT = Path("data")

# Legacy/current canopy data.
CROPPED_DIR = DATA_ROOT / "cropped-images"
CANOPY_DIR = CROPPED_DIR

# Public leaf classifier data.
RAW_DATA_DIR = DATA_ROOT / "raw"
PUBLIC_LEAF_DIR = DATA_ROOT / "public-leaves"
HARMONIZED_LEAF_DIR = DATA_ROOT / "harmonized-leaves"
SPLIT_DIR = DATA_ROOT / "split-images"
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"

# Segmentation/domain-adaptation data.
SEGMENTATION_DATA_DIR = DATA_ROOT / "segmentation"
SEGMENTATION_IMAGE_DIR = SEGMENTATION_DATA_DIR / "images"
SEGMENTATION_MASK_DIR = SEGMENTATION_DATA_DIR / "masks"
FIELD_SEGMENTATION_DATA_DIR = DATA_ROOT / "segmentation-field"
FIELD_SEGMENTATION_IMAGE_DIR = FIELD_SEGMENTATION_DATA_DIR / "images"
FIELD_SEGMENTATION_MASK_DIR = FIELD_SEGMENTATION_DATA_DIR / "masks"
FIELD_LEAF_DIR = DATA_ROOT / "field-leaves"
FIELD_SPLIT_DIR = DATA_ROOT / "field-split-images"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# CLASSES AND DATASET MAPPINGS
# =============================================================================
CLASSES = ["healthy", "leaf_roll", "mosaic"]
NUM_CLASSES = len(CLASSES)
CLASS_NAMES = {
    0: "Healthy",
    1: "Leaf Roll",
    2: "Mosaic",
}
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}

# Map source-specific folder/label names into the unified taxonomy.
SOURCE_CLASS_MAP = {
    "mendeley_viral_foliar_tuber": {
        "Healthy leaf": "healthy",
        "healthy": "healthy",
        "PLRV": "leaf_roll",
        "Potato Leaf Roll Virus": "leaf_roll",
        "leaf_roll": "leaf_roll",
        "Mosaic": "mosaic",
        "Mosaic Virus": "mosaic",
        "mosaic": "mosaic",
        "PSTVD": "ignore",
        "PVY cracking": "ignore",
    },
    "roboflow_viral": {
        "Potato___healthy": "healthy",
        "Potato___leafroll_virus": "leaf_roll",
        "Potato___mosaic_virus": "mosaic",
    },
    "plantvillage_potato_color": {
        "Potato___healthy": "healthy",
        "Potato___Early_blight": "ignore",
        "Potato___Late_blight": "ignore",
    },
    "plantdoc": {
        "Potato leaf": "healthy",
        "Potato leaf early blight": "ignore",
        "Potato leaf late blight": "ignore",
    },
    "canopy_weak": {
        "healthy-russets": "healthy",
        "leaf-roll-russets": "leaf_roll",
    },
}

# Dataset source folders under RAW_DATA_DIR. Downloaded datasets can be placed
# here using these source names before running datasets/ingest.py.
DATASET_SOURCES = {
    "mendeley_viral_foliar_tuber": RAW_DATA_DIR / "mendeley_viral_foliar_tuber",
    "roboflow_viral": RAW_DATA_DIR / "roboflow_viral",
    "plantvillage_potato_color": RAW_DATA_DIR / "plantvillage_potato_color",
    "plantdoc": RAW_DATA_DIR / "plantdoc",
}

# =============================================================================
# IMAGE SETTINGS
# =============================================================================
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
SEGMENTATION_IMG_SIZE = 384
SEGMENTATION_IMG_SHAPE = (SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE, 3)

# Neutral background behind every classifier input: masked-out canopy pixels,
# harmonized public leaves, letterbox padding, and augmentation fill. These must
# agree or the model sees a different background at training and inference time.
NEUTRAL_BACKGROUND_VALUE = 128
NEUTRAL_BACKGROUND_RGB = (NEUTRAL_BACKGROUND_VALUE,) * 3

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 16
SEGMENTATION_BATCH_SIZE = 4
SEED = 42

# K-Fold Cross Validation
N_FOLDS = 5

# Two-stage classifier training
STAGE1_EPOCHS = 40
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 30
STAGE2_LR = 1e-5
FINETUNE_LAYERS_PERCENT = 0.3

# Segmenter training
SEGMENTATION_EPOCHS = 50
SEGMENTATION_LR = 1e-4
SEGMENTATION_FINETUNE_LR = 2e-5
SEGMENTATION_FIELD_SAMPLE_WEIGHT = 3.0
SEGMENTATION_FAILURE_SAMPLE_COUNT = 100

# Regularization
DROPOUT_RATE = 0.4
LABEL_SMOOTHING = 0.1

# Callbacks
EARLY_STOP_PATIENCE = 12
MIN_DELTA = 0.001

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
ROTATION_RANGE = 30
ZOOM_RANGE = 0.15
BRIGHTNESS_RANGE = 0.2
CONTRAST_RANGE = 0.2
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False

MIXUP_ALPHA = 0.0
CUTMIX_ALPHA = 0.0
MIX_PROB = 0.0

# =============================================================================
# TEST-TIME AUGMENTATION
# =============================================================================
TTA_AUGMENTS = 5

# =============================================================================
# MODEL SETTINGS
# =============================================================================
BACKBONE = "MobileNetV3Small"
PRETRAINED_WEIGHTS = "imagenet"
CLASSIFIER_MODEL_NAME = "autorogue_leaf_classifier.keras"
SEGMENTER_MODEL_NAME = "autorogue_leaf_segmenter.keras"

# =============================================================================
# SEGMENTATION AND AGGREGATION
# =============================================================================
SEGMENTATION_THRESHOLD = 0.5
MIN_LEAF_AREA_RATIO = 0.002
MAX_LEAF_AREA_RATIO = 0.35
MASK_FEATHER_RADIUS = 2
MAX_LEAVES_PER_IMAGE = 64
COMPONENT_EROSION_ITERATIONS = 1
COMPONENT_DILATION_ITERATIONS = 1
MIN_ORIGINAL_CROP_SIZE = 64
MIN_ORIGINAL_CROP_AREA = 4096
MAX_CROP_ASPECT_RATIO = 4.0
MAX_CROP_UPSAMPLE_FACTOR = 3.5
MIN_CROP_MASK_COVERAGE = 0.12
CROP_CONTACT_SHEET_SAMPLES = 80

PER_LEAF_CONFIDENCE_THRESHOLD = 0.7
PLANT_CLASS_FRACTION_THRESHOLD = {
    "leaf_roll": 0.2,
    "mosaic": 0.15,
}
PLANT_HEALTHY_MIN_CONFIDENCE = 0.6

# =============================================================================
# TFLITE CONVERSION
# =============================================================================
QUANTIZE_INT8 = True
CLASSIFIER_TFLITE_MODEL_NAME = "autorogue_leaf_classifier.tflite"
SEGMENTER_TFLITE_MODEL_NAME = "autorogue_leaf_segmenter.tflite"
TFLITE_MODEL_NAME = CLASSIFIER_TFLITE_MODEL_NAME

# =============================================================================
# EVALUATION
# =============================================================================
DECISION_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = PER_LEAF_CONFIDENCE_THRESHOLD
FIELD_TEST_FRACTION = 0.2
FIELD_VAL_FRACTION = 0.2
MIN_FIELD_MACRO_F1_FOR_EXPORT = 0.5
MAX_FIELD_UNCERTAIN_RATE_FOR_EXPORT = 0.25

