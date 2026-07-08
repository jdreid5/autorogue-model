# Autorogue

Autorogue is a potato disease detection pipeline that is being rebuilt from a
single whole-canopy binary classifier into a segment-then-classify system.

The target workflow is:

1. Train a multi-class leaf classifier on public leaf datasets.
2. Train a leaf segmenter on potato/canopy leaf-area annotations.
3. Segment incoming canopy images into leaf crops.
4. Classify each leaf crop.
5. Aggregate per-leaf predictions into a plant-level verdict.
6. Validate the full pipeline on held-out field/canopy imagery.

Current target classes are:

- `healthy`
- `leaf_roll`
- `mosaic`

The taxonomy is defined in `config.py`.

## Repository Structure

```text
config.py                     Central configuration, paths, classes, thresholds
preprocess.py                 Classifier input loading, augmentation, preprocessing
train_kfold.py                Multi-class MobileNetV3 classifier training
evaluate.py                   Multi-class classifier evaluation
visualize.py                  Grad-CAM visualisation for classifier attention
convert_tflite.py             TFLite export for classifier and segmenter
split-dataset.py              Source-aware train/val/test splitting

datasets/
  ingest.py                   Ingest public datasets into unified classes
  harmonize.py                Resize and background-normalize leaf images
  canopy_to_leaves.py         Segment field canopy images into weakly-labelled crops

segment/
  data.py                     Convert image/JSON leaf annotations to masks
  model.py                    Lightweight U-Net style leaf segmenter
  train.py                    Segmenter training entry point
  infer.py                    Canopy image -> leaf crop inference
  postprocess.py              Mask thresholding and connected components

pipeline/
  infer_canopy.py             End-to-end canopy inference and aggregation

domain_adapt.py               Fine-tune classifier on weak field leaf crops
field_validate.py             Field-only end-to-end validation
```

## Data Layout

Downloaded public datasets should be placed under `data/raw/`.

Required classifier sources:

```text
data/raw/mendeley_viral_foliar_tuber/
  Healthy leaf/
  PLRV/
  Mosaic/

data/raw/roboflow_viral/
  Potato___healthy/
  Potato___leafroll_virus/
  Potato___mosaic_virus/
```

Optional classifier sources:

```text
data/raw/plantvillage_potato_color/
  Potato___healthy/
  Potato___Early_blight/
  Potato___Late_blight/

data/raw/plantdoc/
  train/
    img/
    ann/
  test/
    img/
    ann/
```

For the current three-class taxonomy, PlantVillage contributes only
`Potato___healthy`; blight classes are ignored. PlantDoc contributes cropped
`Potato leaf` objects as `healthy`; PlantDoc blight classes are ignored.

Required segmentation source:

```text
data/raw/hutton_potato_leaf/
  IMG_8060.JPG
  IMG_8060.json
  ...
```

The Hutton/Zenodo JSON files must have matching image stems. These annotations
are used to train the segmenter and do not provide disease labels.

Existing field/canopy data should remain here:

```text
data/cropped-images/
  healthy-russets/
  leaf-roll-russets/
```

This data is plant-labelled, not leaf-labelled. It is used for weak domain
adaptation and field-only validation.

## Setup

Python 3.9+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

On native Windows, TensorFlow 2.11+ will usually run on CPU only. For GPU
training, use WSL2 or a compatible TensorFlow GPU setup.

## Run Order

### 1. Ingest Public Leaf Datasets

```bash
python -m datasets.ingest
```

This copies source images into unified class folders:

```text
data/public-leaves/
  healthy/
  leaf_roll/
  mosaic/
```

It also writes:

```text
data/public-leaves/manifest.jsonl
```

### 2. Harmonize Public Leaf Images

```bash
python -m datasets.harmonize
```

This standardizes image size and composites leaves over a neutral background.
The goal is to reduce shortcut learning from black/plain/public dataset
backgrounds.

Output:

```text
data/harmonized-leaves/
  healthy/
  leaf_roll/
  mosaic/
```

### 3. Split Classifier Data

```bash
python split-dataset.py
```

Output:

```text
data/split-images/
  train/
  val/
  test/
```

The splitter groups by source, class, and original image identity so augmented
siblings are kept in the same split while avoiding whole-dataset train/test
separation.

Expected rough split proportions are 70/15/15 per class.

### 4. Train the Leaf Classifier

```bash
python train_kfold.py
```

This trains a MobileNetV3 classifier with:

- softmax output over `healthy`, `leaf_roll`, `mosaic`
- k-fold cross-validation
- final deployment training on train + validation data

Primary output:

```text
models/autorogue_leaf_classifier.keras
```

Cross-validation summary:

```text
outputs/kfold_results.json
```

### 5. Prepare Segmentation Data

```bash
python -m segment.data --raw-root data/raw/hutton_potato_leaf
```

Output:

```text
data/segmentation/images/
data/segmentation/masks/
```

Each image/JSON pair becomes one normalized image and one binary leaf-area mask.

### 6. Train the Leaf Segmenter

```bash
python -m segment.train
```

Output:

```text
models/autorogue_leaf_segmenter.keras
```

The segmenter is a lightweight U-Net style semantic mask model. Connected
components are used later to turn the semantic mask into leaf crop candidates.

### 7. Create Weakly-Labelled Field Leaf Crops

```bash
python -m datasets.canopy_to_leaves
```

This runs the segmenter over `data/cropped-images/` and assigns every crop the
plant-level folder label.

Output:

```text
data/field-leaves/
  healthy/
  leaf_roll/
data/field-leaves/manifest.jsonl
```

There is no `mosaic` field folder unless field mosaic canopy images are added.

### 8. Domain Adapt the Classifier

```bash
python domain_adapt.py
```

This fine-tunes the public-data classifier on weakly-labelled field leaf crops.

Output:

```text
models/autorogue_leaf_classifier_field_adapted.keras
```

### 9. Validate End-to-End on Field Canopy Images

```bash
python field_validate.py
```

Output:

```text
outputs/field_validation_results.json
```

This evaluates:

```text
canopy image -> segment leaves -> classify leaves -> aggregate to plant verdict
```

The field validation set currently contains only `healthy` and `leaf_roll`
canopy images.

## End-to-End Inference

Run on a single canopy image:

```bash
python -m pipeline.infer_canopy path/to/image.jpg
```

This returns JSON containing:

- plant-level verdict
- confidence
- number of leaves found
- number of leaves used for aggregation
- per-class leaf fractions
- per-leaf predictions and bounding boxes

By default, it loads:

```text
models/autorogue_leaf_segmenter.keras
models/autorogue_leaf_classifier.keras
```

You can pass custom model paths:

```bash
python -m pipeline.infer_canopy path/to/image.jpg \
  --segmenter-path models/autorogue_leaf_segmenter.keras \
  --classifier-path models/autorogue_leaf_classifier_field_adapted.keras
```

## Mobile Export

After both models exist:

```bash
python convert_tflite.py
```

This exports:

```text
models/autorogue_leaf_classifier.tflite
models/autorogue_leaf_segmenter.tflite
models/android/labels.txt
models/android/model_info.json
```

## Interpreting Current Results

The public leaf classifier is only one part of the system. Its validation score
does not answer whether the full canopy workflow works.

The most important metric is `outputs/field_validation_results.json`, because it
tests the real deployment flow on field canopy images.

If field validation returns mostly `uncertain`, check these in order:

1. Are `data/field-leaves/healthy` and `data/field-leaves/leaf_roll` populated?
2. Are the field leaf crops visually good single-leaf crops?
3. Is the segmenter producing enough crops per canopy image?
4. Are confidence thresholds too strict?
5. Is the weak field fine-tuning set too small/noisy?

In a recent run, the pipeline produced only 101 field crops from 209 canopy
images and field validation predicted mostly `uncertain`. That suggests the
first bottleneck is likely segmentation/post-processing rather than the
classifier alone.

## Known Limitations

- Mosaic has no field/canopy examples in the current local dataset.
- Field adaptation is russet-specific because current canopy data covers only
  one variety.
- Leaf roll may require canopy-level posture/structure cues, not just individual
  leaf crops.
- The Hutton dataset may annotate leaf area rather than clean individual leaf
  instances; this limits instance segmentation quality.
- PlantDoc and PlantVillage are optional support datasets and are not direct
  sources for leaf roll or mosaic.

## Recommended Debugging Workflow

When the full pipeline underperforms, inspect segmentation first:

```text
data/field-leaves/healthy/
data/field-leaves/leaf_roll/
```

If crops are poor or too sparse, improve the segmenter/post-processing before
adjusting classifier training.

If crops look good but predictions are uncertain, lower or calibrate:

```python
PER_LEAF_CONFIDENCE_THRESHOLD
PLANT_CLASS_FRACTION_THRESHOLD
```

in `config.py`, then rerun:

```bash
python field_validate.py
```

## Main Artifacts

```text
models/autorogue_leaf_classifier.keras
models/autorogue_leaf_segmenter.keras
models/autorogue_leaf_classifier_field_adapted.keras
outputs/kfold_results.json
outputs/field_validation_results.json
```
