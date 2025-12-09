# train-model.py
import os
os.environ["KERAS_BACKEND"]

import keras
from keras import layers
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras.utils import image_dataset_from_directory

IMG = 256
BATCH = 32
SEED = 13

def make_ds(subdir):
    ds = keras.preprocessing.image_dataset_from_directory(
        f"data/{subdir}", label_mode="binary",
        image_size=(IMG, IMG), batch_size=BATCH,
        shuffle=(subdir=="train"), seed=SEED
    )
    return ds

train_ds = make_ds("train")
val_ds = make_ds("val")

AUTOTUNE = tf.data.AUTOTUNE
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="augment")

def prep(x, y, training=False):
    if training:
        x = augment(x, training=True)
    x = preprocess_input(x)
    return x, y

train_ds = train_ds.map(lambda x,y: prep(x,y,True), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

base = EfficientNetB0(include_top=False, input_shape=(IMG,IMG,3), weights="imagenet")
base.trainable = False

inputs = keras.Input((IMG,IMG,3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc"),
             keras.metrics.AUC(curve="PR", name="prauc"),
             keras.metrics.Precision(name="precision"),
             keras.metrics.Recall(name="recall")]
)

# Define checkpoint callback to save the best model
ckpt = keras.callbacks.ModelCheckpoint("models/autorgue_stage1.keras", monitor="val_prauc", mode="max", save_best_only=True)
early = keras.callbacks.EarlyStopping(monitor="val_prauc", mode="max", patience=8, restore_best_weights=True)
rlrop = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

# Train the model
model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=[ckpt, early, rlrop])

# Fine tune last ~30% of layers
base.trainable = True
for layer in base.layers[:-int(0.3*len(base.layers))]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc"),
                       keras.metrics.AUC(curve="PR", name="prauc"),
                       keras.metrics.Precision(name="precision"),
                       keras.metrics.Recall(name="recall")])

ckpt2 = keras.callbacks.ModelCheckpoint("models/autorogue_final.keras", monitor="val_prauc",
                                        mode="max", save_best_only=True)
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[ckpt2, early, rlrop])

print("Model training completed and best model saved.")