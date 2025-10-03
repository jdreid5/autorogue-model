import os
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Define directories
train_dir = 'data/training_data'

# Parameters
img_size = 224
batch_size = 32

# ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 # Split for validation
)

# Save the parameters instead of the generator objects
params = {
    'train_dir': train_dir,
    'target_size': (img_size, img_size),
    'batch_size': batch_size,
    'class_mode': 'binary',
    'subset': 'training'
}

# Save the generators for later use
joblib.dump(params, 'scripts/train_params.pkl')

params['subset'] = 'validation'
joblib.dump(params, 'scripts/validation_params.pkl')

print("Data generators created successfully.")