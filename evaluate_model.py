import os
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load validation data generator
validation_params = joblib.load('scripts/validation_params.pkl')

# Extract the directory and other parameters
validation_dir = validation_params.pop('train_dir')

# Print the contents of the validation directory for debugging
print(f"Contents of validation directory ({validation_dir}):")
for root, dirs, files in os.walk(validation_dir):
    print(root, dirs, len(files))

# Recreate ImageDataGenerator
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

# Recreate validation generator
validation_generator = validation_datagen.flow_from_directory(validation_dir, **validation_params)

# Load the trained model
model = load_model('models/autorogue_v0.keras')

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy*100:.2f}%')
print(f'Validation Loss: {loss:.4f}')