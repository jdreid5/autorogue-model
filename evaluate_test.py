# simplified_evaluate_model.py
import tensorflow as tf
from tensorflow.keras.models import load_model

print(tf.__version__)

# Load the trained model from the .keras file
model = load_model('models/autorogue_v0_final.keras')

# Print model summary to check if it's loaded correctly
model.summary()
