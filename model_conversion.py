import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("models/autorogue_v0.keras")

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to a file
with open('models/autorogue_v0.tflite', 'wb') as f:
    f.write(tflite_model)