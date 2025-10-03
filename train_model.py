import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Create ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Split for validation
)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    'data/training_data', 
    target_size=(224, 298),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/training_data', 
    target_size=(224, 298),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Build the model
input_tensor = Input(shape=(224,298,3))
base_model = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
base_model.trainable = False # Freeze the base model

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define checkpoint callback to save the best model
checkpoint = ModelCheckpoint('models/autorogue_v0.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[checkpoint]
)

# Save the final model explicitly
model.save('models/autorogue_v0_final.keras')

# Print output shape of trained model
print("Training completed. Model summary:")
model.summary()

# Print the output shapes of each layer
for layer in model.layers:
    print(f"Layer: {layer.name}, Output shape: {layer.output.shape}")

# Save training history
joblib.dump(history.history, 'scripts/training_history.pkl')

print("Model training completed and best model saved.")