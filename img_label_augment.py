import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Directories
healthy_dir = 'data/healthy-russets'
infected_dir = 'data/leaf-roll-russets'
augmented_healthy_dir = 'data/augmented_images/healthy-russets'
augmented_infected_dir = 'data/augmented_images/leaf-roll-russets'

# Create directories if they don't exist
os.makedirs(augmented_healthy_dir, exist_ok=True)
os.makedirs(augmented_infected_dir, exist_ok=True)

# Initialize ImageDataGenerator
datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='wrap',
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.2
)

# Function to augment images
def augment_images(source_dir, target_dir, prefix):
    for img_file in os.listdir(source_dir):
        img_path = os.path.join(source_dir, img_file)
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=target_dir, save_prefix=prefix, save_format='jpeg'):
            i += 1
            # Generate 5 augmented images per original image
            if i > 5:
                break

# Augment healthy and infected images
augment_images(healthy_dir, augmented_healthy_dir, 'healthy_aug')
augment_images(infected_dir, augmented_infected_dir, 'infected_aug')

print("Data augmentation complete.")