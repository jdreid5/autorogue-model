from PIL import Image
import os

def resize_image(input_path, output_path, max_dimensions, max_size_kb):
    with Image.open(input_path) as img:
        img.thumbnail(max_dimensions, Image.ANTIALIAS)
        
        # Initial save with a high quality to check size
        img.save(output_path, quality=95, optimize=True)
        
        # If the image is still too large, reduce the quality
        while os.path.getsize(output_path) > max_size_kb * 1024:
            with Image.open(output_path) as img:
                img.save(output_path, quality=85, optimize=True)

input_folder = 'healthy russets-001'
output_folder = 'healthy-russets-resize'
max_dimensions = (1024, 1024)  # Example max dimension size to ensure image is below 1.5 MB
max_size_kb = 1500  # 1.5 MB

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):from PIL import Image
import os

def resize_image(input_path, output_path, max_size_kb, max_dimensions=None):
    with Image.open(input_path) as img:
        if max_dimensions:
            img.thumbnail(max_dimensions, Image.LANCZOS)
        
        # Save with initial high quality to check size
        img.save(output_path, quality=95, optimize=True)
        
        # If the image is still too large, reduce the quality
        quality = 95
        while os.path.getsize(output_path) > max_size_kb * 1024 and quality > 10:
            quality -= 5
            with Image.open(output_path) as img:
                img.save(output_path, quality=quality, optimize=True)

input_folder = 'leaf roll russets-001/leaf roll russets'
output_folder = 'leaf-roll-russets-resize'
max_dimensions = (2048, 2048)  # Optional max dimension size
max_size_kb = 1500  # 1.5 MB

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        resize_image(input_path, output_path, max_size_kb, max_dimensions)
    else:
        print(f"Skipping unsupported file format: {filename}")