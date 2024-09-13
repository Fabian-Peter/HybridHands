import h5py
import numpy as np
from PIL import Image

# Step 1: Load the HDF5 file
hdf5_file = './output/render/0.hdf5'  # Replace with the path to your HDF5 file
with h5py.File(hdf5_file, 'r') as h5f:
    # Assuming the image is stored in the dataset 'image_data'
    image_data = h5f['colors'][:]  # Replace 'dataset_name' with the actual dataset name in your file

# Step 2: Check if the image is grayscale or RGB
if len(image_data.shape) == 2:
    # Grayscale image (2D)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255.0  # Normalize to [0, 255]
    image_data = image_data.astype(np.uint8)  # Convert to uint8 type
    img = Image.fromarray(image_data, mode='L')  # 'L' mode is for grayscale images
elif len(image_data.shape) == 3 and image_data.shape[2] == 3:
    # RGB image (3D)
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255.0  # Normalize to [0, 255]
    image_data = image_data.astype(np.uint8)  # Convert to uint8 type
    img = Image.fromarray(image_data, mode='RGB')  # 'RGB' mode is for color images
else:
    raise ValueError("Unexpected image data shape. Expected 2D grayscale or 3D RGB image.")

# Step 3: Save the image as a PNG
output_png = 'output_image.png'
img.save(output_png)
print(f"Image saved as {output_png}")
