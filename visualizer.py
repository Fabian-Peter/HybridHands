
import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# File paths
hdf5_file_path = './output/render/3.hdf5'
csv_file_path = './output/render/image_4_coords.csv'
output_image_path = './output/render/output_image_with_points.png'

# Load the image from .hdf5 file
with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    # Assuming the image data is stored in a dataset named 'image'
    image_data = hdf5_file['colors'][:]
    
# Convert image data to an image format suitable for Pillow
# Assuming the image data is in grayscale format. Adjust if necessary.
image = Image.fromarray(np.uint8(image_data))

# Read 2D coordinates from CSV
coords_df = pd.read_csv(csv_file_path, header=None, names=['x', 'y'])

# Convert coordinates to list of tuples
coordinates = [(int(row['x']), int(row['y'])) for index, row in coords_df.iterrows()]

# Draw coordinates on the image
draw = ImageDraw.Draw(image)

# Draw points on the image
for (x, y) in coordinates:
    # Draw a red dot for each coordinate
    draw.ellipse((x-3, y-3, x+3, y+3), fill='red', outline='red')

# Save the result
image.save(output_image_path)

# Display the image with coordinates
plt.imshow(image)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
