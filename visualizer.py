import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# Directory paths
output_dir = './output/render/'

# Loop through all files in the directory
for i in range(5):  # Assuming you have 5 sets of files (adjust the range as needed)
    
    hdf5_file_path = os.path.join(output_dir, f'{i}.hdf5')
    csv_file_path = os.path.join(output_dir, f'image_{i}_coords.csv')
    output_image_path = os.path.join(output_dir, f'output_image_with_points_{i}.png')

    # Load the image from .hdf5 file
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Assuming the image data is stored in a dataset named 'colors'
        image_data = hdf5_file['colors'][:]
        
    # Convert image data to an image format suitable for Pillow
    # Assuming the image data is RGB or grayscale. Adjust if necessary.
    if image_data.shape[-1] == 3:  # If the last dimension has 3 channels (RGB)
        image = Image.fromarray(np.uint8(image_data), 'RGB')
    else:
        image = Image.fromarray(np.uint8(image_data))  # Assuming grayscale

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

    # Save the result with points drawn
    image.save(output_image_path)

    # Display the image with coordinates
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

    print(f"Processed and saved: {output_image_path}")
