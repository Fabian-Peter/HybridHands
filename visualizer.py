import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

# Directory paths
output_dir = './output/iteration_0'

# Loop through all files in the directory
for i in range(5):  # Adjust the range as needed
    hdf5_file_path = os.path.join(output_dir, f'{i}.hdf5')
    csv_file_path = os.path.join(output_dir, f'{i}_coordinates.csv')
    output_image_path = os.path.join(output_dir, f'output_image_with_points_{i}.png')

    # Load the image from .hdf5 file
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Assuming the image data is stored in a dataset named 'colors'
        image_data = hdf5_file['colors'][:]
        
    # Convert image data to an image format suitable for Pillow
    if image_data.ndim == 3 and image_data.shape[-1] == 3:  # If the last dimension has 3 channels (RGB)
        image = Image.fromarray(np.uint8(image_data), 'RGB')
    else:
        image = Image.fromarray(np.uint8(image_data))  # Assuming grayscale

    # Read 2D coordinates from CSV
    coords_df = pd.read_csv(csv_file_path)
    image_x = coords_df['Image_X'].astype(int)
    image_y = coords_df['Image_Y'].astype(int)

    # Convert coordinates to list of tuples
    coordinates = list(zip(image_x, image_y))

    # Draw coordinates on the image
    draw = ImageDraw.Draw(image)

    # Optionally, load a font (system default or custom)
    try:
        font = ImageFont.truetype("arial.ttf", size=12)  # Adjust the font size
    except IOError:
        font = ImageFont.load_default()  # Fallback to default if the font isn't found

    # Draw points and index numbers on the image
    for idx, (x, y) in enumerate(coordinates):
        # Draw a red dot for each coordinate
        draw.ellipse((x-3, y-3, x+3, y+3), fill='red', outline='red')

        # Draw the index number next to the point
        draw.text((x+5, y-5), str(idx), fill='white', font=font)  # Adjust offset to prevent overlap with the dot

    # Save the result with points drawn
    image.save(output_image_path)

    # Display the image with coordinates
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

    print(f"Processed and saved: {output_image_path}")
