import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

# Directory paths
output_dir = './output/myMarkerHAND/training/rgb'

# Loop through all files in the directory
for i in range(50):  # Adjust the range as needed
    # Change the file paths to point to .jpg images instead of .hdf5
    jpg_file_path = os.path.join(output_dir, f'{i:08}.jpg')  # JPG file
    json_file_path = os.path.join(output_dir, f'{i:08}.json')
    output_image_path = os.path.join(output_dir, f'output_image_with_points_{i}.png')

    # Load the image from .jpg file using Pillow
    image = Image.open(jpg_file_path)

    # Read 3D coordinates (vertices) from the JSON file
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
        xyz_coordinates = json_data['uv']  # Extract the "uv" field (3D coordinates)

    # Convert 3D coordinates to 2D coordinates (x, y), disregarding z
    coordinates = [(int(x), int(y)) for x, y in xyz_coordinates]

    # Draw coordinates on the image
    draw = ImageDraw.Draw(image)

    # Optionally, load a font (system default or custom)
    try:
        font = ImageFont.truetype("arial.ttf", size=9)  # Adjust the font size
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
