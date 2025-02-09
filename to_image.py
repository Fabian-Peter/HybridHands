import os
import h5py
import numpy as np
from PIL import Image

# Input and output folder paths
input_folder = "./output/myRGBHAND/training/rgb"
output_folder = "./output/myRGBHAND/training/rgb"

# Total number of digits for leading zeros (adjust based on your expected file count)
num_digits = 8

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each .hdf5 file in the folder
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".hdf5"):
        # Extract the base number from the filename (e.g., "0" from "0.hdf5")
        base_number = os.path.splitext(filename)[0]
        
        # Pad the base number with leading zeros
        padded_number = str(base_number).zfill(num_digits)
        
        hdf5_path = os.path.join(input_folder, filename)
        
        with h5py.File(hdf5_path, 'r') as f:
            # Access the "colors" dataset
            if "colors" in f:
                colors_data = np.array(f["colors"])
                
                # Check the shape of the dataset
                if colors_data.ndim == 3:  # Single image case
                    # Save as .jpg with padded number
                    output_path = os.path.join(output_folder, f"{padded_number}.jpg")
                    
                    # Convert to Image object and save
                    img = Image.fromarray(colors_data.astype(np.uint8))
                    img.save(output_path, format="JPEG")
                else:
                    print(f"Unexpected shape {colors_data.shape} in {filename}")
            else:
                print(f"'colors' dataset not found in {filename}")