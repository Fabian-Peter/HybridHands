#!/usr/bin/env python3
import os
import glob
import random
from PIL import Image

#creates composit jpg of randomly selected image from background folders and rendered hand pose 
def main():
    input_images_folder = "./output/myRGBHAND/training/rgb"   
    background_folders = [
        "./background_images/dining_room",
        "./background_images/living_room",
        "./background_images/office",
        "./background_images/park",
        "./background_images/recreation_room"
    ]
    #TODO render straight into myHAND data set
    output_folder = "./background_images/__generated/"           
    
    file_types = ["png", "jpg", "jpeg"]
    jpeg_quality = 95

    os.makedirs(output_folder, exist_ok=True)

    background_files = []
    for folder in background_folders:
        for ext in file_types:
            background_files.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
            background_files.extend(glob.glob(os.path.join(folder, f"*.{ext.upper()}")))
    if not background_files:
        print("Error: No background images found in the specified folders!")
        return

    input_files = []
    input_files.extend(glob.glob(os.path.join(input_images_folder, "*.png")))
    input_files.extend(glob.glob(os.path.join(input_images_folder, "*.PNG")))
    if not input_files:
        print("Error: No input PNG images found in", input_images_folder)
        return

    for img_path in input_files:
        print(f"Processing {img_path}...")
        object_img = Image.open(img_path).convert("RGBA")
        width, height = object_img.size

        bg_path = random.choice(background_files)
        background_img = Image.open(bg_path).convert("RGBA")
        background_img = background_img.resize((width, height))

        background_img.paste(object_img, (0, 0), object_img)
        composite_img = background_img

        composite_img = composite_img.convert("RGB")

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_filename = f"{base_name}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        composite_img.save(output_path, "JPEG", quality=jpeg_quality)
        print(f"Saved composite image to: {output_path}")

if __name__ == "__main__":
    main()
