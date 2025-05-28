'''
Quick script to generate dataset.json, an a json that contains the path for all images
'''
import json

# base path
base_path = "/data/myHAND/training/rgb/"
#number of training images
num_files = 15739

file_paths = [f"{base_path}{i:08d}.jpg" for i in range(num_files)]
output_file = "./train.json"

# Write the paths to the JSON file
with open(output_file, "w") as f:
    json.dump(file_paths, f, indent=4)

print(f"JSON file generated at {output_file}")