import json

# Define the base path and number of files
base_path = "/data/myHAND/training/rgb/"
num_files = 15739

# Generate the list of file paths
file_paths = [f"{base_path}{i:08d}.jpg" for i in range(num_files)]

# Define the output JSON file path
output_file = "./train.json"

# Write the paths to the JSON file
with open(output_file, "w") as f:
    json.dump(file_paths, f, indent=4)

print(f"JSON file generated at {output_file}")