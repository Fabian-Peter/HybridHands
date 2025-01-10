import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file
json_path = './epoch_100.json'  # Replace with your JSON file path
with open(json_path, 'r') as f:
    data = json.load(f)  # This will be a list, not a dictionary

# Loop through the JSON to visualize predictions
for idx, entry in enumerate(data):
    img_path = entry['image_path']  # Path to the image
    pred_uv = np.array(entry['pred_uv'])  # Predicted 2D keypoints
    
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.scatter(pred_uv[:, 0], pred_uv[:, 1], c='r', label='Predicted', s=20)
    plt.legend()
    plt.title(f"Predictions on {img_path}")
    plt.axis('off')
    plt.show()
