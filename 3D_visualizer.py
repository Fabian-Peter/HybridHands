import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json

output_dir = './output/myRGBHAND/training/rgb'

# Define the hand skeleton connections
skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

for i in range(50):  # Adjust the range as needed
    json_file_path = os.path.join(output_dir, f'{i:08}.json')
    
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
        xyz = json_data['xyz'] 
        
        x_coords = [point[0] for point in xyz]
        y_coords = [point[1] for point in xyz]
        z_coords = [point[2] for point in xyz]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot keypoints
        ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')

        # Plot skeleton connections
        for start, end in skeleton:
            ax.plot([x_coords[start], x_coords[end]], 
                    [y_coords[start], y_coords[end]], 
                    [z_coords[start], z_coords[end]], c='black')

        # Add labels
        for i, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
            ax.text(x, y, z, f'{i}', color='red', fontsize=8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
