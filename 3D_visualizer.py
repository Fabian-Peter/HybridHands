import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Read the CSV file
csv_file = './output/render/image_0_coords.csv'  # replace with your actual file path
df = pd.read_csv(csv_file)

# Step 2: Extract the Image_X, Image_Y, and Camera_Z columns
image_x = df['Image_X']
image_y = df['Image_Y']
camera_z = df['Camera_Z']

# Step 3: Create a 3D plot using matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Step 4: Scatter plot of the points
ax.scatter(image_x, image_y, camera_z, c='b', marker='o')

# Step 5: Labeling axes
ax.set_xlabel('Image X')
ax.set_ylabel('Image Y')
ax.set_zlabel('Camera Z')

# Step 6: Show the plot
plt.show()