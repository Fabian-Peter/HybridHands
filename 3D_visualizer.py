import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Define the xyz data
xyz = [
    [0.095742, 0.006383, 0.006183], [0.007538, 0.001187, 0.026946], [-0.023645, -0.00934, 0.029672], [-0.024262, -0.029534, 0.038882], [0.000981, 0.004929, 0.002847], [-0.023663, -0.014289, -0.002911], [-0.03265, -0.031787, 0.009554], [0.026961, -0.003563, -0.037091], [0.012884, -0.018151, -0.042999], [0.011877, -0.037049, -0.042143], [0.01396, 0.002435, -0.020523], [-0.005928, -0.018501, -0.019985], [-0.003592, -0.041661, -0.011429], [0.071602, -0.009157, 0.032018], [0.049347, -0.021255, 0.049562], [0.0276, -0.028779, 0.063892], [0.004560890141874552, -0.042859289795160294, 0.08567860722541809], [-0.015621240250766277, -0.0526626817882061, 0.044735901057720184], [-0.03478486090898514, -0.035872381180524826, 0.0347168892621994], [0.008709359914064407, -0.055629100650548935, 0.005527940113097429], [0.011546369642019272, -0.055411189794540405, -0.03221401944756508]
]

# Step 2: Extract X, Y, Z components and indices
x_coords = [point[0] for point in xyz]
y_coords = [point[1] for point in xyz]
z_coords = [point[2] for point in xyz]
indices = range(len(xyz))

# Step 3: Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Step 4: Scatter plot of the points
ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')

# Step 5: Label the points with their indices
for i, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
    ax.text(x, y, z, f'{i}', color='red', fontsize=8)

# Step 6: Labeling axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Step 7: Show the plot
plt.show()
