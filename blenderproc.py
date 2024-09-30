import blenderproc as bproc
from pathlib import Path
import random
import argparse
import bpy
import numpy as np
import csv

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('output_dir', nargs='?', default="./output/render", help="Path to where the final files will be saved")
args = parser.parse_args()
xyz_file_path = 'C:\\Users\\fabia\\Desktop\\HybridHands\\output\\rand_0_joints.xyz'
indices_to_extract = [5, 10, 15, 20, 25]
all_coordinates = []
extracted_coordinates = []
bproc.init()

# Load the objects into the scene
objs = bproc.loader.load_obj("./output/rand_0_skin.obj")
for obj in objs:
    obj.enable_rigidbody(active=True, collision_shape='MESH') 
    obj.set_shading_mode("smooth")
    obj.set_rotation_euler([0, 0, 0])  # Optional: set initial object orientation

# load textures
materials = bproc.material.collect_all()
for mat in materials:
    normal = bpy.data.images.load(r"C:\\Users\\fabia\\Desktop\\NIMBLE_model\\output\\rand_0_normal.png")
    spec = bpy.data.images.load(r"C:\\Users\\fabia\\Desktop\\NIMBLE_model\\output\\rand_0_spec.png")
    dif = bpy.data.images.load(r"C:\\Users\\fabia\\Desktop\\NIMBLE_model\\output\\rand_0_diffuse.png")
    mat.set_principled_shader_value("Base Color", dif)
    mat.set_principled_shader_value("Normal", normal)

def extract_coordinates(filepath, indices):
    
    # Read file and store all coordinates
    with open(filepath, 'r') as file:
        lines = file.readlines()
        # Flatten the list of coordinates
        coords_list = [float(coord) for line in lines for coord in line.split()]
        
        # Group the coordinates into x, y, z tuples
        for i in range(0, len(coords_list), 3):
            x = coords_list[i]
            y = coords_list[i + 1]
            z = coords_list[i + 2]
            all_coordinates.append([x, y, z])
    
    # Now extract the required coordinates at the given indices
    for idx in indices:
        extracted_coordinates.append(all_coordinates[idx - 1])  # Indices are 1-based, adjust by -1

    return extracted_coordinates

# Usage: Provide your xyz_file_path
coordinates = extract_coordinates(xyz_file_path, indices_to_extract)

# Group spheres so we can hide/show them easily later
spheres = []

# Create spheres at the extracted coordinates and enable collision
for coord in extracted_coordinates:
    sphere = bproc.object.create_primitive('SPHERE', scale=[3.5, 3.5, 3.5])

    sphere.enable_rigidbody(active=True, collision_shape='SPHERE')   
    # Slightly offset the sphere along the Z-axis to ensure it rests on top of the mesh
    offset_coord = [coord[0], coord[1], coord[2] + 3.5]  # Offset by the sphere's radius (3.5 in this case)
    sphere.set_location(offset_coord)

    # Create a new unique material for the sphere
    mat_marker = bproc.material.create("MarkerMaterial_" + str(coord))  # Ensure unique name

    # Define your material properties
    grey_col = 1.0  # Example grey color
    rough = np.random.uniform(0, 0.5)
    spec = np.random.uniform(0.3, 1.0)
    met = np.random.uniform(0, 0.5)

    # Set the material properties
    mat_marker.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
    mat_marker.set_principled_shader_value("Roughness", rough)
    mat_marker.set_principled_shader_value("Specular", spec)
    mat_marker.set_principled_shader_value("Metallic", met)
   
    sphere.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)
    sphere.hide(False)

    # Check if the sphere has any materials; if not, add one
    if len(sphere.get_materials()) == 0:
        sphere.add_material(mat_marker)  # Create a new material slot and add the material
    else:
        sphere.set_material(0, mat_marker)  # Assign the material to the sphere at index 0
    
    # Add the sphere to the list for future use
    spheres.append(sphere)

# World coordinates to be projected to 2D
world_coords = np.array(all_coordinates)

room_planes = [
    bproc.object.create_primitive('PLANE', scale=[-644.637,-644.637,-644.637], location=[0, -300, 0], rotation=[90, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[-644.637,-644.637,-644.637], location=[0, 300, 0], rotation=[90, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[-644.637,-644.637,-644.637], location=[0, 0, -300], rotation=[0, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[-644.637,-644.637,-644.637], location=[0, 0, 300], rotation=[0, 0, 0])
]

for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

# Store camera and light parameters for both passes
camera_poses = []
light_positions = []

# different camera angles
for i in range(5):
    # Random camera location
    location = np.random.uniform([-200, -200, 180], [200, 200, 200])
    
    # Compute rotation for camera to point towards poi
    poi = bproc.object.compute_poi(objs)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        poi - location, 
        inplane_rot=np.random.uniform(-0.7854, 0.7854)  # random in-plane rotation
    )
    
    # Given location for light
    given_location = np.array([8.21917, -37.8768, -57.4157])  # Replace x, y, z with your coordinates
    fixed_distance = 100.0  # Fixed distance from the given location

    # Generate random angles for spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)  # Angle around the vertical axis
    phi = np.random.uniform(0, np.pi)        # Angle from the vertical axis

    # Calculate random position
    random_x = fixed_distance * np.sin(phi) * np.cos(theta)
    random_y = fixed_distance * np.sin(phi) * np.sin(theta)
    random_z = fixed_distance * np.cos(phi)

    # Calculate the final light position
    light_position = given_location + np.array([random_x, random_y, random_z])

    # Create the light at the calculated position
    area_light = bproc.types.Light()
    area_light.set_type("POINT")
    area_light.set_location(light_position)
    area_light.set_radius(100)
    area_light.set_energy(500000)

    # Store camera and light positions
    camera_poses.append((location, rotation_matrix))
    light_positions.append(light_position)

    # Set camera pose in the scene
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)  # Add each pose individually

    # Project 3D world coordinates to 2D image coordinates for this camera pose
    image_coords = bproc.camera.project_points(world_coords)

    # Compute the Z-depth of each point in camera space
    world_coords_homogeneous = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))  # Convert to homogeneous coordinates
    camera_coords = world_coords_homogeneous @ np.linalg.inv(cam2world_matrix).T  # Transform to camera space
    z_depths = camera_coords[:, 2]  # The Z coordinate in camera space

    # Save 3D world coordinates, corresponding 2D image coordinates, and Z-depth to CSV file
    csv_output_file = Path(args.output_dir) / f"image_with_spheres_{i}_coords.csv"
    with open(csv_output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow(['World_X', 'World_Y', 'World_Z', 'Image_X', 'Image_Y', 'Camera_Z'])
        for wc, ic, z in zip(world_coords, image_coords, z_depths):
            writer.writerow([wc[0], wc[1], wc[2], ic[0], ic[1], z])

    print(f"Prepared camera pose {i} with spheres visible")

# Render images with spheres visible
data_with_spheres = bproc.renderer.render()

# Save rendered images and data with spheres
bproc.writer.write_hdf5(args.output_dir + "/with_spheres", data_with_spheres)

# Hide spheres and render 5 more images
for sphere in spheres:
    sphere.hide(True)

# Render images with spheres hidden
data_without_spheres = bproc.renderer.render()

# Save rendered images and data without spheres
bproc.writer.write_hdf5(args.output_dir + "/without_spheres", data_without_spheres)

print("Rendered images with and without spheres.")
