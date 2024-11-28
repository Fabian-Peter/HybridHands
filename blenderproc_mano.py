import blenderproc as bproc
import numpy as np
import os
import csv

# ----------------------------------------------
# Helper Function to Extract 3D Coordinates
# ----------------------------------------------
def extract_all_coordinates(file_path_num):
    """
    Extracts all 3D coordinates from the given XYZ file.

    Args:
        file_path_num: The path to the XYZ file.

    Returns:
        A list of all 3D coordinates in the file.
    """
    all_coordinates = []
    with open(file_path_num, 'r') as file:
        lines = file.readlines()
        coords_list = [float(coord) for line in lines for coord in line.split()]
        
        # Group into (x, y, z) tuples
        for i in range(0, len(coords_list), 3):
            x, y, z = coords_list[i:i + 3]
            all_coordinates.append([x, y, z])
    
    return all_coordinates

# ----------------------------------------------
# Helper Function to Project 3D to 2D and Save as CSV
# ----------------------------------------------
def project_coordinates_and_save(world_coords, cam2world_matrix, output_dir, camera_index):
    """
    Projects 3D coordinates into the 2D image plane and saves them as a CSV.

    Args:
        world_coords: The 3D world coordinates to project.
        cam2world_matrix: Camera transformation matrix.
        output_dir: Directory to save the CSV file.
        camera_index: Index of the camera for file naming.
    """
    # Project 3D coordinates to 2D image space
    image_coords = bproc.camera.project_points(world_coords)
    
    # Convert world coordinates to camera coordinates
    world_coords_homogeneous = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))
    camera_coords = world_coords_homogeneous @ np.linalg.inv(cam2world_matrix).T
    z_depths = camera_coords[:, 2]

    # Prepare the output directory
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{camera_index}_coordinates.csv")
    
    # Write to CSV
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Image_X', 'Image_Y', 'Camera_Z'])
        for ic, z in zip(image_coords, z_depths):
            writer.writerow([ic[0], ic[1], z])
    print(f"Coordinates saved to: {csv_path}")

# ----------------------------------------------
# Initialization and Configuration
# ----------------------------------------------
# Vars
number = "0000"
base_path = 'C:\\Users\\fabia\\Desktop\\HybridHands\\output\\poses\\mano'
file_path_obj = os.path.join(base_path, f"{number}.obj")
file_path_num = os.path.join(base_path, f"{number}.xyz")

# Extract coordinates from the XYZ file
all_coordinates = extract_all_coordinates(file_path_num)
print(f"Extracted Coordinates: {all_coordinates}")

# Convert to NumPy array for processing
world_coords = np.array(all_coordinates)

# Initialize BlenderProc
bproc.init()


# ----------------------------------------------
# Material and Object Setup
# ----------------------------------------------
# Create a material
material = bproc.material.create('Hand_Material')
material.set_principled_shader_value("Base Color", [0.886, 0.659, 0.596, 1.0])  # Skin color
material.set_principled_shader_value("Roughness", 0.8)
material.set_principled_shader_value("Specular", 0.5)

# Load the hand mesh
objs = bproc.loader.load_obj(file_path_obj)

for obj in objs:
        obj.enable_rigidbody(active=True, collision_shape='MESH')
        obj.set_rotation_euler([0, 0, 0]) 
    

hand_mesh = objs[0]  # Assuming the hand mesh is the first object loaded

# Add a material slot if none exist
if len(hand_mesh.blender_obj.data.materials) == 0:
    hand_mesh.blender_obj.data.materials.append(None)  # Create an empty material slot

# Assign the material to the first material slot (index 0)
hand_mesh.set_material(0, material)

# ----------------------------------------------
# Light and Camera Configuration
# ----------------------------------------------
# Create a point light
light = bproc.types.Light()
light.set_location([2, -2, 0])
light.set_energy(300)

# Compute point of interest (POI)
poi = bproc.object.compute_poi(objs)

# Output directory for CSV files
csv_output_dir = "output/"

# Sample camera poses
for i in range(5):  # Number of camera poses
    location = np.random.uniform([-0.35, -0.35, 0.35], [0.35, 0.35, 0.35])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

    # Project coordinates and save to CSV
    project_coordinates_and_save(world_coords, cam2world_matrix, csv_output_dir, i)

# ----------------------------------------------
# Rendering and Saving
# ----------------------------------------------
# Render the scene
data = bproc.renderer.render()

# Write the rendering to an HDF5 file
bproc.writer.write_hdf5("output/", data)
