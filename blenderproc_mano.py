import blenderproc as bproc
import numpy as np
import os
from pathlib import Path
import csv
import bpy
CAMERA_ROTATION_RANGE = (-0.7854, 0.7854)  # Range for in-plane rotation

def configure_camera_and_lights(objs, world_coords, output_dir, iteration_index):
    """Sets up cameras and lights and projects 3D coordinates dynamically."""
    for i in range(5):  # Adjust for desired number of camera poses
        # Random camera position
        location = np.random.uniform([-0.35, -0.35, 0.35], [0.35, 0.35, 0.35])
        poi = bproc.object.compute_poi(objs)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(*CAMERA_ROTATION_RANGE))

        # Add camera pose
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

        # Save coordinates for each camera pose
        project_and_save_coordinates(world_coords, cam2world_matrix, i, output_dir, iteration_index)

def clear_scene():
    """Deletes all objects from the current Blender scene."""
    # Get all objects in the current scene
    all_objects = bpy.data.objects
    # Loop through all objects and remove them
    for obj in all_objects:
        # Exclude the camera and lights if you want to keep them
        if obj.type != 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

def project_and_save_coordinates(world_coords, cam2world_matrix, camera_index, output_dir, iteration_index):
    """Projects 3D coordinates to 2D image space and saves them with Z-depth."""
    # Project points to image space
    image_coords = bproc.camera.project_points(world_coords)

    # Calculate Z-depth from camera coordinates
    world_coords_homogeneous = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))
    camera_coords = world_coords_homogeneous @ np.linalg.inv(cam2world_matrix).T
    z_depths = camera_coords[:, 2]

    # Create output directory for current iteration
    iteration_dir = Path(output_dir) / f"iteration_{iteration_index}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    csv_output_file = iteration_dir / f"{camera_index}_coordinates.csv"
    with open(csv_output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Image_X', 'Image_Y', 'Camera_Z'])
        for ic, z in zip(image_coords, z_depths):
            writer.writerow([ic[0], ic[1], z])

    print(f"Coordinates saved to: {csv_output_file}")

def setup_material():
    """Sets up a material with specific properties for the hand mesh."""
    material = bproc.material.create('Hand_Material')
    material.set_principled_shader_value("Base Color", [0.886, 0.659, 0.596, 1.0])  # Skin color
    material.set_principled_shader_value("Roughness", 0.8)
    material.set_principled_shader_value("Specular", 0.5)
    return material

def load_and_prepare_hand_mesh(file_path_obj, material):
    """Loads a hand mesh from an OBJ file and applies the given material."""
    objs = bproc.loader.load_obj(file_path_obj)
    for obj in objs:
        obj.enable_rigidbody(active=True, collision_shape='MESH')
        obj.set_rotation_euler([0, 0, 0])

    hand_mesh = objs[0]  # Assuming the hand mesh is the first object loaded
    if len(hand_mesh.blender_obj.data.materials) == 0:
        hand_mesh.blender_obj.data.materials.append(None)  # Create an empty material slot
    hand_mesh.set_material(0, material)
    return objs

def extract_all_coordinates(file_path_num):
    """Extracts all 3D coordinates from the given XYZ file."""
    all_coordinates = []
    with open(file_path_num, 'r') as file:
        lines = file.readlines()
        coords_list = [float(coord) for line in lines for coord in line.split()]
        for i in range(0, len(coords_list), 3):
            x, y, z = coords_list[i:i + 3]
            all_coordinates.append([x, y, z])
    return all_coordinates

def render_and_save(output_dir, iteration_index):
    """Renders the current scene and saves it to an HDF5 file in the correct iteration folder."""
    iteration_dir = Path(output_dir) / f"iteration_{iteration_index}"
    iteration_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    hdf5_output_path = iteration_dir
    data = bproc.renderer.render()
    bproc.writer.write_hdf5(str(hdf5_output_path), data)
    print(f"HDF5 saved to: {hdf5_output_path}")

def main():
    # Paths
    base_path = 'C:\\Users\\fabia\\Desktop\\HybridHands\\output\\poses\\mano'
    output_dir = "output/"
    num_iterations = 3

    # Initialize BlenderProc
    bproc.init()

    for iteration_index in range(num_iterations):
        # Reset the scene
        bproc.utility.reset_keyframes()
        clear_scene()
        # Dynamically construct file paths
        obj_file = os.path.join(base_path, f"{iteration_index}.obj")
        xyz_file = os.path.join(base_path, f"{iteration_index}.xyz")

        # Check if files exist
        if not os.path.exists(obj_file) or not os.path.exists(xyz_file):
            print(f"Missing files for iteration {iteration_index}. Skipping.")
            continue

        print(f"Processing iteration {iteration_index}: {obj_file} and {xyz_file}")

        # Extract 3D coordinates
        all_coordinates = extract_all_coordinates(xyz_file)
        world_coords = np.array(all_coordinates)

        # Set up material and load hand mesh
        material = setup_material()
        objs = load_and_prepare_hand_mesh(obj_file, material)

        # Configure cameras, lights, and save coordinates
        configure_camera_and_lights(objs, world_coords, output_dir, iteration_index)

        # Render the scene
        render_and_save(output_dir, iteration_index)

    print("Processing completed!")

# Run the main method
if __name__ == "__main__":
    main()
