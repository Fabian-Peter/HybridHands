import blenderproc as bproc
import numpy as np
import os
from pathlib import Path
import json
import bpy
import random
import json
import glob 

CAMERA_ROTATION_RANGE = (-0.7854, 0.7854)  # Range for in-plane rotation
iteration_counter = 0


def configure_camera_and_lights(objs, world_coords, output_dir):
    """
    Sets up a single camera and light, projects coordinates, and saves the JSON file.
    """
    global iteration_counter

    # Predefined light positions
    light_positions = [
        [0.383889, 0, 0],
        [0, 0.383889, 0],
        [0, 0, 0.383889],
        [-0.383889, 0, 0],
        [0, -0.383889, 0],
        [0, 0, -0.383889]
    ]

    # Randomly select one position
    selected_position = random.choice(light_positions)

    # Create and place the light
    light = bproc.types.Light()
    light.set_location(selected_position)
    light.set_energy(10)
    print(f"Light created at: {selected_position}")

    # Fixed distance from the object
    radius = 0.5

    # Random spherical coordinates for a single pose
    theta = np.random.uniform(0, np.pi / 2)  # Polar angle (upper hemisphere)
    phi = np.random.uniform(0, 2 * np.pi)   # Azimuthal angle

    # Spherical to Cartesian conversion
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    location = np.array([x, y, z])

    # Compute point of interest
    poi = bproc.object.compute_poi(objs)

    # Compute rotation matrix
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        poi - location, 
        inplane_rot=np.random.uniform(*CAMERA_ROTATION_RANGE)
    )

    # Add camera pose
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

    # Save coordinates for the camera pose
    project_and_save_coordinates(world_coords, cam2world_matrix, output_dir, objs[0])

     
    


def clear_scene():
    """Deletes all objects from the current Blender scene."""
    # Get all objects in the current scene
    all_objects = bpy.data.objects
    # Loop through all objects and remove them
    for obj in all_objects:
        # Exclude the camera and lights if you want to keep them
        if obj.type != 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)




def project_and_save_coordinates(world_coords, cam2world_matrix, output_dir, obj):
    """
    Projects 3D coordinates to 2D image space and saves them with Z-depth,
    the camera's intrinsic matrix, pixel coordinates, and object vertices in JSON format.
    """
    global iteration_counter

    # Calculate Z-depth from camera coordinates
    world_coords_homogeneous = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))
    camera_coords = world_coords_homogeneous @ np.linalg.inv(cam2world_matrix).T
    z_depths = camera_coords[:, 2]

    # Project 3D points to image space (pixel coordinates)
    image_coords = bproc.camera.project_points(world_coords)

    # Get the camera's intrinsic matrix
    intrinsic_matrix = bproc.camera.get_intrinsics_as_K_matrix()

    # Use bpy to extract the object's vertices in world space
    obj_name = obj.get_name()
    bpy_obj = bpy.data.objects.get(obj_name)
    if bpy_obj is None:
        raise ValueError(f"Object '{obj_name}' not found in bpy context.")

    # Extract vertices in world coordinates
    vertices_world = np.array([bpy_obj.matrix_world @ v.co for v in bpy_obj.data.vertices])

    # Convert vertices to camera space
    vertices_homogeneous = np.hstack((vertices_world, np.ones((vertices_world.shape[0], 1))))
    vertices_camera = vertices_homogeneous @ np.linalg.inv(cam2world_matrix).T

    # Prepare JSON data
    json_data = {
        "uv": [[ic[0], ic[1]] for ic in image_coords],
        "xyz": [[wc[0], wc[1], z] for wc, z in zip(world_coords, z_depths)],
        "hand_type": [1],
        "K": intrinsic_matrix.tolist(),
        "vertices": [[vc[0], vc[1], vc[2]] for vc in vertices_camera],
    }

    # Generate filenames with zero-padded counter
    json_filename = f"{iteration_counter:08d}.json"
    json_output_file = Path(output_dir) / json_filename

    # Save JSON file
    with open(json_output_file, mode='w') as json_file:
        json.dump(json_data, json_file, separators=(', ', ': '), indent=None)

    print(f"Data saved to: {json_output_file}")
   




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
    

     # Extract coordinates at positions 3, 6, 9, 12, and 15 (1-based index)
    selected_indices = [3, 6, 9, 12, 15]
    extracted_coordinates = [all_coordinates[i] for i in selected_indices if i - 1 < len(all_coordinates)]
    

    create_spheres(extracted_coordinates)

    # Print the extracted coordinates
    print(f"Extracted Coordinates at positions {selected_indices}: {extracted_coordinates}")
    
    return all_coordinates


def create_spheres(coordinates):
    spheres = []
    for coord in coordinates:
        sphere = bproc.object.create_primitive('SPHERE', scale=[0.004, 0.004, 0.004])
        sphere.enable_rigidbody(active=True, collision_shape='SPHERE')
        offset_coord = [coord[0]-0.012, coord[1], coord[2]]  # Offset to place on mesh
        sphere.set_location(offset_coord)
        print("created Sphere")
        # Create a unique material for the sphere
        mat_marker = bproc.material.create(f"MarkerMaterial_{coord}")
        grey_col = 1.0

        mat_marker.set_principled_shader_value("Subsurface", 0.2)
        mat_marker.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
        mat_marker.set_principled_shader_value("Roughness", np.random.uniform(0, 0.5))
        mat_marker.set_principled_shader_value("Specular", np.random.uniform(0.3, 1.0))
        mat_marker.set_principled_shader_value("Metallic", np.random.uniform(0, 0))
        
        sphere.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        sphere.hide(False)
        sphere.add_material(mat_marker)

        spheres.append(sphere)


def render_and_save(output_dir):
    """
    Renders the current scene and saves the output to an HDF5 file.
    """
    global iteration_counter  # Access the global iteration_counter

    # Generate filename with zero-padded counter
    hdf5_filename = f"{iteration_counter:08d}.hdf5"
    hdf5_output_file = Path(output_dir) 

    # Render and save
    data = bproc.renderer.render()
    bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=True)

    print(f"HDF5 saved to: {hdf5_output_file}")


   

def main():

    global iteration_counter
    # Paths
    base_path = 'C:\\Users\\fabia\\Desktop\\HybridHands\\output\\poses\\mano'
    output_dir = "output/"

    # Initialize BlenderProc
    bproc.init()

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all .obj and .xyz files sorted by their index
    obj_files = sorted(glob.glob(os.path.join(base_path, '*.obj')), key=lambda x: int(os.path.basename(x).split('.')[0]))
    xyz_files = sorted(glob.glob(os.path.join(base_path, '*.xyz')), key=lambda x: int(os.path.basename(x).split('.')[0]))

    # Check if the number of .obj files and .xyz files match
    if len(obj_files) != len(xyz_files):
        print("Mismatch between the number of .obj and .xyz files. Exiting.")
        return

    # Reset the scene
  

    # Process each .obj and .xyz pair
    for obj_file, xyz_file in zip(obj_files, xyz_files):
        print(f"Processing: {obj_file} and {xyz_file}")

        # Extract 3D coordinates
        all_coordinates = extract_all_coordinates(xyz_file)
        world_coords = np.array(all_coordinates)

        # Set up material and load hand mesh
        material = setup_material()
        objs = load_and_prepare_hand_mesh(obj_file, material)

        # Configure camera, lights, save JSON, and render
        configure_camera_and_lights(objs, world_coords, output_dir)
        render_and_save(output_dir)  # No need to pass iteration_counter

        # Increment the counter after saving both files
        iteration_counter += 1

        bproc.utility.reset_keyframes()
        clear_scene()

    print("Processing completed!")

if __name__ == "__main__":
    main()
