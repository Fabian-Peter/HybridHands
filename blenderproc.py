import blenderproc as bproc
from pathlib import Path
import random
import argparse
import bpy
import numpy as np
import csv

# VARIABLES
SPHERE_RADIUS = 4.0
LIGHT_ENERGY = 500000
LIGHT_RADIUS = 100
FIXED_LIGHT_DISTANCE = 100.0
CAMERA_ROTATION_RANGE = (-0.7854, 0.7854)  # +/- 45 degrees in radians
ROOM_PLANE_SCALE = [-644.637, -644.637, -644.637]

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_dir', nargs='?', default="./output/render",
        help="Path to where the final files will be saved"
    )
    return parser.parse_args() 

def load_scene_objects():
    """Load 3D objects into the scene and set their properties."""
    objs = bproc.loader.load_obj("./output/rand_0_skin.obj")
    for obj in objs:
        obj.enable_rigidbody(active=True, collision_shape='MESH')
        obj.set_shading_mode("smooth")
        obj.set_rotation_euler([0, 0, 0])  # Optional: set initial object orientation
    return objs

def load_materials():
    """Loads and applies textures to the materials in the scene."""
    materials = bproc.material.collect_all()
    for mat in materials:
        normal = bpy.data.images.load(r"C:\\Users\\fabia\\Desktop\\HybridHands\\output\\rand_0_normal.png")
        spec = bpy.data.images.load(r"C:\\Users\\fabia\\Desktop\\HybridHands\\output\\rand_0_spec.png")
        dif = bpy.data.images.load(r"C:\\Users\\fabia\\Desktop\\HybridHands\\output\\rand_0_diffuse.png")
        mat.set_principled_shader_value("Base Color", dif)
        mat.set_principled_shader_value("Normal", normal)
        mat.set_principled_shader_value("Specular", spec)

def extract_all_coordinates(filepath):
    """
    Extracts all 3D coordinates from the given XYZ file.

    Args:
        filepath: The path to the XYZ file.

    Returns:
        A list of all 3D coordinates in the file.
    """
    all_coordinates = []
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
        coords_list = [float(coord) for line in lines for coord in line.split()]
        
        # Group into (x, y, z) tuples
        for i in range(0, len(coords_list), 3):
            x, y, z = coords_list[i:i + 3]
            all_coordinates.append([x, y, z])
    
    return all_coordinates

def extract_coordinates(filepath, indices):
    """
    Extracts coordinates from the given XYZ file based on the specified indices.
    
    Args:
        filepath: The path to the XYZ file.
        indices: A list of indices to extract coordinates for.
    
    Returns:
        A list of extracted 3D coordinates.
    """
    all_coordinates = []
    extracted_coordinates = []

    with open(filepath, 'r') as file:
        lines = file.readlines()
        coords_list = [float(coord) for line in lines for coord in line.split()]
        
        # Group into (x, y, z) tuples
        for i in range(0, len(coords_list), 3):
            x, y, z = coords_list[i:i + 3]
            all_coordinates.append([x, y, z])
    
    # Extract coordinates at specified indices
    extracted_coordinates = [all_coordinates[idx - 1] for idx in indices]  # 1-based to 0-based index
    
    return extracted_coordinates

def create_spheres(coordinates):
    """Creates spheres at the specified coordinates and applies unique materials to each."""
    spheres = []
    for coord in coordinates:
        sphere = bproc.object.create_primitive('SPHERE', scale=[SPHERE_RADIUS] * 3)
        sphere.enable_rigidbody(active=True, collision_shape='SPHERE')
        offset_coord = [coord[0], coord[1], coord[2] + 4.5]  # Offset to place on mesh
        sphere.set_location(offset_coord)

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
    
    return spheres

def create_room():
    """Creates the room"""
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=ROOM_PLANE_SCALE, location=[0, -300, 0], rotation=[90, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=ROOM_PLANE_SCALE, location=[0, 300, 0], rotation=[90, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=ROOM_PLANE_SCALE, location=[0, 0, -300], rotation=[0, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=ROOM_PLANE_SCALE, location=[0, 0, 300], rotation=[0, 0, 0])
    ]

    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

def configure_camera_and_lights(objs, world_coords, output_dir):
    """Sets up camera and lights at various positions and projects 3D coordinates into 2D image space"""
    camera_poses = []
    light_positions = []
    # Amount of rendered images with visible and invisible markers
    for i in range(5):
        location = np.random.uniform([-200, -200, 180], [200, 200, 200])
        poi = bproc.object.compute_poi(objs)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(*CAMERA_ROTATION_RANGE))

        # Random light position in spherical coordinates
        light_position = calculate_random_light_position()
        create_light(light_position)

        camera_poses.append((location, rotation_matrix))
        light_positions.append(light_position)

        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        
        # Pass output_dir to project_and_save_coordinates
        project_and_save_coordinates(world_coords, cam2world_matrix, i, output_dir)

    return camera_poses, light_positions

def calculate_random_light_position():
    """Calculates a random light position based on spherical coordinates."""
    given_location = np.array([8.21917, -37.8768, -57.4157])
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)

    random_x = FIXED_LIGHT_DISTANCE * np.sin(phi) * np.cos(theta)
    random_y = FIXED_LIGHT_DISTANCE * np.sin(phi) * np.sin(theta)
    random_z = FIXED_LIGHT_DISTANCE * np.cos(phi)

    return given_location + np.array([random_x, random_y, random_z])

def create_light(position):
    """Creates a point light at the specified position."""
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location(position)
    light.set_radius(LIGHT_RADIUS)
    light.set_energy(LIGHT_ENERGY)

def project_and_save_coordinates(world_coords, cam2world_matrix, index, output_dir):
    """Projects 3D coordinates to 2D image space and saves them along with Z-depth."""
    image_coords = bproc.camera.project_points(world_coords)
    world_coords_homogeneous = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))
    camera_coords = world_coords_homogeneous @ np.linalg.inv(cam2world_matrix).T
    z_depths = camera_coords[:, 2]

    # Use the passed output_dir instead of args
    csv_output_file_without = Path(output_dir + "/without_spheres") / f"image_{index}_coords.csv"
    with open(csv_output_file_without, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Image_X', 'Image_Y', 'Camera_Z'])
        for wc, ic, z in zip(world_coords, image_coords, z_depths):
            writer.writerow([ic[0], ic[1], z])
    csv_output_file_with = Path(output_dir + "/with_spheres") / f"image_{index}_coords.csv"
    with open(csv_output_file_with, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Image_X', 'Image_Y', 'Camera_Z'])
        for wc, ic, z in zip(world_coords, image_coords, z_depths):
            writer.writerow([ic[0], ic[1], z])

def render_images(spheres, output_dir):
    """Renders images with and without spheres and saves them to /with and /without"""
    data_with_spheres = bproc.renderer.render()
    bproc.writer.write_hdf5(output_dir + "/with_spheres", data_with_spheres)

    for sphere in spheres:
        sphere.hide(True)

    data_without_spheres = bproc.renderer.render()
    bproc.writer.write_hdf5(output_dir + "/without_spheres", data_without_spheres)

def main():
    """Main function to run the entire pipeline."""
    args = parse_arguments()
    bproc.init()

    objs = load_scene_objects()
    load_materials()
    
    # TODO replace with relative path
    xyz_file_path = 'C:\\Users\\fabia\\Desktop\\HybridHands\\output\\rand_0_joints.xyz'
    
    # Extraction of xyz joint coordinates
    all_coordinates = extract_all_coordinates(xyz_file_path)
    
    # Fingertip Coordinate extraction
    indices_to_extract = [5, 10, 15, 20, 25]
    extracted_coordinates = extract_coordinates(xyz_file_path, indices_to_extract)

    # Create markers (spheres) at the specific coordinates
    spheres = create_spheres(extracted_coordinates)
    create_room()

    # Convert all coordinates to NumPy array for further processing
    world_coords = np.array(all_coordinates)

    # Randomly generate camera and light positions
    configure_camera_and_lights(objs, world_coords, args.output_dir)

    # Render images with and without the spheres
    render_images(spheres, args.output_dir)

if __name__ == "__main__":
    main()