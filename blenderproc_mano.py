import blenderproc as bproc
import numpy as np
import os
from pathlib import Path
import json
import bpy
import random
import json
import glob 
from mathutils import Vector
import tempfile
import shutil
import mathutils
import imageio

#CONFIG
CAMERA_ROTATION_RANGE = (-0.7854, 0.7854)  # Range for in-plane rotation
    
LIGHT_POSITIONS = [
    [0.383889, 0, 0],
    [0, 0.383889, 0],
    [0, 0, 0.383889],
    [-0.383889, 0, 0],
    [0, -0.383889, 0],
    [0, 0, -0.383889]
    ]
    
TOP_LIGHT_POSITION = [0, 0, 0.465]
BOTTOM_LIGHT_POSITION = [0, 0, -0.465]
ENERGY_SPOT = 7.5
ENERGY_AREA = 0.5
    
SKIN_TONES = [
    [0x42/255.0, 0x30/255.0, 0x2E/255.0, 1.0],
    [0x74/255.0, 0x57/255.0, 0x49/255.0, 1.0],
    [0x89/255.0, 0x65/255.0, 0x65/255.0, 1.0],
    [0xB0/255.0, 0x98/255.0, 0x84/255.0, 1.0],
    [0x89/255.0, 0x65/255.0, 0x5A/255.0, 1.0],
    [0xF3/255.0, 0xC3/255.0, 0xAD/255.0, 1.0],
    [0xE3/255.0, 0xA5/255.0, 0x8D/255.0, 1.0],
    [0xFF/255.0, 0xD7/255.0, 0xBC/255.0, 1.0],
    [0xE9/255.0, 0xB8/255.0, 0x93/255.0, 1.0]
    ]
    
TARGET_INDICES = [745, 320, 444, 555, 672]
SELECTED_INDICES = [3, 6, 9, 12, 15]
REORDER_MAPPING = [
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4,
    5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
]

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
iteration_counter = 0

POSE_PATH = 'C:\\Users\\fabia\\Desktop\\HybridHands\\output\\poses\\mano'

MARKER_OUTPUT_DIR = "output/myMarkerHAND/training/rgb/"
RGB_OUTPUT_DIR = "output/myRGBHAND/training/rgb/"

def configure_camera_and_lights(objs):
    """
    Sets up a single camera and light, projects coordinates, and saves the JSON file.
    """

    # Randomly select one position
    selected_position = random.choice(LIGHT_POSITIONS)

    # Create and place the light
    light = bproc.types.Light()
    light.set_location(selected_position)
    light.set_energy(ENERGY_SPOT)
    
    top_light = bproc.types.Light()
    top_light.set_type("AREA")
    top_light.set_location(TOP_LIGHT_POSITION)
    top_light.set_energy(ENERGY_AREA)

    bottom_light = bproc.types.Light()
    bottom_light.set_type("AREA")
    bottom_light.set_location(BOTTOM_LIGHT_POSITION)
    bottom_light.set_energy(ENERGY_AREA)

    # Random distance from the object
    radius = random.uniform(0.35, 0.65)
   
    theta = np.random.uniform(0, np.pi / 2) 
    phi = np.random.uniform(0, 2 * np.pi)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    location = np.array([x, y, z])

    # Compute viewport
    poi = bproc.object.compute_poi(objs)

    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        poi - location, 
        inplane_rot=np.random.uniform(*CAMERA_ROTATION_RANGE)
    )

    # Adding camera
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

    return cam2world_matrix

    


def clear_scene():
    """Deletes all objects from the current Blender scene."""
    all_objects = bpy.data.objects
    for obj in all_objects:
        if obj.type != 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)



def setup_material():
    """Sets up a material with specific properties for the hand mesh."""
    random_skin_tone = random.choice(SKIN_TONES)

    # Create and set up the material
    material = bproc.material.create('Hand_Material')
    material.set_principled_shader_value("Base Color", random_skin_tone)
    material.set_principled_shader_value("Roughness", 0.8)
    material.set_principled_shader_value("Specular", 0.5)
    return material

def load_and_prepare_hand_mesh(file_path_obj, material):
    """Loads a hand mesh from an OBJ file and applies the given material."""
    objs = bproc.loader.load_obj(file_path_obj)
    for obj in objs:
        obj.enable_rigidbody(active=True, collision_shape='MESH')
        #Somehow Object initially loaded with 90° rotation, reset
        obj.set_rotation_euler([0, 0, 0])

        # Update the world matrix after setting the rotation
        bpy_obj = obj.blender_obj  
        bpy.context.view_layer.update() 
        bpy_obj.matrix_world = bpy_obj.matrix_local  

    hand_mesh = objs[0] 
    if len(hand_mesh.blender_obj.data.materials) == 0:
        hand_mesh.blender_obj.data.materials.append(None) 
    hand_mesh.set_material(0, material)

    return objs

def generate_freihand_k_matrix():
    """Generate K matrix with FreiHAND-style variations"""
    base_focal = np.random.uniform(300, 600)
    return np.array([
        [base_focal * np.random.uniform(0.98, 1.02), 0, 112],
        [0, base_focal * np.random.uniform(0.98, 1.02), 112],
        [0, 0, 1]
    ])

def is_vertex_visible(camera, vertex, obj, threshold=0.2):
    """
    Checks if a given vertex is visible from the camera by ray casting.

    Args:
        camera (bpy.types.Object): The camera object.
        vertex (mathutils.Vector): The 3D location of the vertex in world coordinates.
        obj (bpy.types.Object): The object to test against.
        threshold (float): Maximum allowable distance between the hit point and the vertex.

    Returns:
        bool: True if the vertex is visible, False otherwise.
    """
    scene = bpy.context.scene
    # Get the depsgraph from the current context.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Starting point: the camera location
    origin = camera.location
    # Direction from the camera to the vertex (normalized)
    direction = (vertex - origin).normalized()
    
    # Perform a ray cast from the camera in the given direction.
    hit, location, normal, index, hit_obj, matrix = scene.ray_cast(depsgraph, origin, direction)
    
    if not hit:
        # No hit means nothing was intersected; treat as not visible.
        return False
    if hit_obj != obj:
        # If the ray hits a different object, the vertex is occluded.
        return False
    # Check if the hit location is close enough to the vertex.
    if (location - vertex).length < threshold:
        return True
    else:
        return False

def project_and_save_coordinates(world_coords, cam2world_matrix, obj):
    """
    Saves JSON data including the intrinsic matrix, all vertices, original coordinates from .xyz,
    and additional coordinates from target_indices in camera space.
    Parameters:
        world_coords (np.ndarray): 3D world coordinates of the keypoints.
        cam2world_matrix (np.ndarray): 4x4 camera-to-world transformation matrix.
        output_dir (str): Directory where the JSON file will be saved.
        obj (bproc.object.Object): The BlenderProc object whose vertices will be extracted.
    """
    global iteration_counter
    
    obj_name = obj.get_name()
    bpy_obj = bpy.data.objects.get(obj_name)
    if bpy_obj is None:
        raise ValueError(f"Object '{obj_name}' not found in Blender context.")

    # Get world2cam matrix (inverse of cam2world)
    world2cam_matrix = np.linalg.inv(cam2world_matrix)
    
    # Extract and transform vertices
    vertices_world = np.array([bpy_obj.matrix_world @ v.co for v in bpy_obj.data.vertices])
    vertices_homogeneous = np.hstack((vertices_world, np.ones((vertices_world.shape[0], 1))))
    vertices_camera = (vertices_homogeneous @ world2cam_matrix.T)[:, :3]

    # Project points to 2D
    image_coords = bproc.camera.project_points(world_coords)
    vertices_coords = bproc.camera.project_points(vertices_world)
    
    # Transform keypoints to camera space
    world_coords_homo = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))
    xyz_camera = (world_coords_homo @ world2cam_matrix.T)[:, :3]
    
    #Add fingertips (target indices) to xyz and uv as 17 - 21 keypoints
    extracted_xyz_world = vertices_world[TARGET_INDICES]


    extracted_xyz_homo = np.hstack((extracted_xyz_world, np.ones((len(extracted_xyz_world), 1))))
    extracted_xyz_camera = (extracted_xyz_homo @ world2cam_matrix.T)[:, :3]
    extracted_uv = [vertices_coords[idx].tolist() for idx in TARGET_INDICES]

    intrinsic_matrix = generate_freihand_k_matrix()
    bproc.camera.set_intrinsics_from_K_matrix(
        intrinsic_matrix, 
        IMAGE_WIDTH, 
        IMAGE_HEIGHT
    )

    data = bproc.renderer.render()
    if isinstance(data["depth"], list):
        if len(data["depth"]) == 1:
            depth_map = np.array(data["depth"][0])
        else:
            # If multiple depth maps are returned, stack them along a new axis.
            depth_map = np.stack(data["depth"], axis=0)
    else:
        depth_map = data["depth"]

    # Compute visibility for the markers (fingertips)
    camera = bpy.data.objects['Camera']
    marker_vertices = [mathutils.Vector(v) for v in extracted_xyz_world]
    marker_visibility = []
    for vertex in marker_vertices:  # marker_vertices should be a list of mathutils.Vector objects.
        visible = is_vertex_visible(camera, vertex, bpy_obj, threshold=0.01)
        marker_visibility.append(1 if visible else 0)

    # ===== Compute and normalize the distance for each marker =====
    # Get the camera world location (as a NumPy array)
    cam_loc = np.array(camera.location)
    # Compute the Euclidean distance from each marker (in world space) to the camera
    marker_distances = np.linalg.norm(extracted_xyz_world - cam_loc, axis=1)
    # Normalize: 1 for the closest marker and 0 for the furthest marker in this image.
    d_min = marker_distances.min()
    d_max = marker_distances.max()
    if d_max - d_min < 1e-6:
        normalized_distances = [1.0 for _ in marker_distances]  # All markers are at nearly the same distance.
    else:
        normalized_distances = ((d_max - marker_distances) / (d_max - d_min)).tolist()
    # ========================================================================
    
    # Prepare JSON data with explicit type conversion
    json_data = {
        "uv": [[float(ic[0]), float(ic[1])] for ic in image_coords] + extracted_uv,
        "xyz": [[float(coord[0]), float(coord[1]), float(coord[2])] for coord in xyz_camera] +
               [[float(coord[0]), float(coord[1]), float(coord[2])] for coord in extracted_xyz_camera],
        "hand_type": 1,
        "marker_visibility": marker_visibility,
        "distance": normalized_distances,
        "K": intrinsic_matrix.tolist(),
        "vertices": [[float(vc[0]), float(vc[1]), float(vc[2])] for vc in vertices_camera],
        "image_path": f"/training/rgb/{iteration_counter:08d}.jpg"
    }

    # Reorder the uv and xyz lists to fit mano annotation
    json_data["uv"] = [json_data["uv"][i] for i in REORDER_MAPPING]
    json_data["xyz"] = [json_data["xyz"][i] for i in REORDER_MAPPING]

    json_data["xyz"] = convert_blender_to_freihand(json_data["xyz"]).tolist()
    json_data["vertices"] = convert_blender_to_freihand(vertices_camera).tolist()

    # Save JSON file
    marker_output_dir = Path(MARKER_OUTPUT_DIR)
    rgb_output_dir = Path(RGB_OUTPUT_DIR)

    marker_output_dir.mkdir(parents=True, exist_ok=True)
    rgb_output_dir.mkdir(parents=True, exist_ok=True)

    json_filename = f"{iteration_counter:08d}.json"

    rgb_json_output_file = rgb_output_dir / json_filename
    with open(rgb_json_output_file, mode='w') as json_file:
        json.dump(json_data, json_file, separators=(', ', ': '), indent=None)
    
    print(f"Data saved to: {rgb_json_output_file}")
    #marker json output
    '''
    marker_json_output_file = marker_output_dir / json_filename
    with open(marker_json_output_file, mode='w') as json_file:
        json.dump(json_data, json_file, separators=(', ', ': '), indent=None)
    
    print(f"Data saved to: {marker_json_output_file}")
    '''
    return extracted_xyz_world

def convert_blender_to_freihand(xyz_blender):
    # Convert from Blender (X, Y, Z) to OpenCV (X, -Y, -Z)
    xyz_freihand = np.array(xyz_blender, dtype=np.float32)
    xyz_freihand[:, 1] *= -1  # Flip Y for OpenCV's downward Y-axis
    xyz_freihand[:, 2] *= -1  # Flip Z for OpenCV's forward direction
    return xyz_freihand

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


#Sphere Creation
def create_spheres(coordinates):
    print("begin sphere creation")
    spheres = []
    for coord in coordinates:
        sphere = bproc.object.create_primitive('SPHERE', scale=[0.004, 0.004, 0.004])
        sphere.enable_rigidbody(active=True, collision_shape='SPHERE')
        offset_coord = [coord[0]-0.004, coord[1], coord[2]]  # Offset to place on mesh
        sphere.set_location(offset_coord)
        # Create a unique material for the sphere
        mat_marker = bproc.material.create(f"MarkerMaterial_{coord}")
        grey_col = 1.0

        mat_marker.set_principled_shader_value("Subsurface", 0.2)
        mat_marker.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
        mat_marker.set_principled_shader_value("Roughness", np.random.uniform(0, 0.5))
        mat_marker.set_principled_shader_value("Specular", np.random.uniform(0.3, 1.0))
        mat_marker.set_principled_shader_value("Metallic", np.random.uniform(0, 0.5))
        
        sphere.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        sphere.hide(False)
        sphere.add_material(mat_marker)

        spheres.append(sphere)

    return spheres
        


def render_and_save(output_dir):
    global iteration_counter
    data = bproc.renderer.render()
    colors = data["colors"] 

    for i, color_img in enumerate(colors):
        file_path = os.path.join(output_dir, f"{iteration_counter:08d}.png")
        imageio.imwrite(file_path, color_img)
    print(f"PNG images saved to: {output_dir}")


def clear_temp_directory():
    """Removes directories starting with 'blender_proc_' in the system temporary directory."""
    temp_dir = Path(tempfile.gettempdir())
    # Find all directories with a name that starts with "blender_proc_"
    for temp_subdir in temp_dir.glob("blender_proc_*"):
        try:
            shutil.rmtree(temp_subdir)
            print(f"Removed temporary directory: {temp_subdir}")
        except Exception as e:
            print(f"Failed to remove {temp_subdir}: {e}")

def main():

    global iteration_counter
 
    bproc.init()
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.camera.set_resolution(IMAGE_HEIGHT, IMAGE_WIDTH)
   
    # Get all .obj and .xyz files sorted by their index
    obj_files = sorted(glob.glob(os.path.join(POSE_PATH, '*.obj')), key=lambda x: int(os.path.basename(x).split('.')[0]))
    xyz_files = sorted(glob.glob(os.path.join(POSE_PATH, '*.xyz')), key=lambda x: int(os.path.basename(x).split('.')[0]))

    # Check if the number of .obj files and .xyz files match
    if len(obj_files) != len(xyz_files):
        print("Mismatch between the number of .obj and .xyz files.")
        return

    # Process each .obj and .xyz pair
    for obj_file, xyz_file in zip(obj_files, xyz_files):
        print(f"Processing: {obj_file} and {xyz_file}")
         # Set up material and load hand mesh
        material = setup_material()
        objs = load_and_prepare_hand_mesh(obj_file, material)
        # Extract 3D coordinates
        all_coordinates = extract_all_coordinates(xyz_file)
        world_coords = np.array(all_coordinates)
        # Configure camera, lights, save JSON, and render
        matrix = configure_camera_and_lights(objs)

        sphere_locations = project_and_save_coordinates(world_coords, matrix, objs[0])

        render_and_save(RGB_OUTPUT_DIR)
        #marker generation and rendering
        '''
        create_spheres(sphere_locations)

        render_and_save(MARKER_OUTPUT_DIR)
        '''
        # Increment the counter after saving both files
        iteration_counter += 1

        bproc.utility.reset_keyframes()
        clear_scene()
        clear_temp_directory()

    print("Processing completed!")

if __name__ == "__main__":
    main()
