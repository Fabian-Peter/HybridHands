import blenderproc as bproc
import numpy as np
import os

# Vars
number = "0000"
base_path = 'C:\\Users\\fabia\\Desktop\\HybridHands\\output\\poses\\mano'
file_path = os.path.join(base_path, f"{number}.obj")

# Initialize BlenderProc
bproc.init()

# Create a material
material = bproc.material.create('Hand_Material')
material.set_principled_shader_value("Base Color", [1.0, 0.8, 0.6, 1.0])  # Skin color
material.set_principled_shader_value("Roughness", 0.8)
material.set_principled_shader_value("Specular", 0.5)

# Load the hand mesh
objs = bproc.loader.load_obj(file_path)
hand_mesh = objs[0]  # Assuming the hand mesh is the first object loaded

# Add a material slot if none exist
if len(hand_mesh.blender_obj.data.materials) == 0:
    hand_mesh.blender_obj.data.materials.append(None)  # Create an empty material slot

# Assign the material to the first material slot (index 0)
hand_mesh.set_material(0, material)

# Create a point light
light = bproc.types.Light()
light.set_location([2, -2, 0])
light.set_energy(300)

# Compute point of interest (POI)
poi = bproc.object.compute_poi(objs)

# Sample camera poses
for i in range(5):
    location = np.random.uniform([-0.4, -0.4, 0.4], [0.4, 0.4, 0.4])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

# Render the scene
data = bproc.renderer.render()

# Write the rendering to an HDF5 file
bproc.writer.write_hdf5("output/", data)
