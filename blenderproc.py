import blenderproc as bproc
from pathlib import Path
import random
import argparse
import bpy
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('output_dir', nargs='?', default="./output/render", help="Path to where the final files, will be saved")
args = parser.parse_args()

bproc.init()

# load the objects into the scene
objs = bproc.loader.load_obj("./output/rand_0_skin.obj")




for o in objs:
    o.set_scale([0.1, 0.1, 0.1])
    o.set_shading_mode('smooth')



# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([15, -15, 15])
light.set_energy(10000)

# Find point of interest, all cam poses should look towards it
poi = bproc.object.compute_poi(objs)

# Sample five camera poses
for i in range(5):
    # Sample random camera location above objects
    location = np.random.uniform([-20, -20, 18], [20, 20, 20])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)
# Find all materials
materials = bproc.material.collect_all()


# Collect all jpg images in the specified directory

for mat in materials:
    # Load one random image
    normal = bpy.data.images.load(r"C:\\Users\\fabia\Desktop\\NIMBLE_model\\output\\rand_0_normal.png")
    spec = bpy.data.images.load(r"C:\\Users\\fabia\Desktop\\NIMBLE_model\\output\\rand_0_spec.png")
    dif = bpy.data.images.load(r"C:\\Users\\fabia\Desktop\\NIMBLE_model\\output\\rand_0_diffuse.png")
    # Set it as base color of the current material
    mat.set_principled_shader_value("Base Color", dif)
    mat.set_principled_shader_value("Normal", normal)
    
    #mat.set_principled_shader_value("Diffuse", dif)
    

# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)