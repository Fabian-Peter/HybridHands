import blenderproc as bproc
from pathlib import Path
import random
import argparse
import bpy
import numpy as np
import csv

parser = argparse.ArgumentParser()
parser.add_argument('output_dir', nargs='?', default="./output/render", help="Path to where the final files will be saved")
args = parser.parse_args()

bproc.init()

# Load the objects into the scene
objs = bproc.loader.load_obj("./output/rand_0_skin.obj")
for obj in objs:
    obj.set_rotation_euler([0, 0, 0])  # [x_rotation, y_rotation, z_rotation] in radians


# List of 3D world coordinates to project
world_coords = np.array([
    [35.135318756103516, -41.1285400390625, -91.99156951904297],
    [24.913267135620117, -63.41144561767578, -57.26049041748047],
    [11.574165344238281, -80.37953186035156, -36.30665969848633],
    [6.444902420043945, -93.48470306396484, -24.419471740722656],
    [27.86547088623047, -27.864673614501953, -85.5152816772461],
    [24.114883422851562, -36.10301971435547, -23.729984283447266],
    [26.697017669677734, -48.845001220703125, 14.80356216430664],
    [24.575424194335938, -60.651668548583984, 34.277671813964844],
    [24.594829559326172, -67.48316955566406, 47.30222702026367],
    [17.341859817504883, -22.846187591552734, -81.578125],
    [2.576883316040039, -37.09212112426758, -25.716793060302734],
    [-3.536139488220215, -56.58207702636719, 13.385337829589844],
    [-6.74940299987793, -77.57178497314453, 28.83423614501953],
    [-3.9952917098999023, -88.01788330078125, 40.851314544677734],
    [7.52364444732666, -22.17104721069336, -79.59907531738281],
    [-17.23605728149414, -39.549800872802734, -36.69884490966797],
    [-31.43714141845703, -69.73556518554688, -12.684276580810547],
    [-37.547325134277344, -92.28812408447266, -3.346729278564453],
    [-41.684776306152344, -108.95523071289062, -3.3334970474243164],
    [-2.402355194091797, -25.037633895874023, -81.02289581298828],
    [-27.361608505249023, -51.23991394042969, -49.83961868286133],
    [-33.08411407470703, -83.12610626220703, -43.53733825683594],
    [-31.530231475830078, -101.22029876708984, -41.35429382324219],
    [-30.807308197021484, -115.76215362548828, -39.91801071166992]
])

# Find all materials
materials = bproc.material.collect_all()

# Collect all jpg images in the specified directory
for mat in materials:
    # Load one random image
    normal = bpy.data.images.load(r"C:\\Users\\fabia\\Desktop\\NIMBLE_model\\output\\rand_0_normal.png")
    spec = bpy.data.images.load(r"C:\\Users\\fabia\\Desktop\\NIMBLE_model\\output\\rand_0_spec.png")
    dif = bpy.data.images.load(r"C:\\Users\\fabia\\Desktop\\NIMBLE_model\\output\\rand_0_diffuse.png")
    # Set it as base color of the current material
    mat.set_principled_shader_value("Base Color", dif)
    mat.set_principled_shader_value("Normal", normal)

# Loop through each camera pose
for i in range(5):
    # Sample random camera location above objects
    location = np.random.uniform([-200, -200, 180], [200, 200, 200])
    # Compute rotation based on vector going from location towards poi
    poi = bproc.object.compute_poi(objs)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location and rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)
    
    

    # Render the scene
    data = bproc.renderer.render()
    # Write the rendered data to a .hdf5 container
    bproc.writer.write_hdf5(args.output_dir, data)
    # Project 3D world coordinates (in this frame) into 2D image coordinates
    image_coords = bproc.camera.project_points(world_coords)

    # Prepare a file to save the results as CSV for this specific image
    csv_output_file = Path(args.output_dir) / f"image_{i}_coords.csv"

    # Open the CSV file for writing
    with open(csv_output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Save the 3D world coordinates and their corresponding 2D image coordinates to the CSV
        for wc, ic in zip(world_coords, image_coords):
            writer.writerow([ ic[0], ic[1]])


