import blenderproc as bproc
import argparse
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
from blenderprocscripts_lib.scene_creation import SceneParameters
from blenderprocscripts_lib.scene_hand import SceneHand
from blenderprocscripts_lib.motive_loader import MotiveLoader
from blenderprocscripts_lib.motive_sampler import MotiveSampler
import numpy as np
import json

# Define a function that samples 6-DoF poses
def _sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    location = np.random.uniform(min, max)
    #location = [0.2,0.2,0.4]
    obj.set_location(location)
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


# Define a function that samples 6-DoF poses
def _sample_pose_floating_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.2], [-0.2, -0.2, 0.25])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    location = np.random.uniform(min, max)
    #location = [0.2,0.2,0.4]
    obj.set_location(location)
    obj.set_rotation_euler(bproc.sampler.uniformSO3())



def set_pose_target(obj: bproc.types.MeshObject, trajectory):
    pass
    


parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved ")
parser.add_argument('hand_model_path', help="The path to the output of the MANO and NIMBLE Pose Editor")
parser.add_argument('--num_scenes', type=int, default=5, help="How many scenes with 10 images each to generate")
parser.add_argument('--recreate', type=int, default=0, help="Recreate Scene")
parser.add_argument('--start_scene', type=int, default=0, help="Start Scene")
parser.add_argument('--motion_file', type=str, help="Path to file with trajectories")
parser.add_argument('--extrude_arm', type=int, default = 0, help="If the wrist face of mano model should be extruded to resemble an arm")
parser.add_argument('--hand_model', type=int, default = 0, help="Decide for either MANO or NIMBLE hands (Mano==0, no hand==2)")

args = parser.parse_args()
bproc.init()

#get trajectory data
motive_data = MotiveLoader.from_motive(args.motion_file)
motive_name = motive_data.get_motive_names()
motive_sampler = MotiveSampler(motive_data, motive_name[0])



# load bop objects into the scene
target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'medical'), model_type = 'cad', mm2m = True)

# load distractor bop objects
ycbv_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'), mm2m = True)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'), split='train')

# set shading and hide objects
for obj in (target_bop_objs + ycbv_dist_bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)
    
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)


light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

light_point = bproc.types.Light()
light_point.set_energy(100)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)


#use one file per scene to store information
dir = os.path.join(args.output_dir, 'bop_data', 'medical', 'scenes')
os.makedirs(dir, exist_ok=True)


#list of properties for ycbv set. if 0 for corresponding id then use default value. otherwise sample base color specular and metallic
mat_props = list(range(1, 22))
n_trajectories = 12

# a scene has a specific number of objects, textures, ...
# a scene has a fixed position of cameras
# sample cameras once per scene
# object might move
for j in range(args.num_scenes):

    trajectory = motive_sampler.sample_tracetory(101, 15000, n_trajectories, 12)

    index = j + args.start_scene

    sampled_parameters = SceneParameters()

    if args.recreate == 0:
        sampled_parameters.load_scene_parameters(dir, index)

    if args.recreate == 1:
        sampled_parameters.scene_objects.sample_ids_learnable(1, [0])
        sampled_parameters.scene_objects.sample_ids_distractor(3, [1,3,4,5,7,8,9,10,11,12,14,16,17,18])

    sampled_target_bop_objs = [target_bop_objs[i] for i in sampled_parameters.scene_objects.obj_ids]
    sampled_distractor_bop_objs = [ycbv_dist_bop_objs[i] for i in sampled_parameters.scene_objects.distractor_ids]

    if args.recreate == 1:
        sampled_parameters.materials.sample_materials(len(sampled_target_bop_objs) + len(sampled_distractor_bop_objs), len(cc_textures))
        sampled_parameters.lighting.sample_lighting()

        # sampled_parameters.scene_objects.sample_scene_objects(sampled_distractor_bop_objs, _sample_pose_func, True)
        # sampled_parameters.scene_objects.sample_one_scene_object(sampled_target_bop_objs, _sample_pose_floating_func)
        #sampled_parameters.scene_objects.sample_scene_objects(sampled_target_bop_objs + sampled_distractor_bop_objs, _sample_pose_func, True)
        #sampled_parameters.scene_objects.sample_one_scene_object(sampled_target_bop_objs, _sample_pose_floating_func)


    # Set materials
    for i, obj in enumerate(sampled_target_bop_objs + sampled_distractor_bop_objs):        
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['medical']: # and mat_props[render_id] != 0:
            mat.set_principled_shader_value("Base Color", [sampled_parameters.materials.grey_col, sampled_parameters.materials.grey_col, 
                                                           sampled_parameters.materials.grey_col, 1])        
        mat.set_principled_shader_value("Roughness", sampled_parameters.materials.roughness[i])
        if obj.get_cp("bop_dataset_name") == 'medical': #and mat_props[render_id] != 0:
            mat.set_principled_shader_value("Specular", sampled_parameters.materials.specular)
            mat.set_principled_shader_value("Metallic", sampled_parameters.materials.metallic)
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)

    if args.recreate == 1:
        hand_parameters = SceneHand(args.hand_model,args.hand_model_path,args.extrude_arm)
        hand_parameters.sample_scene_hand(4,5)
        hand = hand_parameters.set_hand(sampled_target_bop_objs[0])
        #hand = bproc.loader.load_obj(hand_path)[0]
        #hand_parameters.set_pose(sampled_target_bop_objs[0], hand)
        sampled_parameters.scene_objects.sample_scene_objects(sampled_distractor_bop_objs, _sample_pose_func, True)
        sampled_parameters.scene_objects.sample_one_scene_object(sampled_target_bop_objs, _sample_pose_floating_func)
        

    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=sampled_parameters.lighting.plane_emission_strength,
                                    emission_color=sampled_parameters.lighting.plane_emission_color)  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(sampled_parameters.lighting.point_color)
    location = sampled_parameters.lighting.point_location
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    #random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(cc_textures[sampled_parameters.materials.cc_textures])


    if args.recreate == 0:
        #use same order as object sampling above
        sampled_parameters.scene_objects.set_scene_objects(sampled_distractor_bop_objs + sampled_target_bop_objs)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs + sampled_distractor_bop_objs + [hand])

    n_cameras = 8
    cam_poses = 0
    
    if args.recreate == 1:
        sampled_parameters.cameras.set_n_cameras(n_cameras)


    #create first camera setup
    #reload for all other scenes
    while cam_poses < n_cameras:

        if args.recreate == 1:
            sampled_parameters.cameras.sample_camera(cam_poses, sampled_target_bop_objs)
       
        # Sample location
        location = sampled_parameters.cameras.location[cam_poses]
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = sampled_parameters.cameras.poi[cam_poses]
        #poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=1, replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=sampled_parameters.cameras.inplane_rotation[cam_poses])
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        #print("before if check")
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    #print("after cam poses")
    if args.recreate == 1:
        sampled_parameters.save_scene_parameters(dir, index)

    # render the whole pipeline
    base_position = sampled_target_bop_objs[0].get_location() 
    for k in range(n_trajectories):
        t, r = motive_sampler.get_T_and_R(k)
        t = motive_sampler.scale_trajectory(t)
        sampled_target_bop_objs[0].set_location(base_position + t)
        sampled_target_bop_objs[0].set_rotation_euler(r)
        data = bproc.renderer.render()

        # Write data in bop format
        bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                            target_objects = sampled_target_bop_objs,
                            dataset = 'medical',
                            depth_scale = 0.1,
                            depths = data["depth"],
                            colors = data["colors"], 
                            color_file_format = "JPEG",
                            ignore_dist_thres = 10,
                            calc_mask_info_coco = False,
                            frames_per_chunk=n_cameras*n_trajectories)
    
    hand.delete()
    #hand = None
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):    
        obj.set_location([0, 0, 0])
        obj.set_rotation_euler([0, 0, 0])      
        obj.disable_rigidbody()
        obj.hide(True)






