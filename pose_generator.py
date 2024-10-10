import os
import torch
import numpy as np
import pickle
from NIMBLELayer import NIMBLELayer
from utils import batch_to_tensor_device, save_textured_nimble, smooth_mesh
import pytorch3d
import pytorch3d.io
from pytorch3d.structures.meshes import Meshes

if __name__ == "__main__":
    device = torch.device('cpu')  # Ensure everything is loaded on the CPU

    pm_dict_name = r"assets/NIMBLE_DICT_9137.pkl"
    tex_dict_name = r"assets/NIMBLE_TEX_DICT.pkl"

    if os.path.exists(pm_dict_name):
        pm_dict = np.load(pm_dict_name, allow_pickle=True)  # Load directly with torch
        pm_dict = batch_to_tensor_device(pm_dict, device)  # Ensure it's on the CPU

    if os.path.exists(tex_dict_name):
        tex_dict = np.load(tex_dict_name, allow_pickle=True)  # Load directly with torch
        tex_dict = batch_to_tensor_device(tex_dict, device)  # Ensure it's on the CPU

    if os.path.exists(r"assets/NIMBLE_MANO_VREG.pkl"):
        nimble_mano_vreg = np.load(r"assets/NIMBLE_MANO_VREG.pkl", allow_pickle=True)  # Load directly with torch
        nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)  # Ensure it's on the CPU
    else:
        nimble_mano_vreg = None

    nlayer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=True, pose_ncomp=30, shape_ncomp=20, nimble_mano_vreg=nimble_mano_vreg)

    num_poses = 10  # Define how many different random poses you want to generate
    batch_size = 1  # Number of samples per batch, can be increased for parallel processing

    output_folder = "output/poses"
    os.makedirs(output_folder, exist_ok=True)

    # Generate and save multiple random poses
    for pose_index in range(num_poses):
        pose_param = torch.rand(batch_size, 30) * 2 - 1  # Random pose parameters
        shape_param = torch.rand(batch_size, 20) * 2 - 1  # Random shape parameters
        tex_param = torch.rand(batch_size, 10) - 0.5  # Random texture parameters

        # Forward pass through NIMBLE layer to get vertices and textures
        skin_v, muscle_v, bone_v, bone_joints, tex_img = nlayer.forward(pose_param, shape_param, tex_param, handle_collision=True)

        # Convert to PyTorch3D meshes
        skin_p3dmesh = Meshes(skin_v, nlayer.skin_f.repeat(batch_size, 1, 1))
        muscle_p3dmesh = Meshes(muscle_v, nlayer.muscle_f.repeat(batch_size, 1, 1))
        bone_p3dmesh = Meshes(bone_v, nlayer.bone_f.repeat(batch_size, 1, 1))

        # Smooth the meshes
        skin_p3dmesh = smooth_mesh(skin_p3dmesh)
        muscle_p3dmesh = smooth_mesh(muscle_p3dmesh)
        bone_p3dmesh = smooth_mesh(bone_p3dmesh)

        # Convert vertices to MANO format
        skin_mano_v = nlayer.nimble_to_mano(skin_v, is_surface=True)

        # Detach and move data to CPU for saving
        tex_img = tex_img.detach().cpu().numpy()
        skin_v_smooth = skin_p3dmesh.verts_padded().detach().cpu().numpy()
        bone_joints = bone_joints.detach().cpu().numpy()

        # Save all the generated files for each pose
        for i in range(batch_size):
            np.savetxt(f"{output_folder}/rand_{pose_index}_joints.xyz", bone_joints[i])
            np.savetxt(f"{output_folder}/rand_{pose_index}_manov.xyz", skin_mano_v[i])

            pytorch3d.io.IO().save_mesh(bone_p3dmesh[i], f"{output_folder}/rand_{pose_index}_bone.obj")
            pytorch3d.io.IO().save_mesh(muscle_p3dmesh[i], f"{output_folder}/rand_{pose_index}_muscle.obj")
            save_textured_nimble(f"{output_folder}/rand_{pose_index}.obj", skin_v_smooth[i], tex_img[i])

        print(f"Pose {pose_index} generated and saved.")
