import numpy as np
import torch
from smplx import MANO
import trimesh
import os

# Paths
mano_model_path = "./assets/MANO_RIGHT.pkl"  # Update with your MANO model path
output_path = "./output/poses/mano"  # Update with desired output path
os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

# MANO Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mano_layer = MANO(model_path=mano_model_path, use_pca=False, is_rhand=True).to(device)

# Parameters
num_samples = 10  # Number of samples to generate
pose_std = 0.2  # Standard deviation for random noise in pose
shape_std = 0.03  # Standard deviation for random noise in shape

# Generate Multiple Samples
for i in range(num_samples):
    # Random pose and shape parameters
    pose_mean = torch.zeros((1, 45), device=device)  # Neutral pose
    random_pose = pose_mean + pose_std * torch.randn((1, 45), device=device)
    random_pose = torch.clamp(random_pose, -1.0, 1.0)  # Clamp pose values

    shape_mean = torch.zeros((1, 10), device=device)  # Neutral shape
    random_shape = shape_mean + shape_std * torch.randn((1, 10), device=device)

    # Add global orientation (neutral orientation in this case)
    global_orient = torch.zeros((1, 3), device=device)  # Neutral global orientation

    # Generate Hand Mesh and Joints
    output = mano_layer(betas=random_shape, global_orient=global_orient, hand_pose=random_pose)
    vertices = output.vertices[0].detach().cpu().numpy()  # Mesh vertices
    joints = output.joints[0].detach().cpu().numpy()  # Joint positions

    # File names
    obj_filename = os.path.join(output_path, f"{i}.obj")
    xyz_filename = os.path.join(output_path, f"{i}.xyz")

    # Export .obj file
    mesh = trimesh.Trimesh(vertices, mano_layer.faces)
    mesh.export(obj_filename)

    # Export joint coordinates to .xyz
    np.savetxt(xyz_filename, joints, fmt="%.6f")

    print(f"Saved sample {i:04d} to {obj_filename} and {xyz_filename}")

print("All samples generated and saved!")