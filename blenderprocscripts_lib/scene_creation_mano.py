import os
from pathlib import Path
import blenderproc as bproc
import bpy

class SceneParameters:
    def __init__(self, index, base_path):
        self.index = index
        self.base_path = base_path
        
        # Paths for the specific iteration
        self.obj_file = Path(f"{base_path}/rand_{index}_skin.obj")
        self.normal_map = Path(f"{base_path}/rand_{index}_normal.png")
        self.spec_map = Path(f"{base_path}/rand_{index}_spec.png")
        self.diffuse_map = Path(f"{base_path}/rand_{index}_diffuse.png")
        self.joints_file = Path(f"{base_path}/rand_{index}_joints.xyz")
        
        # Store the loaded objects and coordinates
        self.objects = []
        self.coordinates = []
        
        # Camera poses for this scene
        self.camera_poses = []

    def load_scene_objects(self):
        """Load the .obj file and return the objects."""
        self.objects = bproc.loader.load_obj(str(self.obj_file))
        for obj in self.objects:
            obj.enable_rigidbody(active=True, collision_shape='MESH')
            obj.set_shading_mode("smooth")
            obj.set_rotation_euler([0, 0, 0])
        return self.objects

    def load_materials(self):
        """Load and apply textures from the corresponding maps."""
        materials = bproc.material.collect_all()
        for mat in materials:
            normal = bpy.data.images.load(str(self.normal_map))
            spec = bpy.data.images.load(str(self.spec_map))
            dif = bpy.data.images.load(str(self.diffuse_map))
            mat.set_principled_shader_value("Base Color", dif)
            mat.set_principled_shader_value("Normal", normal)
            mat.set_principled_shader_value("Specular", spec)

    def extract_all_coordinates(self, filepath):
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

    def extract_coordinates(self):
        """Extract coordinates from the specified joints file."""
        self.coordinates = self.extract_all_coordinates(str(self.joints_file))
        return self.coordinates

    def add_camera_pose(self, cam_pose):
        """Add a camera pose for the scene."""
        self.camera_poses.append(cam_pose)
