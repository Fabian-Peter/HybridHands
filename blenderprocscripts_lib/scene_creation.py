import numpy as np
import blenderproc as bproc
import os
from typing import List, Callable
from blenderprocscripts_lib.scene_objects import SceneObjects
from blenderprocscripts_lib.scene_materials import SceneMaterials
from blenderprocscripts_lib.scene_lighting import SceneLighting
from blenderprocscripts_lib.scene_cameras import SceneCameras
from blenderprocscripts_lib.scene_hand import SceneHand

class SceneParameters:

    def __init__(self) -> None:
        self.materials = SceneMaterials()
        self.lighting = SceneLighting()
        self.cameras = SceneCameras()
        self.scene_objects = SceneObjects()


    def save_scene_parameters(self, output_dir, scene_counter):

        self.materials.save_materials(output_dir, scene_counter)
        self.lighting.save_lighting(output_dir, scene_counter)
        self.scene_objects.save_scene_objects(output_dir, scene_counter)
        self.cameras.save_camera(output_dir, scene_counter)


    def load_scene_parameters(self, input_dir, scene_counter):
            
        self.materials.load_materials(input_dir, scene_counter)
        self.lighting.load_lighting(input_dir, scene_counter)
        self.scene_objects.load_scene_objects(input_dir, scene_counter)
        self.cameras.load_camera(input_dir, scene_counter)








# Define a function that samples 6-DoF poses
def _sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    location = np.random.uniform(min, max)
    #location = [0.2,0.2,0.4]
    obj.set_location(location)
    obj.set_rotation_euler(bproc.sampler.uniformSO3())