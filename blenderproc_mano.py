import blenderproc as bproc
from pathlib import Path
import random
import argparse
import bpy
import numpy as np
import csv
import os
import sys
#costum package import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

base_path = 'C:\\Users\\fabia\\Desktop\\HybridHands\\output\\poses\\mano'

def load_scene_objects(number):

    """Load 3D objects into the scene and set their properties."""
    objs_file = os.path.join(base_path, f"{number}.obj")
    objs = bproc.loader.load_obj(obj_file)
    for obj in objs:
        obj.enable_rigidbody(active=True, collision_shape='MESH')
        obj.set_shading_mode("smooth")
        obj.set_rotation_euler([0, 0, 0])  # Optional: set initial object orientation
    return objs


def main():
    """Main function to run the entire pipeline."""
    args = parse_arguments()
    bproc.init()

    
    num_iterations = 10  # Number of files to process

    for index in range(num_iterations):
        # Clear previous objects before loading new scene
        load_scene_objects(number)
        clear_scene()


if __name__ == "__main__":
    main()
