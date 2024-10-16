import numpy as np
import blenderproc as bproc
import os
from pathlib import Path  

class SceneMaterials:

    def __init__(self):
        self.grey_col = None
        self.roughness = None
        self.specular = None
        self.metallic = None
        self.cc_textures = None


    def sample_materials(self, n_objs, n_cc_textures):
        self.grey_col = np.random.uniform(0.1, 0.9)
        self.roughness = []
        for i in range(n_objs):
            self.roughness.append(np.random.uniform(0, 0.5))
        self.specular = np.random.uniform(0.3, 1.0)
        self.metallic = np.random.uniform(0, 0.5)
        self.cc_textures = np.random.randint(0,n_cc_textures-1) #np.random.choice(cc_textures)


    def save_materials(self, dir, scene_counter):
    
        filename = os.path.join(dir, 'materials_{:06d}.npz'.format(scene_counter))
        np.savez(filename, grey_col = self.grey_col, roughness = self.roughness, specular = self.specular, metallic = self.metallic, cc_textures = self.cc_textures)

    
    def load_materials(self, dir, scene_counter):
        filename = os.path.join(dir, 'materials_{:06d}.npz'.format(scene_counter))
        data = np.load(filename)
        self.grey_col = data['grey_col']
        self.roughness = data['roughness']
        self.specular = data['specular']
        self.metallic = data['metallic']
        self.cc_textures = data['cc_textures']
