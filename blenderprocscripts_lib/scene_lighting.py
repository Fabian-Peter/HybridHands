import numpy as np
import blenderproc as bproc
import os

class SceneLighting:

    def __init__(self):
        self.plane_emission_strength = None
        self.plane_emission_color = None
        self.point_color = None
        self.point_location = None


    def sample_lighting(self):
        self.plane_emission_strength = np.random.uniform(3,6)
        self.plane_emission_color = np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
        self.point_color = np.random.uniform([0.5,0.5,0.5],[1,1,1])
        self.point_location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
        

    def save_lighting(self, dir, scene_counter):
        filename = os.path.join(dir, 'lighting_{:06d}.npz'.format(scene_counter))
        np.savez(filename, plane_emission_strength = self.plane_emission_strength, plane_emission_color = self.plane_emission_color, point_color = self.point_color, point_location = self.point_location)
        
    
    def load_lighting(self, dir, scene_counter):
        filename = os.path.join(dir, 'lighting_{:06d}.npz'.format(scene_counter))
        data = np.load(filename)
        self.plane_emission_strength = data['plane_emission_strength']
        self.plane_emission_color = data['plane_emission_color']
        self.point_color = data['point_color']
        self.point_location = data['point_location']