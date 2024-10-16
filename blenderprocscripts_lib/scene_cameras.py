import numpy as np
import blenderproc as bproc
import os

class SceneCameras:

    def __init__(self):
        self.location = None
        self.inplane_rotation = None
        self.poi = None

    
    def set_n_cameras(self, n_cameras):
        self.location = np.zeros((n_cameras, 3))
        self.inplane_rotation = np.zeros((n_cameras, 1))
        self.poi = np.zeros((n_cameras, 3))


    def sample_camera(self, camera_id, obj):

        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min =1.45,
                                radius_max = 1.9,
                                elevation_min = 5,
                                elevation_max = 89)
            
        self.location[camera_id] = location
        self.poi[camera_id] = bproc.object.compute_poi(np.random.choice(obj, size=1, replace=False))

        inplane_rotation = np.random.uniform(-3.14159, 3.14159)
        self.inplane_rotation[camera_id] = inplane_rotation


    def save_camera(self, dir, scene_counter):
        filename = os.path.join(dir, 'camera_{:06d}.npz'.format(scene_counter))
        np.savez(filename, location = self.location, inplane_rotation = self.inplane_rotation, poi = self.poi)


    def load_camera(self, dir, scene_counter):
        filename = os.path.join(dir, 'camera_{:06d}.npz'.format(scene_counter))
        data = np.load(filename)
        self.location = data['location']
        self.inplane_rotation = data['inplane_rotation']
        self.poi = data['poi']
