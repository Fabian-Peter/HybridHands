import os
import numpy as np
from typing import List, Callable
import blenderproc as bproc


class SceneObjects:
    """
    A class to manage scene objects in BlenderProc.

    This class provides functionalities for randomly selecting IDs of objects, sampling
    their respective 6D poses, simulating physics, and saving/loading scene object data. The class handles both 
    distractor and learnable objects in a 3D scene.
    """

    def __init__(self):
        self.local2world_mat = []

        self.distractor_ids = None

        self.obj_ids = None
        
    def sample_ids_distractor(self, n_distractors: int, list_distractors: List[int]) -> None:
        """
        Selects a specified number of unique distractor IDs from a given list.

        This function randomly selects 'n_distractors' unique IDs from 'list_distractors'.
        The selected IDs are stored in 'self.distractor_ids'. It ensures that the number 
        of IDs to select does not exceed the length of 'list_distractors' and that the 
        number of distractors requested is non-negative.

        Parameters:
        n_distractors (int): The number of unique distractor IDs to be selected.
        list_distractors (List[int]): A list of available distractor IDs to select from.

        Returns:
        None: The function updates 'self.distractor_ids' with the selected IDs.

        Raises:
        ValueError: If 'list_distractors' contains fewer elements than 'n_distractors'.
        ValueError: If 'n_distractors' is negative.
        """

        self.distractor_ids = []

        # Check if list_distractors has enough elements
        if len(list_distractors) < n_distractors:
            raise ValueError("list_distractors must have at least as many elements as n_distractors")
        
         # Check if n_distractors is non-negative
        if n_distractors < 0:
            raise ValueError("Number of distractor objects must be non-negative")
        
        # Select unique distractor IDs
        self.distractor_ids = np.random.choice(list_distractors,n_distractors,replace=False)
  

    def sample_ids_learnable(self, n_learnable: int, list_learnable: List[int]) -> None:
        """
        Selects a specified number of unique learnable object IDs from a given list.

        This function randomly selects 'n_learnable' unique IDs from 'list_learnable'.
        The selected IDs are stored in 'self.obj_ids'. It ensures that the number 
        of IDs to select does not exceed the length of 'list_learnable' and that the 
        number of learnable objects requested is at least one.

        Parameters:
        n_learnable (int): The number of unique learnable object IDs to be selected.
        list_learnable (List[int]): A list of available learnable object IDs to select from.

        Returns:
        None: The function updates 'self.obj_ids' with the selected IDs.

        Raises:
        ValueError: If 'list_learnable' contains fewer elements than 'n_learnable'.
        ValueError: If 'n_learnable' is less than 1.
        """
        self.obj_ids = []

        # Check if list_learnable has enough elements
        if len(list_learnable) < n_learnable:
            raise ValueError("list_learnable must have at least as many elements as n_learnable")
        
        # Check if n_learnable is at least 1
        if n_learnable < 1:
            raise ValueError("Number of learnable objects must be at least one")
        
        # Select unique learnable object IDs
        self.obj_ids = np.random.choice(list_learnable, size = n_learnable, replace=False)


    def sample_scene_objects(self, objects_to_sample: list[bproc.types.MeshObject], pose_sampler: Callable[[bproc.types.MeshObject], None], 
                             physics_enabled: bool = True) -> None:
        """
        Samples poses for scene objects and optionally simulates physics.

        This function takes a list of objects, samples their poses using a provided
        pose sampling function, and optionally runs a physics simulation. The local to world 
        transformation matrices of the sampled objects are stored after pose sampling and 
        physics simulation.

        Parameters:
        objects_to_sample (list): A list of objects (typically mesh objects) whose poses are to be sampled.
        pose_sampler (Callable[[bproc.types.MeshObject], None]): A function that takes a mesh object as input and 
            samples a pose for it. This function is expected to modify the object's pose directly.
        physics_enabled (bool, optional): A flag to enable or disable physics simulation after pose sampling. 
            Defaults to True.

        Return:
        None: After pose sampling (and optional physics simulation), retrieves and stores the local to 
        world transformation matrix for each object in `self.local2world_mat`.
        """

        # Sample poses for each object
        bproc.object.sample_poses(objects_to_sample = objects_to_sample, sample_pose_func = pose_sampler, max_tries = 1000)

        # Simulate physics and fix final poses if physics is enabled
        if physics_enabled:
            bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                        max_simulation_time=10,
                                                        check_object_interval=1,
                                                        substeps_per_frame = 20,
                                                        solver_iters=25)
        
        # Store the local to world transformation matrices for each object
        for obj in objects_to_sample:
            mat = obj.get_local2world_mat()
            self.local2world_mat.append(mat)


    def sample_one_scene_object(self, objects_to_sample: list[bproc.types.MeshObject], 
                                pose_sampler: Callable[[bproc.types.MeshObject], None]) -> None:
        """
        Sample poses for scene objects without physics and places them in an existing scene.

        This function takes a list of objects and samples their poses using a provided pose sampling function.
        The local to world transformation matrices of the sampled objects are stored after pose sampling.
        This function is primarly used to create floating objects in a scene where some objects are already placed with
        a physics simulation.

        Parameters:
        objects_to_sample (list): A list of objects (typically mesh objects) whose poses are to be sampled.
        pose_sampler (Callable[[bproc.types.MeshObject], None]): A function that takes a mesh object as input and 
            samples a pose for it. This function is expected to modify the object's pose directly.

        Return:
        None: After pose sampling, retrieves and stores the local to 
        world transformation matrix for each object in `self.local2world_mat`.

        """

        # Sample poses for each object
        bproc.object.sample_poses(objects_to_sample = objects_to_sample, sample_pose_func = pose_sampler, max_tries = 1000)

        # Store the local to world transformation matrices for each object
        for obj in objects_to_sample:
            mat = obj.get_local2world_mat()
            self.local2world_mat.append(mat)


    def save_scene_objects(self, dir: str, scene_counter: int) -> None:
        """
        Saves the scene object data to a file.

        This function saves the local to world transformation matrices, distractor IDs,
        and object IDs of the scene objects to a .npz file. The file is named using
        a scene counter to distinguish between different scenes.

        Parameters:
        dir (str): The directory path where the scene object data file will be saved.
        scene_counter (int): A counter or identifier for the current scene, used in naming the saved file.

        Raises:
        ValueError: If the total number of objects (distractor + learnable) does not match the
        number of local2world matrices available.
        """

        if (len(self.distractor_ids) + len(self.obj_ids)) != len(self.local2world_mat):
            raise ValueError("Number of objects (distractor + learnable) must be equal to local2world matrices.")
        
        filename = os.path.join(dir, 'scene_objects_{:06d}.npz'.format(scene_counter))
        np.savez(filename, local2world_mat = self.local2world_mat, distractor_ids = self.distractor_ids, obj_ids = self.obj_ids)


    def load_scene_objects(self, dir: str, scene_counter: int) -> None:
        """
        Loads the scene object data from a file.

        This function reads the .npz file containing the local to world transformation matrices,
        distractor IDs, and object IDs of the scene objects, and updates the respective attributes
        of the instance.

        Parameters:
        dir (str): The directory path from where the scene object data file will be loaded.
        scene_counter (int): The counter or identifier for the specific scene whose data is to be loaded.
        """
        
        filename = os.path.join(dir, 'scene_objects_{:06d}.npz'.format(scene_counter))
        data = np.load(filename)
        self.local2world_mat = data['local2world_mat']
        self.distractor_ids = data['distractor_ids']
        self.obj_ids = data['obj_ids']


    def set_scene_objects(self, objects_to_sample: list[bproc.types.MeshObject]):
        """
        Sets the local to world transformation matrices for the given mesh objects.

        This function updates the local to world transformation matrices of each object
        in 'objects_to_sample' with the corresponding matrix stored in 'self.local2world_mat'.
        It checks to ensure the number of objects matches the number of matrices available.

        Parameters:
        objects_to_sample (List[bproc.types.MeshObject]): A list of mesh objects whose 
        transformation matrices are to be updated.

        Raises:
        ValueError: If the number of objects in 'objects_to_sample' does not match the number
        of matrices in 'self.local2world_mat'.
        """
        
        if len(objects_to_sample) != len(self.local2world_mat):
            raise ValueError("Number of objects and local2world matrices must be equal.")

        for i, obj in enumerate(objects_to_sample):
            obj.set_local2world_mat(self.local2world_mat[i])


    def move_scene_object_trajectory(self, object_to_move, trajectory):
        pass

