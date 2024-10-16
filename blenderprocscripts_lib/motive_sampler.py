from. motive_loader import MotiveLoader
import numpy as np
import pandas as pd

class MotiveSampler:
   def __init__(self, trajectory: MotiveLoader, name: str):
      self.trajectory = trajectory
      self.name = name
      self.sampled_trajectory = None


   def scale_trajectory(self, pos):
      return pos * 0.001
   

   def get_T_and_R(self, i):
      t = self.sampled_trajectory[i][0][:3] - self.sampled_trajectory[0][0][:3]
      r = self.sampled_trajectory[i][0][3:]
      return t, r 

   def sample_tracetory(self, min, max, sequence_length, t_spacing):
      """
      min max is used to ignore faulty beginning and end frames
      """
      counter = 0
      trajectories = []
      starting_frame = np.random.randint(min,max)
      max_tries = 5
      for i in range(sequence_length):
         frame = starting_frame + i * t_spacing
         data_frame = self.trajectory.get_data_frame(self.name, frame)

         
         has_nan = data_frame.isna().any().any()
         counter = 0
         while(has_nan and counter < max_tries):

            frame += 1
            counter += 1
            data_frame = self.trajectory.get_data_frame(self.name, frame)
            has_nan = data_frame.isna().any().any()
         
         #break and resample
         if counter == max_tries:
            break
         else:
            trajectories.append(data_frame["RigidBody"].to_numpy())


      if counter == max_tries:
         trajectories = self.sample_tracetory(min, max, sequence_length, t_spacing)

      self.sampled_trajectory = trajectories
      return trajectories
         

