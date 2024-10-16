import numpy as np
from typing import Optional
import blenderproc as bproc
import os
import bpy
import bmesh
import math

import mathutils
from mathutils import Matrix
from mathutils import Vector


class SceneHand:

    def __init__(self, hand_model: int, hand_model_path: str, extrude: int) -> None:
        self.hand_set = None
        self.hand_alt = None
        self.hand_model = hand_model
        self.hand_path = None
        self.pose_path = None
        self.hand_model_path = hand_model_path
        self.extrude = extrude

        self.manoColors = [[0.666, 0.461, 0.266, 1],
              [0.599, 0.393, 0.246, 1],
              [0.456, 0.275, 0.159, 1],
              [0.337, 0.141, 0.048, 1],
              [0.112, 0.033, 0.019, 1],]


    def close_mano_gap(self, hand_mesh_object: bproc.types.MeshObject, extrude_arm: bool) -> None:
        hand_mesh_object.edit_mode()
        hand = hand_mesh_object.mesh_as_bmesh()
        hand.edges.ensure_lookup_table()

        edge_i = [2254, 1273, 1953, 932, 306, 2051, 1876, 1875, 1901, 2257, 1362, 2261, 1622, 1621, 1976, 2167]
        edges = [hand.edges[edge_i[i]] for i in range(len(edge_i))]
        bmesh.ops.edgeloop_fill(hand, edges=edges)
        
        if extrude_arm:
            hand.faces.ensure_lookup_table()
            direction = hand.faces[1538].normal * 25
            newGeom = bmesh.ops.extrude_face_region( hand, geom = [hand.faces[1538]], use_select_history = True )
            verts = [e for e in newGeom['geom'] if isinstance(e, bmesh.types.BMVert)]
            bmesh.ops.translate( hand, vec = direction, verts = verts )

        hand_mesh_object.update_from_bmesh(hand)
        hand_mesh_object.object_mode()
    

    def sample_scene_hand(self, n_available_sets: int, n_hand_alts: int) -> None:
        self.hand_set = np.random.randint(n_available_sets)
        self.hand_alt = np.random.randint(n_hand_alts)


    def save_scene_hand(self, dir: str, scene_counter: int) -> None:
        filename = os.path.join(dir, 'scene_hand_{:06d}.npz'.format(scene_counter))
        np.savez(filename, hand_set = self.hand_set, hand_alt = self.hand_alt)   
   

    def load_scene_hand(self, dir: str, scene_counter: int) -> None:
        filename = os.path.join(dir, 'scene_objects_{:06d}.npz'.format(scene_counter))
        data = np.load(filename)
        self.hand_set = data['hand_set']
        self.hand_alt = data['hand_alt']

    def set_hand(self, current_bop_obj: bproc.types.MeshObject) -> Optional[bproc.types.MeshObject]:
        if self.hand_model == 0:
            #hand_path = self.hand_model_path + "/ManoViewer/IKSolveResult/" + current_bop_obj.get_name() + f"-{self.hand_set}/"
            self.hand_path = self.hand_model_path + current_bop_obj.get_name() + f"-{self.hand_set}/"
            if not os.path.exists(self.hand_path + f"manoHand_{self.hand_alt}.stl"):
                print("No MANO hand and pose found for " + current_bop_obj.get_name())
                return None

            self.pose_path = self.hand_path + f"manoHand_{self.hand_alt}.pose"
            hand = bproc.loader.load_obj(self.hand_path + f"manoHand_{self.hand_alt}.stl")[0]
            
            # Close the Mano Hands open wrist with blender utility
            extrude = True if self.extrude == 1 else 0
            self.close_mano_gap(hand, extrude)
            #return self.hand_path + f"manoHand_{self.hand_alt}.stl"


        elif self.hand_model == 1:
            self.hand_path = self.hand_model_path + "/NimbleViewer/IKSolveResult/" + current_bop_obj.get_name() + f"-{self.hand_set}/"
            if not os.path.exists(self.hand_path + f"rand_{self.hand_alt}_skin.obj"):
                print("No NIMBLE hand and pose found for " + current_bop_obj.get_name())
                return None
            self.pose_path = self.hand_path + f"nimbleHand_{self.hand_alt}.pose"
            hand = bproc.loader.load_obj(self.hand_path + f"rand_{self.hand_alt}_skin.obj")[0]
            #return self.hand_path + f"rand_{self.hand_alt}_skin.obj"
        else:
            ValueError("Unsupported hand model: " + str(self.hand_model))


        # Load and apply the relative hand pose and scaling factor
        with open(self.pose_path, "r") as f:
            scale = 1.0/200 # We need this factor because the objects are scaled in the Pose Editor but not here
            lines = f.readlines()
            loc = [float(x)*scale for x in lines[0][1:-2].split(", ")]
            rot = [math.radians(float(x)) for x in lines[1][1:-2].split(", ")]
            hand.set_location(loc)
            rot = mathutils.Euler((rot[1], rot[2], rot[0]), 'YXZ').to_matrix().to_euler('XYZ')
            hand.set_rotation_euler(rot) 
            scaleAdjust = 0.22 if self.hand_model == 1 else 1    # This needs to happen because Nimble and Mano have different scales. Doesnt affect translation because it is already applied in the pose editor.
            hand.set_scale([scale * scaleAdjust, scale * scaleAdjust, scale * scaleAdjust])
            hand.set_parent(current_bop_obj)


        # Setting color and shading for the hand
        mesh = hand.get_mesh()
        for f in mesh.polygons:
            f.use_smooth = True
        # Set Color if its a mano Hand that has no texture
        if self.hand_model == 0:
            mat = hand.get_materials()[0]
            mat.set_principled_shader_value("Base Color", self.manoColors[self.hand_alt])
            mat.set_principled_shader_value("Roughness", 1.0)  
            mat.set_principled_shader_value("Specular", 0.2)
        
        return hand








