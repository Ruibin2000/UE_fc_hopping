# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, ITURadioMaterial, SceneObject


import numpy as np

import drjit as dr
import mitsuba as mi
import gym

class UE2:
    def __init__(self, scene, id, material, scaling=1):


        self.scene = scene
        self.scaling = scaling
        self.scene = scene
        self.id = id

        self.center_pos = np.array([0,0,0]) * self.scaling

        # new patch pattern
        self.rx1_loc_pos = np.array([-0.036,0.075,0]) * self.scaling
        self.rx1_loc_orientation = np.array([np.pi,0,0])

        # new patch pattern
        self.rx2_loc_pos = np.array([0.036,-0.075,0]) * self.scaling
        self.rx2_loc_orientation = np.array([0,0,0])

        # dipole
        self.rx3_loc_pos = np.array([0.036,0,0]) * self.scaling
        self.rx3_loc_orientation = np.array([0,0,0])

        # dipole
        self.rx4_loc_pos = np.array([-0.036,0,0]) * self.scaling
        self.rx4_loc_orientation = np.array([0,0,0])


        self.ue_material = material

        # Create a tr38901 12G receiver
        self.rx1 = Receiver(name="rx-1",
                    position=self.center_pos + self.rx1_loc_pos,
                    orientation=self.rx1_loc_orientation,
                    display_radius=2)
        self.rx2 = Receiver(name="rx-2",
                    position=self.center_pos + self.rx2_loc_pos,
                    orientation=self.rx2_loc_orientation,
                    display_radius=2)
        
        # Create a dipole 6G receiver
        self.rx3 = Receiver(name="rx-3",
                    position=self.center_pos + self.rx3_loc_pos,
                    orientation=self.rx3_loc_orientation,
                    display_radius=2)
        
        self.rx4 = Receiver(name="rx-4",
                    position=self.center_pos + self.rx4_loc_pos,
                    orientation=self.rx4_loc_orientation,
                    display_radius=2)


        # Create a new object with the given parameters
        self.body = SceneObject(fname='UE2.ply',
                               name=f"ue-{self.id}",
                               radio_material=self.ue_material)
        
        self.scene.edit(add = self.body)
        self.body.scaling = self.scaling

        self.scene.remove("rx-1")
        self.scene.remove("rx-2")
        self.scene.remove("rx-3")
        self.scene.remove("rx-4")
        self.scene.add(self.rx1)
        self.scene.add(self.rx2)
        self.scene.add(self.rx3)
        self.scene.add(self.rx4)      
        
    def reset(self, displacement):

        # Reset the position and orientation of the object
        self.set_location(displacement)
        

        self.body.orientation = np.array([0, 0, 0])

        self.rx1.orientation = self.rx1_loc_orientation
        self.rx2.orientation = self.rx2_loc_orientation
        self.rx3.orientation = self.rx3_loc_orientation
        self.rx4.orientation = self.rx4_loc_orientation
        
    def set_location(self,displacement):
        # Update the position of the object
        # print(self.body.position + displacement)
        self.body.position = displacement
        self.center_pos = displacement

        self.rx1.position = self.rx1_loc_pos + self.center_pos
        self.rx2.position = self.rx2_loc_pos + self.center_pos

        self.rx3.position = self.rx3_loc_pos + self.center_pos
        self.rx4.position = self.rx4_loc_pos + self.center_pos

    def R_matrix (self, yaw, pitch, roll):
        # Rotation about X (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])

        # Rotation about Y (pitch)
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Rotation about Z (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation: ZYX (yaw → pitch → roll)
        R = Rz @ Ry @ Rx

        return R


    def set_orientation(self, alpha, beta, gamma):

        self.set_location(self.center_pos)


        self.body.orientation = np.array([alpha, beta, gamma])
        self.rx1.orientation = self.rx1_loc_orientation + np.array([alpha, beta, gamma])
        self.rx2.orientation = self.rx2_loc_orientation + np.array([alpha, beta, gamma])
        self.rx3.orientation = self.rx3_loc_orientation + np.array([alpha, beta, gamma])
        self.rx4.orientation = self.rx4_loc_orientation + np.array([alpha, beta, gamma])
        R = self.R_matrix(alpha, beta, gamma)
        

        self.rx1.position = R @ self.rx1_loc_pos + self.center_pos
        self.rx2.position = R @ self.rx2_loc_pos + self.center_pos
        self.rx3.position = R @ self.rx3_loc_pos + self.center_pos
        self.rx4.position = R @ self.rx4_loc_pos + self.center_pos
        
            
            
            
    # def delete(self):
    #     # Remove receivers
    #     for rx in ["rx-1", "rx-2", "rx-3", "rx-4"]:
    #         self.scene.remove(rx)

    #     # # Print material usage before deletion
    #     # for name, obj in self.scene.objects.items():
    #     #     print(f'{name:<15}{obj.radio_material.name}')

    #     # Remove UE object
    #     self.body.radio_material = "itu_concrete"  # Break the material reference
    #     self.scene.edit(remove=f"ue-{self.id}")  # Now safe to remove the object

    #     # Dereference material from any remaining objects
    #     for name, obj in self.scene.objects.items():
    #         if obj.radio_material.name == "UE-material":
    #             obj.radio_material = None  # or assign a default material if required

    #     # Now remove the material
    #     self.scene.remove("UE-material")

    #     # Final check
    #     print("#------------------------#")
    #     for name, obj in self.scene.objects.items():
    #         print(f'{name:<15}{obj.radio_material.name if obj.radio_material else "None"}')
            
    #     print("successfully remove the ue")
    #     print("#------------------------#")


    
