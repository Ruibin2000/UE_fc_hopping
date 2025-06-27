import os # Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 2 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

no_preview = True # Toggle to False to use the preview widget
                  # instead of rendering for scene visualization

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, ITURadioMaterial, SceneObject

import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi

import numpy as np
from tqdm import trange

# # Import Sionna utils from the wireless class
# try:
#     import sionnautils
# except ImportError as e:
#     # Install Sionna if package is not already installed
#     !pip install git+https://github.com/sdrangan/wirelesscomm.git
#     import sionnautils

import sionnautils
from scipy.spatial.transform import Rotation as R

from UE2_config import UE2

import mitsuba as mi
import drjit as dr
from sionna.rt import AntennaPattern, PlanarArray, register_antenna_pattern
from sionnautils.custom_scene import list_scenes, get_scene

import json



class Engine:
            
    ###############################################################################
    def __init__(self):
        # ------------------------------------------------------------------
        # properties
        self.pattern_list = ["my_pattern", "dipole"]
        self.frequency_list = [15e9,3.5e9]
        self.B_list = [200e6,100e6]
        self.Ptx_dBm = 51 # dBm
        self.NF_dB = 7
        self.N0_dBm_per_Hz = -174
        self.measure_time = 0.05    # 50 ms
        
        self.rho_max = 4.8
        self.alpha = 0.6
        
        self.rand_seed = 21
        np.random.seed(self.rand_seed)
        
        self.p_solver  = PathSolver()
        
        # ------------------------------------------------------------------
        # tx_loc =[50,-115,10]        # s1_1
        # tx_loc =[-50,45,10]       # s1_2
        tx_loc =[-300,0,10]       # s2
        look_at_loc = [-280, -100, 1]
        
        self.region_id_init = 0
        self.region_id = self.region_id_init
        
        self.ue_yaw_init = 0
        self.ue_yaw = self.ue_yaw_init
        self.ue_pitch_init = 0
        self.ue_pitch = self.ue_pitch_init
        self.ue_loc_init = [-125,35,1]   # start point
        self.ue_loc = self.ue_loc_init
        
        
        # ------------------------------------------------------------------
        # initialize data record
      
        # self.yaw_list = []
        # self.pitch_list = []
        
        # self.paths = None
        self.rx1_a_power = None
        self.rx1_a_power_lin = None
        self.rx2_a_power = None
        self.rx2_a_power_lin = None
        self.rx3_a_power = None
        self.rx3_a_power_lin = None
        self.rx4_a_power = None
        self.rx4_a_power_lin = None
        
        
        self.rx1_snr_dB = None
        self.rx1_R = None
        
        self.rx2_snr_dB = None
        self.rx2_R = None
        
        self.rx3_snr_dB = None
        self.rx3_R = None
        
        self.rx4_snr_dB = None
        self.rx4_R = None
        
        self.rx_Rate = [0,0,0,0]
        self.rx_hPower_lin = [0,0,0,0]
        
        
        # ------------------------------------------------------------------
        # register the customized antenna pattern
        # register_antenna_pattern("my_pattern", self.my_pattern_factory)
        self.desired_pol = "V"
        register_antenna_pattern("my_pattern", lambda: MyPattern(self.desired_pol))

        # ------------------------------------------------------------------
        # initialize the scene
        scenes = list_scenes()
        print(scenes)

        scene_path, map_data = get_scene('nyu_tandon')
        for k, v in map_data.items():
            print(f'{k}: {v}')

        self.scene = load_scene(scene_path,merge_shapes=True)

        floor = self.scene.get('ground')
        # print(f'Floor material: {floor.radio_material.name}')
        floor.radio_material = ITURadioMaterial("itu_concrete",
                                        "concrete",
                                        thickness=0.01,
                                        color=(0.5, 0.5, 0.5))

        self.scene.remove("itu_wet_ground")

        for name, obj in self.scene.objects.items():
            print(f'{name:<15}{obj.radio_material.name}')
        # print(self.scene.radio_materials)
        
        # ------------------------------------------------------------------
        # initialize the UE
        ue_material = ITURadioMaterial(f"UE-material-{1}",
                                        "metal",
                                        thickness=0.1,
                                        color=(0.8, 0, 0))

        self.ue1 = UE2(scene = self.scene, id= 1, material=ue_material, scaling=1)
        self.ue1.reset(self.ue_loc)
        
        # ------------------------------------------------------------------
        # set the tx
        self.scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")
        
        # Create transmitter
        tx = Transmitter(name="tx",
                        position=tx_loc,
                        display_radius=2)

        # Add transmitter instance to scene
        self.scene.remove("tx")
        self.scene.add(tx)
        
        tx.look_at(look_at_loc)
        
    def reset(self):

        # ------------------------------------------------------------------
        # ue location
        # self.corner_loc = np.array([[-125,35,1], [35,35,1],[35,-100,1],[-125,-100,1]])
        # self.v_list = np.array([[10,0,0], [0,-10,0],[-10,0,0], [0,10,0]])
        self.region_id = self.region_id_init
        self.ue_yaw = self.ue_yaw_init
        self.ue_pitch = self.ue_pitch_init
        self.ue_loc = self.ue_loc_init
            
        # ------------------------------------------------------------------
        # initialize data record
        self.rx1_a_power = None
        self.rx1_a_power_lin = None
        self.rx2_a_power = None
        self.rx2_a_power_lin = None
        self.rx3_a_power = None
        self.rx3_a_power_lin = None
        self.rx4_a_power = None
        self.rx4_a_power_lin = None
        
            
        self.rx1_snr_dB = None
        self.rx1_R = None
        
        self.rx2_snr_dB = None
        self.rx2_R = None
           
        self.rx3_snr_dB = None
        self.rx3_R = None
            
        self.rx4_snr_dB = None
        self.rx4_R = None
            
        self.rx_Rate = [0,0,0,0]
        self.rx_hPower_lin = [0,0,0,0]
        
        self.ue1.reset(self.ue_loc)
            
        np.random.seed(self.rand_seed)
            
    
    ###############################################################################
    
    
    def run_from_file(self, routes_file):
        
        with open(routes_file, 'r') as f:
            loaded_routes = json.load(f)
        
        rate_15 = []
        rate_35 = []
        
        rate_total = []
        hp_total = []
        
        orientation_total = []
        
        N_measure = len(loaded_routes)

        self.reset()
        for i in trange(N_measure):
        # for i in trange(500):

            self.run_with_routes([1,1,1,1], loaded_routes[i])
            rate_total.append(self.rx_Rate)
            hp_total.append(self.rx_hPower_lin)
            orientation_total.append(self.ue_yaw)
            
        return rate_total, hp_total, orientation_total
    
    def run_with_routes(self, trigger_table, ue_loc):
        
        
        # self.update_loc_const_10mps(self.measure_time)
        self.ue_loc = ue_loc
        self.update_orientation(self.measure_time)
        self.run_RT()
        self.compute_Rate()
        return  np.max(np.array(self.rx_Rate) * np.array(trigger_table))

    
    def run_RT(self):
        for idx in range(len(self.frequency_list)):


            self.scene.frequency = self.frequency_list[idx]
            # Configure antenna array for all receivers
            if idx == 0:
                self.scene.rx_array = PlanarArray(num_rows=1,
                                        num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern=self.pattern_list[idx])
            else:
                self.scene.rx_array = PlanarArray(num_rows=1,
                                        num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern=self.pattern_list[idx],
                                        polarization="V")
            
            self.ue1.set_location(self.ue_loc)
            self.ue1.set_orientation(self.ue_yaw, self.ue_pitch, 0)
            
            paths = self.p_solver(scene=self.scene,
                        max_depth=5,
                        los=True,
                        specular_reflection=True,
                        diffuse_reflection=True,
                        refraction=True,
                        synthetic_array=False,
                        seed=41)
            
            a, tau = paths.cir(normalize_delays=False, out_type="numpy")
            
            if idx == 0:
                self.rx1_a_power_lin = float(np.sum(np.abs(a[0])**2))
                self.rx1_a_power = self.lin2db(self.rx1_a_power_lin)
                
                self.rx2_a_power_lin = float(np.sum(np.abs(a[1])**2))
                self.rx2_a_power = self.lin2db(self.rx2_a_power_lin)
            else:
                self.rx3_a_power_lin = float(np.sum(np.abs(a[2])**2))
                self.rx3_a_power = self.lin2db(self.rx3_a_power_lin)
                
                self.rx4_a_power_lin = float(np.sum(np.abs(a[3])**2))
                self.rx4_a_power = self.lin2db(self.rx4_a_power_lin)
                
            # self.paths = paths
            
            del paths
            
            # print(self.rx1_a_power_list)
            # print(self.rx2_a_power_list)
            # print(self.rx3_a_power_list)
            # print(self.rx4_a_power_list)

    def compute_Rate(self):
        # rx1_a_power_lin=  self.db2lin(np.array(self.rx1_a_power))
        # rx2_a_power_lin=  self.db2lin(np.array(self.rx2_a_power))
        # rx3_a_power_lin=  self.db2lin(np.array(self.rx3_a_power))
        # rx4_a_power_lin=  self.db2lin(np.array(self.rx4_a_power))
        
        Ptx_W = self.db2lin(self.Ptx_dBm - 30) # W
        for i in range(len(self.frequency_list)):

            B = self.B_list[i]
            N_dBm = self.N0_dBm_per_Hz + self.lin2db(B) + self.NF_dB
            N_W = self.db2lin(N_dBm - 30)
            
            
            if i == 0:
                rx1_snr_linear = (Ptx_W * self.rx1_a_power_lin) / N_W
                self.rx1_snr_dB = self.lin2db(rx1_snr_linear)
                # self.rx1_snr_dB_list.append(self.rx1_snr_dB)
                if self.rx1_snr_dB > -5:
                    self.rx1_R = B * np.minimum(self.alpha * np.log2(1 + rx1_snr_linear), self.rho_max) / 1e6
                    # self.rx1_R = B *np.log2(1 + rx1_snr_linear)/ 1e6
                else:
                    self.rx1_R = 0
                # self.rx1_R_list.append(self.rx1_R)
                
                rx2_snr_linear = (Ptx_W * self.rx2_a_power_lin) / N_W
                self.rx2_snr_dB = self.lin2db(rx2_snr_linear)
                # self.rx2_snr_dB_list.append(self.rx2_snr_dB)
                if self.rx2_snr_dB > -5:
                    self.rx2_R = B * np.minimum(self.alpha * np.log2(1 + rx2_snr_linear), self.rho_max) / 1e6
                    # self.rx2_R = B *np.log2(1 + rx2_snr_linear)/ 1e6
                else:
                    self.rx2_R = 0
                # self.rx2_R_list.append(self.rx2_R)
            else:
                rx3_snr_linear = (Ptx_W * self.rx3_a_power_lin) / N_W
                self.rx3_snr_dB = self.lin2db(rx3_snr_linear)
                # self.rx3_snr_dB_list.append(self.rx3_snr_dB)
                if self.rx3_snr_dB > -5:
                    self.rx3_R = B * np.minimum(self.alpha * np.log2(1 + rx3_snr_linear), self.rho_max) / 1e6
                    # self.rx3_R = B *np.log2(1 + rx3_snr_linear)/ 1e6
                else:
                    self.rx3_R = 0
                # self.rx3_R_list.append(self.rx3_R)
                
                rx4_snr_linear = (Ptx_W * self.rx4_a_power_lin) / N_W
                self.rx4_snr_dB = self.lin2db(rx4_snr_linear)
                # self.rx4_snr_dB_list.append(self.rx4_snr_dB)
                if self.rx4_snr_dB > -5:
                    self.rx4_R = B * np.minimum(self.alpha * np.log2(1 + rx4_snr_linear), self.rho_max) / 1e6
                    # self.rx4_R = B *np.log2(1 + rx4_snr_linear)/ 1e6
                else:
                    self.rx4_R = 0
                # self.rx4_R_list.append(self.rx4_R)
            
        # self.rx_Rate = np.array([self.rx1_R, self.rx2_R, self.rx3_R, self.rx4_R])
        self.rx_Rate = [self.rx1_R, self.rx2_R, self.rx3_R, self.rx4_R]
        self.rx_hPower_lin = [self.rx1_a_power_lin, self.rx2_a_power_lin, self.rx3_a_power_lin, self.rx4_a_power_lin]
        # print(self.rx_Rate)
        # print(self.rx1_R_list)
        # print(self.rx3_R_list)
        # print(self.rx4_R_list)
    
    ###############################################################################
        
    def update_orientation(self, t):
        angle_speed = 180.0 / 360    # per second
        step_yaw = np.random.uniform(- angle_speed * np.pi * t, angle_speed * np.pi * t)
        step_pitch = np.random.uniform(- angle_speed * 0.5 * np.pi * t, angle_speed * 0.5 * np.pi * t)
        self.ue_yaw+= step_yaw
        self.ue_pitch += step_pitch

    ###############################################################################
    # helper function
    def db2lin(self,db_values):
        return np.power(10.0, db_values / 10.0)

    def lin2db(self,linear_values):
        return 10.0 * np.log10(linear_values)
        
    def show_path(self):
        self.scene.preview(paths = self.paths)
        


class MyPattern(AntennaPattern):
    def vertical_cut(self, theta: mi.Float) -> mi.Float:
        theta_3dB = dr.deg2rad(125.0)
        SLA_v = 22.5
        return -dr.minimum(12 * dr.square((theta - dr.deg2rad(90)) / theta_3dB), SLA_v)

    def horizontal_cut(self, phi: mi.Float) -> mi.Float:
        phi_3dB = dr.deg2rad(125.0)
        A_max = 22.5
        return -dr.minimum(12 * dr.square(phi / phi_3dB), A_max)

    def combined_pattern(self, theta: mi.Float, phi: mi.Float) -> mi.Float:
        A_max = 22.5
        a_v = self.vertical_cut(theta)
        a_h = self.horizontal_cut(phi)
        total = a_v + a_h
        return -dr.minimum(-total, A_max)

    def __init__(self, polarization: str = "V"):
        if polarization not in {"V", "H", "VH"}:
            raise ValueError("Polarization must be 'V', 'H', or 'VH'")
        self.polarization = polarization

        def my_pattern(theta, phi):
            gain_dB = self.combined_pattern(theta, phi)
            gain_linear = dr.power(10.0, gain_dB / 20.0)

            if self.polarization == "V":
                c_theta = mi.Complex2f(gain_linear, dr.zeros(mi.Float, dr.width(theta)))
                c_phi = mi.Complex2f(dr.zeros(mi.Float, dr.width(phi)), dr.zeros(mi.Float, dr.width(phi)))
            elif self.polarization == "H":
                c_theta = mi.Complex2f(dr.zeros(mi.Float, dr.width(theta)), dr.zeros(mi.Float, dr.width(theta)))
                c_phi = mi.Complex2f(gain_linear, dr.zeros(mi.Float, dr.width(phi)))
            else:  # "VH"
                scale = gain_linear / dr.sqrt(2.0)
                c_theta = mi.Complex2f(scale, dr.zeros(mi.Float, dr.width(theta)))
                c_phi = mi.Complex2f(scale, dr.zeros(mi.Float, dr.width(phi)))

            return c_theta, c_phi

        self.patterns = [lambda theta, phi: my_pattern(theta, phi)]