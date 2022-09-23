import numpy as np
import math
#from math_tool import *

class UAV(object):
    """
    UAV object with coordinate 
    And with ULA antenas, default 8 
    And limited power
    And with fixed rotation angle
    """
    def __init__(self, coordinate, index = 0, rotation = 0, ant_num=16, ant_type = 'ULA', max_movement_per_time_slot = 0.5):
        """
        coordinate is the init coordinate of UAV, meters, np.array
        """
        self.max_movement_per_time_slot = max_movement_per_time_slot
        self.type = 'UAV'
        self.coordinate = coordinate
        self.rotation = rotation
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]
        self.index = index

        # init beamforming matrix in UAV (must be inited in env.py)
        self.G = np.mat(np.zeros((ant_num, 1)))
        self.G_Pmax = 0

    def reset(self, coordinate):
        """
        reset UAV coordinate
        """
        self.coordinate = coordinate
        
    def update_coor_sys(self, delta_angle):
        """
        used in function move to update the relevant coordinate system 
        """
        self.rotation = self.rotation + delta_angle
        coor_sys_x = np.array([\
        math.cos(self.rotation),\
        math.sin(self.rotation),\
        0])
        coor_sys_z = np.array([\
        0,\
        0,\
        -1])
        coor_sys_y = np.cross(coor_sys_z, coor_sys_x)
        self.coor_sys = np.array([coor_sys_x,coor_sys_y,coor_sys_z])
        
    def update_coordinate(self, distance_delta_d, direction_fai):
        """
        used in function move to update UAV cordinate
        """
        delta_x = distance_delta_d * math.cos(direction_fai)
        delta_y = distance_delta_d * math.sin(direction_fai)
        self.coordinate[0] += delta_x
        self.coordinate[1] += delta_y

    def move(self, distance_delta_d, direction_fai, delta_angle = 0):
        """
        preform the 2D movement every step
        """
        self.update_coordinate(distance_delta_d, direction_fai)
        self.update_coor_sys(delta_angle)

class RIS(object):
    """
    reconfigrable intelligent surface
    with N reflecting elements, UPA, default 4 X 4 = 16
    continues phase shift
    """
    def __init__(self, coordinate, coor_sys_z, index = 0, ant_num=36, ant_type = 'UPA'):
        """
        coordinate is the init coordinate of with N reflecting elements, meters, np.array
        norm_vec is the normal vector of the reflecting direction
        !!! ant_num Must be the square of a certain int number
        """
        self.type = 'RIS'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        coor_sys_z = coor_sys_z / np.linalg.norm(coor_sys_z)
        coor_sys_x = np.cross(coor_sys_z, np.array([0,0,1]))
        coor_sys_x = coor_sys_x / np.linalg.norm(coor_sys_x)
        coor_sys_y = np.cross(coor_sys_z, coor_sys_x)
        self.coor_sys = [coor_sys_x,coor_sys_y,coor_sys_z]
        self.index = index

        # init reflecting phase shift
        self.Phi = np.mat(np.diag(np.ones(self.ant_num, dtype=complex)), dtype = complex)

class User(object):
    """
    user with single antenas
    """
    def __init__(self, coordinate, index, ant_num = 1, ant_type = 'single'):
        """
        coordinate is the init coordinate of user, meters, np.array
        ant_num is the antenas number of user
        """
        self.type = 'user'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        # init the capacity
        self.capacity = 0
        self.secure_capacity = 0
        self.QoS_constrain = 0
        # init the comprehensive_channel, (must used in env.py to init)
        self.comprehensive_channel = 0
        # init receive noise sigma in dB
        self.noise_power = -114

    def reset(self, coordinate):
        """
        reset user coordinate
        """
        self.coordinate = coordinate
        
    def update_coordinate(self, distance_delta_d, direction_fai):
        """
        used in function move to update UAV cordinate
        """
        delta_x = distance_delta_d * math.cos(direction_fai)
        delta_y = distance_delta_d * math.sin(direction_fai)
        self.coordinate[0] += delta_x
        self.coordinate[1] += delta_y

    def move(self, distance_delta_d, direction_fai):
        """
        preform the 2D movement every step
        """
        self.update_coordinate(distance_delta_d, direction_fai)
        
class Attacker(object):
    """
    Attacker with single antenas
    """
    def __init__(self, coordinate, index, ant_num = 1, ant_type= 'single'):
        """
        coordinate is the init coordinate of Attacker, meters, np.array
        ant_num is the antenas number of Attacker
        """
        self.type = 'attacker'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        self.coor_sys = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        # init the capacity, this is a K length np.array ,shape: (K,)
        # represent the attack rate for kth user, (must init in env.py)
        self.capacity = 0
        self.comprehensive_channel = 0
        # init receive noise sigma in dBmW
        self.noise_power = -114

    def reset(self, coordinate):
        """
        reset attacker coordinate
        """
        self.coordinate = coordinate

    def update_coordinate(self, distance_delta_d, direction_fai):
        """
        used in function move to update UAV cordinate
        """
        delta_x = distance_delta_d * math.cos(direction_fai)
        delta_y = distance_delta_d * math.sin(direction_fai)
        self.coordinate[0] += delta_x
        self.coordinate[1] += delta_y

    def move(self, distance_delta_d, direction_fai):
        """
        preform the 2D movement every step
        """
        self.update_coordinate(distance_delta_d, direction_fai)