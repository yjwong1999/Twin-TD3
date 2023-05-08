import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.io import loadmat, savemat
import pandas as pd
import os
import copy
import math


######################################################
# new for energy 
# energy related parameters of rotary-wing UAV
# based on Energy Minimization in Internet-of-Things System Based on Rotary-Wing UAV
P_i = 790.6715
P_0 = 580.65
U2_tip = (200) ** 2
s = 0.05
d_0 = 0.3
p = 1.225
A = 0.79
delta_time = 0.1 #0.1/1000 #0.1ms

# add ons hover veloctiy
# based on https://www.intechopen.com/chapters/57483
m = 1.3 # mass: assume 1.3kg https://www.droneblog.com/average-weights-of-common-types-of-drones/#:~:text=In%20most%20cases%2C%20toy%20drones,What%20is%20this%3F
g = 9.81 # gravity
T = m * g # thrust
v_0 = (T / (A * 2 * p)) ** 0.5

def get_energy_consumption(v_t):
    '''
    arg
    1) v_t = displacement per time slot
    '''
    energy_1 = P_0 \
                + 3 * P_0 * (abs(v_t)) ** 2 / U2_tip \
                + 0.5 * d_0 * p * s * A * (abs(v_t))**3
    
    energy_2 = P_i * ((
                    (1 + (abs(v_t) ** 4) / (4 * (v_0 ** 4))) ** 0.5 \
                    - (abs(v_t) ** 2) / (2 * (v_0 **2)) \
                ) ** 0.5)
    
    energy = delta_time * (energy_1 + energy_2)
    return energy 

ENERGY_MIN = get_energy_consumption(0.25)
ENERGY_MAX = get_energy_consumption(0)

######################################################


# modified from data_manager.py
init_data_file = 'data/init_location.xlsx'
def read_init_location(entity_type = 'user', index = 0):
    if entity_type == 'user' or 'attacker' or 'RIS' or 'RIS_norm_vec' or 'UAV':
        return np.array([\
        pd.read_excel(init_data_file, sheet_name=entity_type)['x'][index],\
        pd.read_excel(init_data_file, sheet_name=entity_type)['y'][index],\
        pd.read_excel(init_data_file, sheet_name=entity_type)['z'][index]])
    else:
        return None


# load and plot everything
class LoadAndPlot(object):
    """
    load date and plot 2022-07-22 16_16_26
    """
    def __init__(self, store_paths, \
                       user_num = 2, attacker_num = 1, RIS_ant_num = 4, \
                       ep_num = 300, step_num = 100): # RIS_ant_num = 16 (not true)

        self.store_paths = store_paths
        self.color_list = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
#        self.store_path = store_path + '//'
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.RIS_ant_num = RIS_ant_num
        self.ep_num = ep_num
        self.step_num = step_num


    def load_one_ep(self, file_name):
        m = loadmat(self.store_path + file_name)
        return m


    def load_all_steps(self):
        result_dic = {}
        result_dic.update({'reward':[]})

        result_dic.update({'user_capacity':[]})
        for i in range(self.user_num):
            result_dic['user_capacity'].append([])

        result_dic.update({'secure_capacity':[]})
        for i in range(self.user_num):
            result_dic['secure_capacity'].append([])

        result_dic.update({'attaker_capacity':[]})
        for i in range(self.attacker_num):
            result_dic['attaker_capacity'].append([])
        
        result_dic.update({'RIS_elements':[]})
        for i in range(self.RIS_ant_num):
            result_dic['RIS_elements'].append([])

        for ep_cnt in range(self.ep_num):
            mat_ep = self.load_one_ep("simulation_result_ep_" + str(ep_cnt) + ".mat")

            one_ep_reward = mat_ep["result_" + str(ep_cnt)]["reward"][0][0]
            result_dic['reward'] += list(one_ep_reward[:, 0])

            one_ep_user_capacity = mat_ep["result_" + str(ep_cnt)]["user_capacity"][0][0]
            for i in range(self.user_num):
                result_dic['user_capacity'][i] += list(one_ep_user_capacity[:, i])
            
            one_ep_secure_capacity = mat_ep["result_" + str(ep_cnt)]["secure_capacity"][0][0]
            for i in range(self.user_num):
                result_dic['secure_capacity'][i] += list(one_ep_secure_capacity[:, i])
            
            one_ep_attaker_capacity = mat_ep["result_" + str(ep_cnt)]["attaker_capacity"][0][0]
            for i in range(self.attacker_num):
                result_dic['attaker_capacity'][i] += list(one_ep_attaker_capacity[:, i])

            one_ep_RIS_first_element = mat_ep["result_" + str(ep_cnt)]["reflecting_coefficient"][0][0]
            for i in range(self.RIS_ant_num):
                result_dic['RIS_elements'][i] += list(one_ep_RIS_first_element[:, i])

        return result_dic


    def plot(self):
        """
        plot result
        b--blue c--cyan(青色） g--green k--black m--magenta（紫红色） r--red w--white y--yellow 
        """

        
      
        ###############################
        # plot trajectory
        ###############################
        # create a fig
        fig, ax = plt.subplots(figsize=(5.4,5.2))
        #fig = plt.figure('trajectory')
        MARKER_SIZE = 8

        # colour
        color_list_template = ['b', 'g', 'c', 'k', 'm', 'r', 'y', 'black', 'red']
        color_list_template = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_list = copy.deepcopy(color_list_template)

        # get init location
        init_uav_coord = read_init_location(entity_type = 'UAV')
        init_ris_coord = read_init_location(entity_type = 'RIS')
        init_eaves_coord = read_init_location(entity_type = 'attacker')
        init_user_coord_0 = read_init_location(entity_type = 'user', index=0)
        init_user_coord_1 = read_init_location(entity_type = 'user', index=1)
    
        
        plt.text(20, init_uav_coord[0]-1, 'UAV Initial Coordinate', fontsize = 11)
        plt.plot([init_uav_coord[1]], [init_uav_coord[0]], marker="s", markersize=MARKER_SIZE, markeredgecolor="black", markerfacecolor="none")
        
        plt.text(46, init_ris_coord[0]-1, 'RIS', fontsize = 11)
        plt.plot([init_ris_coord[1]], [init_ris_coord[0]], marker="d", markersize=MARKER_SIZE, markeredgecolor="black", markerfacecolor="none")
        
        plt.text(36, init_eaves_coord[0]-1, 'Eavesdropper', fontsize = 11)
        plt.plot([init_eaves_coord[1]], [init_eaves_coord[0]], marker="v", markersize=MARKER_SIZE, markeredgecolor="black", markerfacecolor="none")
        
        # paths
#        store_paths = ['data/storage/ddpg 2', 'data/storage/td3 3', 'data/storage/ddpg seem 3', 'data/storage/td3 seem 5']
#        store_paths = ['data/storage/ddpg 2', 'data/storage/td3 3', 'data/storage/ddpg seem 3', 'data/storage/td3 seem 5']
        legends = ['TDDRL', 'TTD3', 'TDDRL (Energy Penalty)', 'TTD3 (Energy Penalty)']
        legends = ['Benchmark 1', 'Benchmark 2', 'Benchmark 3', 'Proposed method']
     
        for store_path, legend in zip(self.store_paths, legends):
            # read the mat file
            i = 5 - 1 # episode 300
            filename = f'simulation_result_ep_{i}.mat'
            filename = os.path.join(store_path, filename)
            data = loadmat(filename)
        
            # uav movt
            uav_coord = [ [init_uav_coord[0]], [init_uav_coord[1]] ]
        
            uav_movt = data[f'result_{i}'][0][0][-1]
            for j in range(uav_movt.shape[0]):
                move_x = uav_movt[j][0]
                move_y = uav_movt[j][1]
        
                prev_x = uav_coord[0][-1]
                prev_y = uav_coord[1][-1]
        
                current_x = prev_x + move_x
                current_y = prev_y + move_y
        
                uav_coord[0].append(current_x)
                uav_coord[1].append(current_y)
            plt.plot(uav_coord[1],uav_coord[0], c=color_list.pop(0), label=legend)
        
        # user 0 movt
        direction_fai = -1/2*math.pi 
        distance_delta_d = 0.25
        user_coord_0 = [ [init_user_coord_0[0]], [init_user_coord_0[1]] ]
        plt.text(29, init_user_coord_0[0]-1, 'User 1 Initial Coordinate', fontsize = 11)
        #color_list = copy.deepcopy(color_list_template)
        for j in range(uav_movt.shape[0]):
            delta_x = distance_delta_d * math.cos(direction_fai)
            delta_y = distance_delta_d * math.sin(direction_fai)
        
            prev_x = user_coord_0[0][-1]
            prev_y = user_coord_0[1][-1]
        
            current_x = prev_x + delta_x
            current_y = prev_y + delta_y
        
            user_coord_0[0].append(current_x)
            user_coord_0[1].append(current_y)
        plt.plot(user_coord_0[1],user_coord_0[0], c=color_list.pop(0), linestyle='dashed', linewidth=2, label='User 1')
        plt.plot(user_coord_0[1][0], user_coord_0[0][0], marker="o", markersize=MARKER_SIZE, markeredgecolor="black", markerfacecolor="none")
        
        
        # user 1 movt
        direction_fai = -1/2*math.pi 
        distance_delta_d = 0.25
        user_coord_1 = [ [init_user_coord_1[0]], [init_user_coord_1[1]] ]
        plt.text(13, init_user_coord_1[0]-1, 'User 2 Initial Coordinate', fontsize = 11)
        #color_list = copy.deepcopy(color_list_template)
        for j in range(uav_movt.shape[0]):
            delta_x = distance_delta_d * math.cos(direction_fai)
            delta_y = distance_delta_d * math.sin(direction_fai)
        
            prev_x = user_coord_1[0][-1]
            prev_y = user_coord_1[1][-1]
        
            current_x = prev_x + delta_x
            current_y = prev_y + delta_y
        
            user_coord_1[0].append(current_x)
            user_coord_1[1].append(current_y)
        plt.plot(user_coord_1[1],user_coord_1[0], c=color_list.pop(0), linestyle='dashed', linewidth=2, label='User 2')
        plt.plot(user_coord_1[1][0], user_coord_1[0][0], marker="o", markersize=MARKER_SIZE, markeredgecolor="black", markerfacecolor="none")
        
        # plot a line between last coord of user 0 and user 1
        plt.plot([user_coord_0[1][-1], user_coord_1[1][0-1]], [user_coord_0[0][-1], user_coord_1[0][-1]], 'gray', linestyle='dashed')
        
        # plot midpoint between last coord of user 0 and user 1
        plt.plot([(user_coord_0[1][-1] + user_coord_1[1][0-1])/2], [(user_coord_0[0][-1] + user_coord_1[0][0-1])/2], marker="o", markersize=MARKER_SIZE, markeredgecolor="black", markerfacecolor="none")
        plt.text(12, 18, "Midpoint of \ntwo user's last location", fontsize = 11)
       
        plt.legend(loc='center right', fontsize=10)
        plt.grid()
        plt.xlim(0, 50)
        plt.ylim(-10, 30)
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.savefig('data/trajectory.png')
        #plt.cla()

        
if __name__ == '__main__':
    LoadPlotObject = LoadAndPlot(
        store_paths = ['data/storage/scratch/ddpg_ssr', 'data/storage/scratch/td3_ssr', 'data/storage/scratch/ddpg_see', 'data/storage/scratch/td3_see'],
        ep_num=300,
        )
    LoadPlotObject.plot()

    

