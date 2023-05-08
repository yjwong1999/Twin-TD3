import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
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
        # plot average sum secrecy rate of each episode
        ###############################
        fig = plt.figure('average_sum_secrecy_rate')
        
#        store_paths = ['data/storage/ddpg 4', 'data/storage/td3 3', 'data/storage/ddpg seem 6', 'data/storage/td3 seem 2']
      #  store_paths = r'data/storage/scratch/td3_ssr'
        legends = ['TDDRL', 'TTD3', 'TDDRL (Energy Penalty)', 'TTD3 (Energy Penalty)']
        all_average_sum_secrecy_rate = []
        for store_path, legend in zip(self.store_paths, legends):
            self.store_path = store_path + '//'
            self.all_steps = self.load_all_steps()
            
            sum_secrecy_rate = np.array(self.all_steps['secure_capacity'])
            sum_secrecy_rate = np.sum(sum_secrecy_rate, axis = 0)
            average_sum_secrecy_rate = []
            ssr = []
            for i in range(0, self.ep_num * self.step_num, self.step_num):
                ssr_one_episode = sum_secrecy_rate[i:i+self.step_num] # ssr means Sum Secrecy Rate
                ssr.append(ssr_one_episode)
                try:
                    _ = sum(ssr_one_episode) / len(ssr_one_episode)
                except:
                    _ = 0
                average_sum_secrecy_rate.append(_)
                
            all_average_sum_secrecy_rate.append(average_sum_secrecy_rate)  
            plt.plot(range(len(average_sum_secrecy_rate)), average_sum_secrecy_rate, label=legend)
            plt.xlabel("Episodes (Ep)")
            plt.ylabel("Average Sum Secrecy Rate")
                
        plt.legend()
        plt.savefig('data/average_sum_secrecy_rate11.png')
        
        
        # transpose
        '''
        all_average_sum_secrecy_rate = np.array(all_average_sum_secrecy_rate)
        all_average_sum_secrecy_rate = np.transpose(all_average_sum_secrecy_rate)
        all_average_sum_secrecy_rate = list(all_average_sum_secrecy_rate)
        '''
        
        # dictionary of lists  
        dict = {legend: average_sum_secrecy_rate for legend, average_sum_secrecy_rate in zip(legends, all_average_sum_secrecy_rate)} 
        df = pd.DataFrame(dict)
        df.to_excel('data/average_sum_secrecy_rate.xlsx', index=False) 

        
if __name__ == '__main__':
    LoadPlotObject = LoadAndPlot(
        store_paths = ['data/storage/scratch/ddpg_ssr', 'data/storage/scratch/td3_ssr', 'data/storage/scratch/ddpg_see', 'data/storage/scratch/td3_see'],
        ep_num = 300,
        )
    LoadPlotObject.plot()

    

