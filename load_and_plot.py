import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.io import loadmat, savemat
import pandas as pd
import os
import copy
import math

import argparse

# get argument from user
parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, required = False, default=None, help='the path where the training/simulation data is stored')
parser.add_argument('--ep_num', type = int, required = False, default=100, help='total number of episodes')


# extract argument
args = parser.parse_args()
STORE_PATH = args.path
EP_NUM = args.ep_num
if STORE_PATH is None:
    STORE_PATH = 'data/storage/2022-09-13 11_39_46' # best TD3 so far
    STORE_PATH = 'data/storage/2021-01-08 16_52_32_robust_2' # DDPG Benchmark from https://ieeexplore.ieee.org/document/9434412


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
    def __init__(self, store_path, \
                       user_num = 2, attacker_num = 1, RIS_ant_num = 4, \
                       ep_num = EP_NUM, step_num = 100): # RIS_ant_num = 16 (not true)

        self.color_list = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
        self.store_path = store_path + '//'
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.RIS_ant_num = RIS_ant_num
        self.ep_num = ep_num
        self.step_num = step_num

        self.all_steps = self.load_all_steps()


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
        if not os.path.exists(self.store_path + 'plot'):
            os.makedirs(self.store_path + 'plot')
            os.makedirs(self.store_path + 'plot/RIS')

        color_list = ['b', 'g', 'c', 'k', 'm', 'r', 'y']
        
        
        ###############################
        # plot reward
        ###############################
        fig = plt.figure('reward')
        plt.plot(range(len(self.all_steps['reward'])), self.all_steps['reward'])
        plt.xlabel("Time Steps ($t$)")
        plt.ylabel("Reward")
        plt.savefig(self.store_path + 'plot/reward.png')
        plt.cla()
        
        
        ###############################
        # plot secure capacity
        ###############################
        fig = plt.figure('secure_capacity')
        for i in range(self.user_num):
            plt.plot(range(len(self.all_steps['secure_capacity'][i])), self.all_steps['secure_capacity'][i], c=color_list[i])
        plt.legend(['user_' + str(i) for i in range(self.user_num)])
        plt.xlabel("Time Steps ($t$)")
        plt.ylabel("Secure Capacity")
        plt.savefig(self.store_path + 'plot/secure_capacity.png')
        plt.cla()

        
        ###############################
        # plot average sum secrecy rate of each episode
        ###############################
        fig = plt.figure('average_sum_secrecy_rate')
        sum_secrecy_rate = np.array(self.all_steps['secure_capacity'])
        sum_secrecy_rate = np.sum(sum_secrecy_rate, axis = 0)
        average_sum_secrecy_rate = []
        for i in range(0, self.ep_num * self.step_num, self.step_num):
            ssr_one_episode = sum_secrecy_rate[i:i+self.step_num] # ssr means Sum Secrecy Rate
            try:
                _ = sum(ssr_one_episode) / len(ssr_one_episode)
            except:
                _ = 0
            average_sum_secrecy_rate.append(_)
        plt.plot(range(len(average_sum_secrecy_rate)), average_sum_secrecy_rate)
        plt.xlabel("Episodes (Ep)")
        plt.ylabel("Average Sum Secrecy Rate")
        plt.savefig(self.store_path + 'plot/average_sum_secrecy_rate.png')
        plt.cla()
        
        
        ###############################
        # plot user capacity
        ###############################
        fig = plt.figure('user_capacity')
        for i in range(self.user_num):
            plt.plot(range(len(self.all_steps['user_capacity'][i])), self.all_steps['user_capacity'][i], c=color_list[i])
        plt.legend(['user_' + str(i) for i in range(self.user_num)])
        plt.xlabel("Time Steps ($t$)")
        plt.ylabel("User Capacity")
        plt.savefig(self.store_path + 'plot/user_capacity.png')
        plt.cla()

        
        ###############################
        # plot attacker capacity
        ###############################
        fig = plt.figure('attaker_capacity')
        for i in range(self.attacker_num):
            plt.plot(range(len(self.all_steps['attaker_capacity'][i])), self.all_steps['attaker_capacity'][i], c=color_list[i])
        plt.legend(['attacker_' + str(i) for i in range(self.attacker_num)])
        plt.xlabel("Time Steps ($t$)")
        plt.ylabel("Attack Capacity")
        plt.savefig(self.store_path + 'plot/attaker_capacity.png')
        plt.close('all')
        
        
        ###############################
        # plot ris
        ###############################
        for i in range(self.RIS_ant_num):
            self.plot_one_RIS_element(i)
            
        
        ###############################
        # plot trajectory
        ###############################
        self.plot_trajectory()

    
    def plot_one_RIS_element(self, index):
        """
        docstring
        """
        ax_real_imag = plt.subplot(1,1,1)
        ax_pase = ax_real_imag.twinx()
        #plt.ylim(ymax = 1, ymin = -1)
        #plt.xlim(xmax = 10000 , xmin = 10000 - 100)
        ax_real_imag.plot(range(len(self.all_steps['RIS_elements'][index])), np.real(self.all_steps['RIS_elements'][index]), c = self.color_list[0])
        ax_real_imag.plot(range(len(self.all_steps['RIS_elements'][index])), np.imag(self.all_steps['RIS_elements'][index]), c = self.color_list[1])
        phase_list = []
        for complex_num in self.all_steps['RIS_elements'][index]:
            phase_list.append(cmath.phase(complex_num))
        plt.ylim(ymax = cmath.pi, ymin = -cmath.pi)
        ax_pase.plot(range(len(self.all_steps['RIS_elements'][index])), phase_list, c = self.color_list[2])
#        plt.xlabel("Time Steps ($t$)")
#        plt.ylabel("RIS Dimension")
        # plt.set_ylabel("position")
        # plt.set_ylabel("position")
        # plt.set_xlabel("Time Steps ($t$)")
        plt.savefig(self.store_path + 'plot/RIS/RIS_' + str(index) + '_element.png')
        plt.close('all')
        pass

        
    def plot_trajectory(self):
        # get init location
        init_uav_coord = read_init_location(entity_type = 'UAV')
        init_user_coord_0 = read_init_location(entity_type = 'user', index=0)
        init_user_coord_1 = read_init_location(entity_type = 'user', index=1)
        
        ep_num = EP_NUM
        interval = int(0.2 * EP_NUM)
        ep_list = [0] + [i for i in range(20-1, ep_num, interval)]
        if EP_NUM - 1 not in ep_list: ep_list.append(EP_NUM - 1)
        color_list_template = ['b', 'g', 'c', 'k', 'm', 'r', 'y', 'black', 'red']
        
        
        color_list = copy.deepcopy(color_list_template)
        for i in ep_list:
            # read the mat file
            filename = f'simulation_result_ep_{i}.mat'
            filename = os.path.join(STORE_PATH, filename)
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
            plt.plot(uav_coord[1],uav_coord[0], c=color_list.pop(0))
        
        # user 0 movt
        direction_fai = -1/2*math.pi 
        distance_delta_d = 0.2
        user_coord_0 = [ [init_user_coord_0[0]], [init_user_coord_0[1]] ]
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
        plt.plot(user_coord_0[1],user_coord_0[0], c=color_list.pop(0))
        plt.plot(user_coord_0[1][0], user_coord_0[0][0], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
        
        
        # user 1 movt
        direction_fai = -1/2*math.pi 
        distance_delta_d = 0.2
        user_coord_0 = [ [init_user_coord_1[0]], [init_user_coord_1[1]] ]
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
        plt.plot(user_coord_0[1],user_coord_0[0], c=color_list.pop(0))
        plt.plot(user_coord_0[1][0], user_coord_0[0][0], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
        
        plt.legend(ep_list)
        plt.grid()
        plt.xlim(0, 50)
        plt.ylim(-10, 30)
        plt.gca().invert_yaxis()
        plt.savefig(self.store_path + 'plot/trajectory.png')
        plt.cla()


    def restruct(self):
        savemat(self.store_path + 'all_steps.mat',self.all_steps)
        return 0
if __name__ == '__main__':
    LoadPlotObject = LoadAndPlot(
        store_path = STORE_PATH,
        )
    LoadPlotObject.plot()
    LoadPlotObject.restruct()

    

