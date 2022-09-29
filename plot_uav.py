import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
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
    #STORE_PATH = 'data/storage/ddpg'
    #STORE_PATH = 'data/storage/2021-01-08 16_52_32_robust_2' # DDPG Benchmark from https://ieeexplore.ieee.org/document/9434412


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
plt.gca().invert_yaxis()
#plt.xlim(0, 50)
#plt.ylim(-5, 25)
plt.show()




'''
# read the mat file
i = 0
filename = f'simulation_result_ep_{i}.mat'
filename = os.path.join(STORE_PATH, filename)
data = loadmat(filename)
print(data[f'result_{i}'][0][0][-1])
'''
