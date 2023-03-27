# debug field
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse

# get argument from user
parser = argparse.ArgumentParser()
parser.add_argument('--drl', type = str, required = True, default='td3', help='which drl algo would you like to choose [ddpg, td3]')
parser.add_argument('--reward', type = str, required = True, default='see', help='which reward would you like to implement [ssr, see]')
args = parser.parse_args()
DRL_ALGO = args.drl
REWARD_DESIGN = args.reward

assert DRL_ALGO in ['ddpg', 'td3'], "DRL must be ['ddpg', 'td3']"
assert REWARD_DESIGN in ['ssr', 'see'], "DRL must be ['ssr', 'see']"

if DRL_ALGO == 'td3':
    from td3 import Agent
    import td3_2 # seems not used
elif DRL_ALGO == 'ddpg':
    from ddpg import Agent
    import ddpg_2 # seems not used
import ddpg

from env import MiniSystem
import numpy as np
import math
import time
import torch

# 1 init system model
system = MiniSystem(
    user_num=2,
    RIS_ant_num=4,
    UAV_ant_num=4,
    if_dir_link=1,
    if_with_RIS=True,
    if_move_users=True,
    if_movements=True,
    reverse_x_y=(False, False),
    if_UAV_pos_state = True,
    reward_design = REWARD_DESIGN
    )
if_Theta_fixed = False
if_G_fixed = False
if_BS = False
if_robust = True
# 2 init RL Agent

episode_num = 300
episode_cnt = 0
step_num = 100

agent_1_param_dic = {}
agent_1_param_dic["alpha"] = 0.0001
agent_1_param_dic["beta"] = 0.001
agent_1_param_dic["input_dims"] = system.get_system_state_dim()
agent_1_param_dic["tau"] = 0.001
agent_1_param_dic["batch_size"] = 64
agent_1_param_dic["n_actions"] = system.get_system_action_dim() - 2
agent_1_param_dic["action_noise_factor"] = 0.1
agent_1_param_dic["memory_max_size"] = int(5/5 * episode_num * step_num) #/2
agent_1_param_dic["agent_name"] = "G_and_Phi"
agent_1_param_dic["layer1_size"] = 800
agent_1_param_dic["layer2_size"] = 600
agent_1_param_dic["layer3_size"] = 512
agent_1_param_dic["layer4_size"] = 256

agent_2_param_dic = {}
agent_2_param_dic["alpha"] = 0.0001
agent_2_param_dic["beta"] = 0.001
agent_2_param_dic["input_dims"] = 3
agent_2_param_dic["tau"] = 0.001
agent_2_param_dic["batch_size"] = 64
agent_2_param_dic["n_actions"] = 2
agent_2_param_dic["action_noise_factor"] = 0.5
agent_2_param_dic["memory_max_size"] = int(5/5 * episode_num * step_num) #/2
agent_2_param_dic["agent_name"] = "UAV"
agent_2_param_dic["layer1_size"] = 400
agent_2_param_dic["layer2_size"] = 300
agent_2_param_dic["layer3_size"] = 256
agent_2_param_dic["layer4_size"] = 128

agent_1 = Agent(
    alpha       = agent_1_param_dic["alpha"],
    beta        = agent_1_param_dic["beta"],
    input_dims  = [agent_1_param_dic["input_dims"]],
    tau         = agent_1_param_dic["tau"],
    env         = system,
    batch_size  = agent_1_param_dic["batch_size"],
    layer1_size=agent_1_param_dic["layer1_size"],
    layer2_size=agent_1_param_dic["layer2_size"], 
    layer3_size=agent_1_param_dic["layer3_size"],
    layer4_size=agent_1_param_dic["layer4_size"],
    n_actions   = agent_1_param_dic["n_actions"],
    max_size = agent_1_param_dic["memory_max_size"],
    agent_name= agent_1_param_dic["agent_name"]
    ) 

agent_2 = Agent(
    alpha       = agent_2_param_dic["alpha"],
    beta        = agent_2_param_dic["beta"],
    input_dims  = [agent_2_param_dic["input_dims"]],
    tau         = agent_2_param_dic["tau"],
    env         = system,
    batch_size  = agent_2_param_dic["batch_size"],
    layer1_size=agent_2_param_dic["layer1_size"],
    layer2_size=agent_2_param_dic["layer2_size"], 
    layer3_size=agent_2_param_dic["layer3_size"],
    layer4_size=agent_2_param_dic["layer4_size"],
    n_actions   = agent_2_param_dic["n_actions"],
    max_size = agent_2_param_dic["memory_max_size"],
    agent_name= agent_2_param_dic["agent_name"]
    ) 


'''
agent_2.load_models(
     load_file_actor = '/home/tham/Desktop/uav-td3/data/storage/2022-09-13 11_39_46/best/Actor_G_and_Phi_td3',
     load_file_critic_1 ='/home/tham/Desktop/uav-td3/data/storage/2022-09-13 11_39_46/best/Critic_1_G_and_Phi_td3',
     load_file_critic_2 ='/home/tham/Desktop/uav-td3/data/storage/2022-09-13 11_39_46/best/Critic_2_G_and_Phi_td3'
     )
'''

'''
# with benchmark data
agent_2.load_models(
     load_file_actor = '/home/tham/Desktop/uav-td3/data/storage/2021-01-08 16_52_32_robust_2/Actor_UAV_ddpg',
     load_file_critic ='/home/tham/Desktop/uav-td3/data/storage/2021-01-08 16_52_32_robust_2/Critic_UAV_ddpg'
     )
'''

meta_dic = {}
print("***********************system information******************************")
print("folder_name:     "+str(system.data_manager.time_stemp))
meta_dic['folder_name'] = system.data_manager.time_stemp
print("user_num:        "+str(system.user_num))
meta_dic['user_num'] = system.user_num
print("if_dir:          "+str(system.if_dir_link))
meta_dic['if_dir_link'] = system.if_dir_link
print("if_with_RIS:     "+str(system.if_with_RIS))
meta_dic['if_with_RIS'] = system.if_with_RIS
print("if_user_m:       "+str(system.if_move_users))
meta_dic['if_move_users'] = system.if_move_users
print("RIS_ant_num:     "+str(system.RIS.ant_num))
meta_dic['system_RIS_ant_num'] = system.RIS.ant_num
print("UAV_ant_num:     "+str(system.UAV.ant_num))
meta_dic['system_UAV_ant_num'] = system.UAV.ant_num
print("if_movements:    "+str(system.if_movements))
meta_dic['system_if_movements'] = system.if_movements
print("reverse_x_y:     "+str(system.reverse_x_y))
meta_dic['system_reverse_x_y'] = system.reverse_x_y
print("if_UAV_pos_state:"+str(system.if_UAV_pos_state))
meta_dic['if_UAV_pos_state'] = system.if_UAV_pos_state

print("ep_num:          "+str(episode_num))
meta_dic['episode_num'] = episode_num
print("step_num:        "+str(step_num))
meta_dic['step_num'] = step_num
print("***********************agent_1 information******************************")
tplt = "{0:{2}^20}\t{1:{2}^20}"
for i in agent_1_param_dic:
    parm = agent_1_param_dic[i]
    print(tplt.format(i, parm, chr(12288)))
meta_dic["agent_1"] = agent_1_param_dic

print("***********************agent_2 information******************************")
for i in agent_2_param_dic:
    parm = agent_2_param_dic[i]
    print(tplt.format(i, parm, chr(12288)))
meta_dic["agent_2"] = agent_2_param_dic

system.data_manager.save_meta_data(meta_dic)

print("***********************traning information******************************")

best = -99999
while episode_cnt < episode_num:
    # 1 reset the whole system
    system.reset()
    step_cnt = 0
    score_per_ep = 0

    # 2 get the initial state
    if if_robust:
        tmp = system.observe()
        #z = np.random.multivariate_normal(np.zeros(2), 0.5*np.eye(2), size=len(tmp)).view(np.complex128)
        z = np.random.normal(size=len(tmp))
        observersion_1 = list(
            np.array(tmp) + 0.6 *1e-7* z
            )
    else:
        observersion_1 = system.observe()
    observersion_2 = list(system.UAV.coordinate)
    if episode_cnt == 80:
        print("break point")
    while step_cnt < step_num:
        # 1 count num of step in one episode
        step_cnt += 1
        # judge if pause the whole system
        if not system.render_obj.pause:
            # 2 choose action acoording to current state
            """
            epsilon = 0.5
            epsilon = epsilon + epsilon * (episode_cnt*100 + step_cnt)/(step_num*episode_num)
            ran = np.random.uniform()
            if ran < epsilon:
                action = agent.choose_action(observersion, greedy=0)
            else:
                action = list(np.random.uniform(-1,1,n_actions))
            """
            action_1 = agent_1.choose_action(observersion_1, greedy=agent_1_param_dic["action_noise_factor"] * math.pow((1-episode_cnt / episode_num), 2))
            action_2 = agent_2.choose_action(observersion_2, greedy=agent_2_param_dic["action_noise_factor"]* math.pow((1-episode_cnt / episode_num), 2))
            if if_BS:
                action_2[0]=0
                action_2[1]=0

            if if_Theta_fixed:
                action_1[0+2 * system.UAV.ant_num * system.user_num:] = len(action_1[0+2 * system.UAV.ant_num * system.user_num:])*[0]
                
            if if_G_fixed:
                action_1[0:0+2 * system.UAV.ant_num * system.user_num]=np.array([-0.0313, -0.9838, 0.3210, 1.0, -0.9786, -0.1448, 0.3518, 0.5813, -1.0, -0.2803, -0.4616, -0.6352, -0.1449, 0.7040, 0.4090, -0.8521]) * math.pow(episode_cnt / episode_num, 2) * 0.7
                #action_1[0:0+2 * system.UAV.ant_num * system.user_num]=len(action_1[0:0+2 * system.UAV.ant_num * system.user_num])*[0.5]
            # 3 get newstate, reward
            if system.if_with_RIS:
                new_state_1, reward, done, info = system.step(
                    action_0=action_2[0],
                    action_1=action_2[1],
                    G=action_1[0:0+2 * system.UAV.ant_num * system.user_num],
                    Phi=action_1[0+2 * system.UAV.ant_num * system.user_num:],
                    set_pos_x=action_2[0],
                    set_pos_y=action_2[1]
                )
                new_state_2 = list(system.UAV.coordinate)
            else:
                new_state_1, reward, done, info = system.step(
                    action_0=action_2[0],
                    action_1=action_2[1],
                    G=action_1[0:0+2 * system.UAV.ant_num * system.user_num],
                    set_pos_x=action_2[0],
                    set_pos_y=action_2[1]
                )
                new_state_2 = list(system.UAV.coordinate)

            score_per_ep += reward
            # 4 store state pair into mem pool
            agent_1.remember(observersion_1, action_1, reward, new_state_1, int(done))
            agent_2.remember(observersion_2, action_2, reward, new_state_2, int(done))
            # 5 update DDPG net
            agent_1.learn()
            agent_2.learn() #agent_2.learn()

            #system.render_obj.render(0.001) # no rendering for faster
            observersion_1 = new_state_1
            observersion_2 = new_state_2
            if done == True:
                break
            
        else:
            #system.render_obj.render_pause()  # no rendering for faster
            time.sleep(0.001) #time.sleep(1)
    system.data_manager.save_file(episode_cnt=episode_cnt)
    system.reset()
    print("ep_num: "+str(episode_cnt)+"   ep_score:  "+str(score_per_ep))
    episode_cnt +=1
    if episode_cnt % 10 == 0:
        agent_1.save_models(best=False)
        agent_2.save_models(best=False)
    if best < score_per_ep:
        agent_1.save_models(best=True)
        agent_2.save_models(best=True)
        best = score_per_ep
