from ddpg import Agent
from env1 import minimal_IRS_system # orginal
from env import MiniSystem
import numpy as np
#from utils import plotLearning
import matplotlib.pyplot as plt
import os
from data_manager import DataManager
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

IRS_system = minimal_IRS_system(K = 1)
#IRS_system = MiniSystem()
K = IRS_system.K
M = IRS_system.M
N = IRS_system.N
RL_state_dims = 2*K + 2*K**2 + 2*N + 2*M*K + 2*N*M + 2*K*N
RL_input_dims = RL_state_dims
RL_action_dims = 2 * (M * K) + N

steps_per_ep = 200
alpha_actor_learning_rate = 0.001
beta_critic_learning_rate = 0.001
agent = Agent(alpha=alpha_actor_learning_rate, beta=beta_critic_learning_rate, input_dims=[RL_input_dims], tau=0.001, env=IRS_system,
              batch_size=64,  layer1_size=400 * 2, layer2_size=300 * 2, n_actions=RL_action_dims)

scores = []
for i in range(1000):
    observersion = IRS_system.reset()
    done = False
    done_sys = False
    score = 0
    cnt_in_one_epi = 0
    best_bit_per_Hz = 0
    draw_bit_rate_list = []
    draw_tran_power_list = []
    #draw_bit_rate_one_element = {'if_exceed_max_power':False,'bit_rate' : 0}
    while not done:
        cnt_in_one_epi += 1
        if cnt_in_one_epi > 500:
            done = True
        action = agent.choose_action(observersion)
        new_state, reward, done_sys , info = IRS_system.step(action)

        bit_per_Hz = IRS_system.calculate_data_rate()   
        #draw_bit_rate_one_element['bit_rate'] = bit_per_Hz 
        #draw_bit_rate_one_element['if_exceed_max_power']=done_sys
        draw_bit_rate_list.append(bit_per_Hz)

        total_power = IRS_system.calculate_total_transmit_power()
        draw_tran_power_list.append(total_power)
        if done_sys == False:# if not exceed max transmit power            
            if bit_per_Hz > best_bit_per_Hz:
                best_bit_per_Hz = bit_per_Hz
        agent.remember(observersion, action, reward, new_state, int(done))
        agent.learn()
        score += reward
        observersion = new_state
        IRS_system.render()
    plt.cla()
    plt.plot(range(len(draw_bit_rate_list)), draw_bit_rate_list, color = 'green')
    plt.plot(range(len(draw_tran_power_list)), draw_tran_power_list, color = 'red')

    # plt.show()
    filename_i =os.path.abspath(os.curdir) + '\\main_foder\\image_result\\' + str(i) + '.png'
    plt.savefig(filename_i)
    scores.append(score)
    #if i % 25 == 0:
        #agent.save_models()

    print('episode ', i, 'score %.2f' % score, 'best sum rate %.3f bit/s/Hz' % best_bit_per_Hz,
          'trailing 100 games avg %.4f' % np.mean(scores[-100:]))
filename = 'C:\\demo\\IRS_DDPG_minimal\\main_foder\\LunarLander-alpha000025-beta00025-400-300.png'
# plotLearning(scores, filename, window=100)
