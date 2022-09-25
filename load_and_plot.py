import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.io import loadmat, savemat
import os
# load mat
class LoadAndPlot(object):
    """
    load date and plot 2022-07-22 16_16_26
    """
    def __init__(self, store_path = '/home/tham/Desktop/uav-td3/data/storage/2022-09-08 19_57_43/', \
                       user_num = 2, attacker_num = 1, RIS_ant_num = 4, \
                       ep_num = 100, step_num = 100): # RIS_ant_num = 16 (not true)

        self.color_list = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
        self.store_path = store_path
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

        fig = plt.figure('reward')
        plt.plot(range(len(self.all_steps['reward'])), self.all_steps['reward'])
        plt.xlabel("Time Steps ($t$)")
        plt.ylabel("Reward")
        plt.savefig(self.store_path + 'plot/reward.png')
        plt.cla()

        fig = plt.figure('secure_capacity')
        for i in range(self.user_num):
            plt.plot(range(len(self.all_steps['secure_capacity'][i])), self.all_steps['secure_capacity'][i], c=color_list[i])
        plt.legend(['user_' + str(i) for i in range(self.user_num)])
        plt.xlabel("Time Steps ($t$)")
        plt.ylabel("Secure Capacity")
        plt.savefig(self.store_path + 'plot/secure_capacity.png')
        plt.cla()

        fig = plt.figure('average_sum_secrecy_rate')
        sum_secrecy_rate = np.array(self.all_steps['secure_capacity'])
        sum_secrecy_rate = np.sum(sum_secrecy_rate, axis = 0)
        average_sum_secrecy_rate = []
        for i in range(0, self.ep_num * self.step_num, self.step_num):
            ssr_one_episode = sum_secrecy_rate[i:i+self.step_num] # ssr means Sum Secrecy Rate
            _ = sum(ssr_one_episode) / len(ssr_one_episode) 
            average_sum_secrecy_rate.append(_)
        plt.plot(range(len(average_sum_secrecy_rate)), average_sum_secrecy_rate)
        plt.xlabel("Episodes ($Ep$)")
        plt.ylabel("Average Sum Secrecy Rate")
        plt.savefig(self.store_path + 'plot/average_sum_secrecy_rate.png')
        plt.cla()

        fig = plt.figure('user_capacity')
        for i in range(self.user_num):
            plt.plot(range(len(self.all_steps['user_capacity'][i])), self.all_steps['user_capacity'][i], c=color_list[i])
        plt.legend(['user_' + str(i) for i in range(self.user_num)])
        plt.xlabel("Time Steps ($t$)")
        plt.ylabel("User Capacity")
        plt.savefig(self.store_path + 'plot/user_capacity.png')
        plt.cla()

        fig = plt.figure('attaker_capacity')
        for i in range(self.attacker_num):
            plt.plot(range(len(self.all_steps['attaker_capacity'][i])), self.all_steps['attaker_capacity'][i], c=color_list[i])
        plt.legend(['attacker_' + str(i) for i in range(self.attacker_num)])
        plt.xlabel("Time Steps ($t$)")
        plt.ylabel("Attack Capacity")
        plt.savefig(self.store_path + 'plot/attaker_capacity.png')
        plt.close('all')

        for i in range(self.RIS_ant_num):
            self.plot_one_RIS_element(i)
    
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
    def restruct(self):
        savemat(self.store_path + 'all_steps.mat',self.all_steps)
        return 0
if __name__ == '__main__':
    LoadPlotObject = LoadAndPlot(
       # store_path = "./paper/my/plot/compare/2020-12-06 15_35_34_with_RIS_16/",
       # store_path = 'D:\Drive 3\Tham post detail UAV VS RIS\Learning-Based Robust and Secure Transmission for Reconfigurable Intelligent Surface Aided Millimeter Wave UAV Communications\WCL-pulish-code-master\data\storage\2022-06-17 11_31_30\plot\RIS',
        # user_num=2,
        # RIS_ant_num = 4
        )
    LoadPlotObject.plot()
    LoadPlotObject.restruct()

    

