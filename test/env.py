import numpy as np
import math
import cmath
np.random.seed(42)

class minimal_IRS_system():
    def __init__(self, BS_M = 8, IRS_N_x = 4, IRS_N_y =2, K = 8, statistic = False):
        self.M = BS_M  # number of BS antennas
        self.N = IRS_N_x * IRS_N_y  # number of IRS elements
        self.K = K  # number of users
        self.z_BS = 0 # hight of BS antenas
        self.BS_rotation = 0 #
        self.BS_coordinate = np.array([0, 0, 30])
        self.BS_elevation_angle = 0 
        self.BS_max_power = 1000 # BS power constrain, W

        self.BS_normal_vecter = np.array([math.cos(self.BS_rotation), math.sin(self.BS_rotation), math.sin(self.BS_elevation_angle)*np.linalg.norm([-math.cos(self.BS_rotation), math.sin(self.BS_rotation)])])
        self.BS_normal_vecter = self.BS_normal_vecter / np.linalg.norm(self.BS_normal_vecter)

        # channel parameters class
        class channel_parameters():
            """
            channel parameters
            """
            def __init__(self):
                self.noise_dBm = noise_dBm = -114       # channel noise, -114 dBm
                self.noise_segma = 0                    # channel noise std,
                self.d_0 = 1                            # path loss referance distance 1 m
                self.rho_0 = 0.01                       # path loss parameter
                self.alpha_BS_to_IRS = 3                       # path loss exponent
                self.alpha_IRS_to_user = 2.5                      # path loss exponent
                self.alpha_BS_to_user = 3.5                      # path loss exponent
                self.K_BS = math.pow(10, 3/10)          # rician factor, 3dB
                self.K_IRS = math.pow(10, 3/10)         # rician factor, 3dB
        self.channel_parameters = channel_parameters()
        self.channel_parameters.noise_segma = math.pow(self.dBm_to_W(self.channel_parameters.noise_dBm),0.5)

        # IRS initial
        class IRS():
            """
            store user parameters
            """
            def __init__(self):
                self.IRS_N_x = IRS_N_x
                self.IRS_N_y = IRS_N_y
                self.IRS_N = IRS_N_x * IRS_N_y
                self.x_IRS = 1000                       # position x of IRS, 1000 m
                self.y_IRS = 0                          # position y of IRS, 0 m
                self.z_IRS = 25                          # position z of IRS, 0 m
                self.IRS_coordinate = np.array([self.x_IRS,self.y_IRS,self.z_IRS])
                self.deltaD_to_lambda = 0.5
                self.d_BS_to_IRS = np.linalg.norm(self.IRS_coordinate)
                self.theta_IRS = math.atan(self.y_IRS / self.x_IRS) # theta_IRS
                self.psi_IRS = 0                        # rotation of IRS , range: -pi/2 + theta_IRS  ~  pi/2 - theta_IRS
                self.elevation_angle = 0                # elevation angle

                self.IRS_normal_vecter = np.array([-math.cos(self.psi_IRS), math.sin(self.psi_IRS),math.sin(self.elevation_angle)*np.linalg.norm([-math.cos(self.psi_IRS), math.sin(self.psi_IRS)])])
                self.IRS_normal_vecter = self.IRS_normal_vecter / np.linalg.norm(self.IRS_normal_vecter)
                self.theta_BS_to_IRS = 0
                self.psi_BS_to_IRS = 0
                self.theta_IRS_to_BS = 0
                self.psi_IRS_to_BS = 0
                
        self.IRS = IRS()
        
        # users initial
        class user():
            """
            store user parameters
            """
            def __init__(self):
                self.user_index = 0                     # user ID
                self.distance_to_IRS = 10               # distance to IRS(need caculation), range of (10, 30)
                self.psi = 0                            # psi angle to IRS
                self.psi_IRS_surface = 0                # psi used to calculate the UPA response
                self.x_user = 10                        # position x 
                self.y_user = 10                        # position y
                self.z_user = 0                         # position z
                self.user_coordinate = np.array([self.x_user,self.y_user,self.z_user])
                self.distance_to_BS = np.linalg.norm(self.user_coordinate)                 # distance to BS
                self.theta_user = 0                     # theta to BS
                self.theta_IRS_to_kth_user = 0
                self.psi_IRS_to_kth_user = 0
                
        mini_distance = 10                                                 # minimal distance for users to IRS, m
        max_distance = 30                                                  # maxmial distance for users to IRS, m
        statistic_distance_list = [10, 12, 14, 17, 20, 23, 25, 29]
        statistic_angle_list = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
        stocastic_distance_list = np.random.uniform(mini_distance, max_distance, self.K)    # list of dis
        stocastic_angle_list = np.random.uniform(- math.pi / 2, math.pi / 2, self.K)                      # list of angle
        self.user_list = []
        for index_user in range(K):
            user_temp = user()
            user_temp.user_index = index_user
            if statistic == True:
                user_temp.distance_to_IRS = statistic_distance_list[index_user]
                user_temp.psi = statistic_angle_list[index_user]
                user_temp.x_user = self.IRS.x_IRS - user_temp.distance_to_IRS * math.cos(user_temp.psi)
                user_temp.y_user = self.IRS.y_IRS + user_temp.distance_to_IRS * math.sin(user_temp.psi)
            else:
                user_temp.distance_to_IRS = stocastic_distance_list[index_user]
                user_temp.psi = stocastic_angle_list[index_user]
                user_temp.x_user = self.IRS.x_IRS - user_temp.distance_to_IRS * math.cos(user_temp.psi)
                user_temp.y_user = self.IRS.y_IRS + user_temp.distance_to_IRS * math.sin(user_temp.psi)
            user_temp.psi_IRS_surface = user_temp.psi - self.IRS.psi_IRS
            #user_temp.distance_to_BS = np.linalg.norm([user_temp.x_user, user_temp.y_user])
            user_temp.theta_user = math.atan(user_temp.y_user / user_temp.x_user )
            user_temp.user_coordinate = np.array([user_temp.x_user,user_temp.y_user,user_temp.z_user])
            self.user_list.append(user_temp)
            

        # transmit signal
        self.X_transmit = np.mat(np.ones((K, 1)))       # BS transmit signal, unit variance, shape: K X 1

        # active beamforming at BS
        self.G_beamforming = np.mat(np.ones((self.M, K),dtype=complex))    # BS beamforming matrix, shape: M X K
        self.BS_max_power = self.calculate_total_transmit_power()
        self.reset_G = np.mat(np.ones((self.M, K),dtype=complex))

        # channal gain from BS to IRS
        #self.H_BS_to_IRS = np.mat(np.ones((self.N, self.M)))      # channel gain BS to IRS, shape: N X M
        self.H_BS_to_IRS = self.calculate_channel_BS_to_IRS(self.IRS)

        # channal gain from IRS to users
        # self.H_IRS_to_user = np.mat(np.ones((K, self.N),dtype=complex))    # channel gain IRS to user, shape: K X N
        self.H_IRS_to_user = self.calculate_channel_IRS_to_user(self.IRS)

        # IRS phase shifter parameter
        self.Fai = np.mat(np.identity(self.N), dtype=complex)               # IRS phase shift 
        self.reset_Fai = np.mat(np.identity(self.N), dtype=complex)
        # data rate matrex container
        self.data_rate = np.mat(np.ones((K,1)))         # channal capacities for all K users

        # calculate sum of all user data rate
        self.sum_of_user_data_rates = self.calculate_data_rate()

    def calculate_total_transmit_power(self):
        """
        calculate total power
        """
        G = self.G_beamforming
        G_H = G.H
        temp_power = 0
        for m in range(self.M):
            temp_power += G[m,:] * G_H[:, m]
        return float(np.real(temp_power[0,0]))

    def dBm_to_W(self, dBm):
        mW = math.pow(10, dBm/10)
        W = mW/1000.0
        return W

    def generate_noise(self, mu, segma, size = 1):
        """
        generates noise of variance segma ^ 2
        size can be a tuple like (3,3)

        output :    a float of noise
                    or an array with size size of average mu and std of segma
        """
        result = np.random.normal(mu, segma, size)
        if size == 1:
            return float(result)
        else:
            return result

    def calculate_received_signal_amplitude(self):
        """
        calculate_received_signal_amplitude from BS to user

        output :    a matix , size: K X 1
        """   
        noise_array = self.generate_noise(0, self.channel_parameters.noise_segma, (self.K,1))
        noise_matrix = np.mat(noise_array)

        Y_matrix = self.H_IRS_to_user * self.Fai * self.H_BS_to_IRS * self.G_beamforming * self.X_transmit + noise_matrix
        return Y_matrix

    def calculate_SINR_of_kth_user(self, k):
        """
        calculate kth user's SINR

        output  :   a float
        """
        expected_amplitude = abs(self.H_IRS_to_user[k] * self.Fai * self.H_BS_to_IRS * self.G_beamforming[:, k])
        expected_power = math.pow(expected_amplitude, 2)

        noise_power = 0
        noise_power = noise_power + math.pow(self.channel_parameters.noise_segma, 2)
        for n in range(self.K):
            if n != k:
                noise_power = noise_power + math.pow(abs(self.H_IRS_to_user[n] * self.Fai * self.H_BS_to_IRS * self.G_beamforming[:, n]),2)
        return expected_power/noise_power

    def calculate_data_rate(self):
        """
        calculate K data rates

        output :    sum of datarate
        """
        for kth in range(self.K):
            self.data_rate[kth][0] = math.log2(1 + self.calculate_SINR_of_kth_user(kth))
        return float(self.data_rate.sum())

    def calculate_theta_and_psi(self, ref_norm_vector, coordinate):
        """
        calculate_theta_and_psi
        """
        theta_ref_to_coor =  math.acos(abs(np.dot(coordinate/ np.linalg.norm(coordinate),ref_norm_vector)))
        coordinate_xoy = coordinate - abs(math.cos(theta_ref_to_coor)) * np.linalg.norm(coordinate) * ref_norm_vector
        temp_x_vector = [ref_norm_vector[1], -ref_norm_vector[0], 0]
        psi_ref_to_coor = abs(np.dot(temp_x_vector, coordinate_xoy)) / (np.linalg.norm(temp_x_vector) * np.linalg.norm(coordinate_xoy))
        if math.isnan(psi_ref_to_coor):
            psi_ref_to_coor = 0
        
        return theta_ref_to_coor, psi_ref_to_coor
    def calculate_channel_BS_to_IRS(self, IRS): # IRS elements spacing half wavelength
        """
        denate the self.H_BS_to_IRS (N X M)
        """
        temp_vertical_matrix = np.zeros((IRS.IRS_N,1), dtype=complex)
        temp_horizontal_matrix = np.zeros((1, self.M), dtype=complex)
        """
        theta_BS_to_IRS = math.acos(np.dot(IRS.IRS_coordinate/ np.linalg.norm(IRS.IRS_coordinate),self.BS_normal_vecter))
        IRS_xoy = IRS.IRS_coordinate - abs(math.cos(theta_BS_to_IRS)) * np.linalg.norm(IRS.IRS_coordinate) * self.BS_normal_vecter
        temp_x_BS = [self.BS_normal_vecter[1], -self.BS_normal_vecter[0], 0]
        psi_BS_to_IRS = abs(np.dot(temp_x_BS, IRS_xoy)) / (np.linalg.norm(temp_x_BS) * np.linalg.norm(IRS_xoy))
        if math.isnan(psi_BS_to_IRS):
            psi_BS_to_IRS = 0
        
        theta_IRS_to_BS = math.acos(abs(np.dot(IRS.IRS_coordinate/ np.linalg.norm(IRS.IRS_coordinate), IRS.IRS_normal_vecter)))
        BS_xoy = -1 * np.array(IRS.IRS_coordinate) - abs(math.cos(theta_IRS_to_BS)) * np.linalg.norm(IRS.IRS_coordinate) * IRS.IRS_normal_vecter
        temp_x_IRS = [IRS.IRS_normal_vecter[1], -IRS.IRS_normal_vecter[0], 0]
        psi_IRS_to_BS = abs(np.dot(temp_x_IRS, BS_xoy)) / (np.linalg.norm(temp_x_IRS) * np.linalg.norm(BS_xoy))
        if math.isnan(psi_IRS_to_BS):
            psi_IRS_to_BS = 0
        # old way to calculate theta and psi
        """
        theta_BS_to_IRS, psi_BS_to_IRS = self.calculate_theta_and_psi(self.BS_normal_vecter, IRS.IRS_coordinate - self.BS_coordinate)
        theta_IRS_to_BS, psi_IRS_to_BS = self.calculate_theta_and_psi(IRS.IRS_normal_vecter, self.BS_coordinate - IRS.IRS_coordinate)
        IRS.theta_BS_to_IRS = theta_BS_to_IRS
        IRS.psi_BS_to_IRS = psi_BS_to_IRS
        IRS.theta_IRS_to_BS = theta_IRS_to_BS
        IRS.psi_IRS_to_BS = psi_IRS_to_BS

        for i in range(IRS.IRS_N_y):
            for j in range(IRS.IRS_N_x):
                temp_vertical_matrix[i * IRS.IRS_N_x + j][0] = \
                cmath.exp(-1j * 2 * math.pi * IRS.deltaD_to_lambda * \
                (i * math.sin(theta_IRS_to_BS)* math.cos(psi_IRS_to_BS)+ \
                j * math.sin(theta_IRS_to_BS)* math.sin(psi_IRS_to_BS)) \
                )
        for i in range(self.M):
            temp_horizontal_matrix[0][i] = \
            cmath.exp(-1j * 2 * math.pi * IRS.deltaD_to_lambda * \
            (i * math.sin(theta_BS_to_IRS)* math.cos(psi_BS_to_IRS)) \
            )

        temp_horizontal_matrix = np.mat(temp_horizontal_matrix)
        temp_vertical_matrix = np.mat(temp_vertical_matrix)
        H_LOS = temp_vertical_matrix * temp_horizontal_matrix

        H = math.pow(self.channel_parameters.rho_0* (math.pow(IRS.d_BS_to_IRS/self.channel_parameters.d_0, - self.channel_parameters.alpha_BS_to_IRS)), 0.5) \
        * (H_LOS)
        return H
        
    def calculate_channel_IRS_to_user(self, IRS):
        """
        channel gain IRS to user, shape: K X N
        """
        temp_array = np.zeros((self.K, self.N), dtype=complex)

        for kth in range(self.K):
            theta_IRS_to_kth_user, psi_IRS_to_kth_user = self.calculate_theta_and_psi(IRS.IRS_normal_vecter, self.user_list[kth].user_coordinate - IRS.IRS_coordinate)
            self.user_list[kth].theta_IRS_to_kth_user = theta_IRS_to_kth_user
            self.user_list[kth].psi_IRS_to_kth_user = psi_IRS_to_kth_user
            for i in range(IRS.IRS_N_y):
                for j in range(IRS.IRS_N_x):
                    temp_array[kth][i * IRS.IRS_N_x + j] = \
                    cmath.exp(-1j * 2 * math.pi * IRS.deltaD_to_lambda * \
                    (i * math.sin(theta_IRS_to_kth_user)* math.cos(psi_IRS_to_kth_user)+ \
                    j * math.sin(theta_IRS_to_kth_user)* math.sin(psi_IRS_to_kth_user)) \
                    )
        H_LOS = temp_array
        for kth in range(self.K):
            H_LOS[kth] = math.pow(self.channel_parameters.rho_0* (math.pow(np.linalg.norm(IRS.IRS_coordinate - self.user_list[kth].user_coordinate)/self.channel_parameters.d_0, - self.channel_parameters.alpha_BS_to_IRS)), 0.5) \
            * (H_LOS[kth])
        
        H = np.mat(H_LOS)
        return H

    def reset(self):
        """
        RL implement, 
        """
        # 1 reset all parameters
        self.G_beamforming = self.reset_G
        self.Fai = self.reset_Fai
        # 2 get state
        return self.get_state()

    def apply_action(self, action):
        """
        apply action
        """
        # 0 parameters and all kinds of matrix
        K = self.K
        M = self.M
        N = self.N
        # 1 apply action
        # 1.1 divide action list into two parts
        # 1.1.1 beamforming part, MXK
        action_beamforming = action[0 : 2*M*K]
        # 1.1.2 IRS reflecting coinfetions
        action_Fai = action[2*M*K : 2*M*K + 2*N]

        # 1.2 apply action
        # 1.2.1 beamforming part, MXK
        for m in range(M):
            for k in range(K):
                index = m * K + k
                temp_Gmk = action_beamforming[2*index] + 1j * action_beamforming[2*index + 1]
                self.G_beamforming[m, k] = temp_Gmk
        # 1.2.2 IRS reflecting coinfetions
        for n in range(N):
            temp_Fainn = action_Fai[2*n] + 1j * action_Fai[2*n + 1]
            self.Fai[n, n] = temp_Fainn

        return True
    def get_state(self):
        """
        get current state
        """
        # 0 parameters and all kinds of matrix
        K = self.K
        M = self.M
        N = self.N
        new_state_dims = 2*K + 2*K**2 + 2*N + 2*M*K + 2*N*M + 2*K*N
        
        # 1 get new state
        new_state = []
        G = self.G_beamforming
        G_H = self.G_beamforming.H
        H_IRS_to_user = self.H_IRS_to_user
        Fai = self.Fai
        H_BS_to_IRS = self.H_BS_to_IRS
        # 1.1 transmit power to k users, 2K
        state_tran_power_BS_to_users = [] # result container
        for kth in range(K):
            state_tran_power_BS_to_users.append(\
            math.pow(float(np.real(G_H[kth,:] * G[:,kth])),2))
            state_tran_power_BS_to_users.append(\
            math.pow(float(np.imag(G_H[kth,:] * G[:,kth])),2))
        
        # 1.2 received power for users, 2K^2
        state_action_receive_power_for_users = []
        for k1 in range(K):
            user_k1_received_power_from_all_other_users = \
            H_IRS_to_user[k1, :] * Fai * H_BS_to_IRS * G
            for k2 in range(K):
                state_action_receive_power_for_users.append(\
                math.pow(np.real(user_k1_received_power_from_all_other_users[0, k2]), 2))
                state_action_receive_power_for_users.append(\
                math.pow(np.imag(user_k1_received_power_from_all_other_users[0, k2]), 2))
                
        # 1.3 beamforming at BS, 2*M*K+2*N
        state_beamforming_at_BS = []
        for m in range(M):
            for k in range(K):
                state_beamforming_at_BS.append(\
                math.pow(np.real(G[m, k]), 2))
                state_beamforming_at_BS.append(\
                math.pow(np.imag(G[m, k]), 2))
        for n in range(N):
            state_beamforming_at_BS.append(\
            math.pow(np.real(Fai[n, n]), 2))
            state_beamforming_at_BS.append(\
            math.pow(np.imag(Fai[n, n]), 2))

        # 1.4 channel state at both BS-IRS & IRS-users, 2*N*M+2*K*M
        state_channel_BS_IRS_and_IRS_users = []
        for n in range(N):
            for m in range(M):
                state_channel_BS_IRS_and_IRS_users.append(\
                math.pow(np.real(H_BS_to_IRS[n, m]), 2))
                state_channel_BS_IRS_and_IRS_users.append(\
                math.pow(np.imag(H_BS_to_IRS[n, m]), 2))
        for k in range(K):
            for n in range(N):
                state_channel_BS_IRS_and_IRS_users.append(\
                math.pow(np.real(H_IRS_to_user[k, n]), 2))
                state_channel_BS_IRS_and_IRS_users.append(\
                math.pow(np.imag(H_IRS_to_user[k, n]), 2))
        # 1.5 finnish getting state
        new_state.append(state_tran_power_BS_to_users)
        new_state.append(state_action_receive_power_for_users)
        new_state.append(state_beamforming_at_BS)
        new_state.append(state_channel_BS_IRS_and_IRS_users)

        result = state_tran_power_BS_to_users + \
        state_action_receive_power_for_users + \
        state_beamforming_at_BS + \
        state_channel_BS_IRS_and_IRS_users
        return result
        

    def step(self, action):
        """
        RL function ,state is a 1 X (2K+2K^2+2N+2MK+2NM+2KN) vecter
        """
        done = False
        # 1 get new state
        new_state = self.get_state()
        # 2 apply action 
        self.apply_action(action)
        # 3 judge if done
        total_power = self.calculate_total_transmit_power()
        if total_power > self.BS_max_power:
            done = True
        # 4 get other reward
        reward = 0
        if total_power > self.BS_max_power:
            reward = np.real(self.BS_max_power - total_power)
            # reward = np.real(self.BS_max_power[0,0] - total_power[0,0] )/1000
        else:
            reward = self.calculate_data_rate()*100

        info = 0
        return new_state, reward, done, info

    def render(self):
        """
        show system function
        """


        return True
     