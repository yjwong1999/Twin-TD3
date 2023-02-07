import os
import torch as T
#import torch.cuda as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_manager import DataManager
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class AWGNActionNoise(object):
    def __init__(self, mu = 0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        #self.mu = mu
        #self.sigma = sigma
        x = np.random.normal(size=self.mu.shape) * self.sigma
        return x

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, n_actions, name,
                 chkpt_dir='C:\\demo\\IRS_TD3_minimal\\main_foder\\tmp\\TD3', load_file = ''):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_TD3')
        self.best_checkpoint_file = os.path.join(chkpt_dir,'best', name+'_TD3')
        self.load_file = 'C:\\demo\\other_branch\\Learning-based_Secure_Transmission_for_RIS_Aided_mmWave-UAV_Communications_with_Imperfect_CSI\\data\\mannal_store\\models\\Critic_UAV_TD3'
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        #self.fc1.weight.data.uniform_(-f1, f1)
        #self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        #f2 = 0.002
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        #self.fc2.weight.data.uniform_(-f2, f2)
        #self.fc2.bias.data.uniform_(-f2, f2)
        self.bn4 = nn.LayerNorm(self.fc4_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc4_dims)
        f5 = 0.003
        self.q = nn.Linear(self.fc4_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f5, f5)
        T.nn.init.uniform_(self.q.bias.data, -f5, f5)
        #self.q.weight.data.uniform_(-f3, f3)
        #self.q.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
#        if torch.cuda.available():
#            import torch.cuda as T
#        else:
#            import torch as T
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc3(state_value)
        state_value = self.bn3(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc4(state_value)
        state_value = self.bn4(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self, best=True):
        if best:
            print('... saving best checkpoint ...')
            T.save(self.state_dict(), self.best_checkpoint_file)
        else:
            print('... saving checkpoint ...')
            T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self,load_file = ''):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(load_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, n_actions, name,
                 chkpt_dir='C:\\demo\\IRS_TD3_minimal\\main_foder\\tmp\\TD3', load_file = ''):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims        
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_TD3')
        self.best_checkpoint_file = os.path.join(chkpt_dir,'best', name+'_TD3')
        self.load_file = 'C:\\demo\\other_branch\\Learning-based_Secure_Transmission_for_RIS_Aided_mmWave-UAV_Communications_with_Imperfect_CSI\\data\\mannal_store\\models\\Actor_UAV_TD3'
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
#        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
#        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        #f2 = 0.002
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
#        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
#        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
#        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
#        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
#        T.nn.init.uniform_(self.fc4.weight.data, -f4, f4)
#        T.nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        self.bn4 = nn.LayerNorm(self.fc4_dims)

        #f3 = 0.004
        f5 = 0.003
        self.mu = nn.Linear(self.fc4_dims, self.n_actions)
#        T.nn.init.uniform_(self.mu.weight.data, -f5, f5)
#        T.nn.init.uniform_(self.mu.bias.data, -f5, f5)
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self, best=True):
        if best:
            print('... saving best checkpoint ...')
            T.save(self.state_dict(), self.best_checkpoint_file)
        else:
            print('... saving checkpoint ...')
            T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, load_file=''):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(load_file))

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, layer3_size=256, layer4_size=128, batch_size=64, 
                 update_actor_interval=2, noise = 'AWGN', agent_name = 'default', load_file = ''):
        self.load_file = load_file
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.layer3_size = layer3_size
        self.layer4_size = layer4_size
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                  name='Actor_' + agent_name,chkpt_dir=env.data_manager.store_path )
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                    name='Critic_1_' + agent_name,chkpt_dir=env.data_manager.store_path)
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                    name='Critic_2_' + agent_name,chkpt_dir=env.data_manager.store_path)

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                         name='TargetActor_' + agent_name,chkpt_dir=env.data_manager.store_path)
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                           name='TargetCritic_1_' + agent_name,chkpt_dir=env.data_manager.store_path)
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, layer3_size, layer4_size, n_actions=n_actions,
                                           name='TargetCritic_2_' + agent_name,chkpt_dir=env.data_manager.store_path)
        if noise == 'OU':
            self.noise = OUActionNoise(mu=np.zeros(n_actions))
        elif noise == 'AWGN':
            self.noise = AWGNActionNoise(mu = np.zeros(n_actions))
        # tau = 1 means copy parameters to target
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, greedy=0.5, epsilon = 1):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(greedy * self.noise(),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        # the done here is opposite of the done in the env
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)

        # trun s, a, r, new_s into tensor
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)

        # trun on evaliation mode of target actor, target critic, critic net
        # fix these three nets
        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        self.critic_1.eval()
        self.critic_2.eval()

        
        target_actions = self.target_actor.forward(new_state)
        # target_actions = target_actions + \
        #         T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        # # might break if elements of min and max are not all equal
        # target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])
        
        critic_value_1_ = self.target_critic_1.forward(new_state, target_actions)
        critic_value_2_ = self.target_critic_2.forward(new_state, target_actions)
        
        critic_value_1 = self.critic_1.forward(state, action)
        critic_value_2 = self.critic_2.forward(state, action)
        
        critic_value_ = T.min(critic_value_1_, critic_value_2_)
        
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic_1.device)
        target = target.view(self.batch_size, 1)
        
        # here update the critic net using mse of (r + gamma * Q_argmax_a*(newstate, a*)) - Q(state, action)
        self.critic_1.train()
        self.critic_2.train()
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        
        critic_1_loss = F.mse_loss(target, critic_value_1)
        critic_2_loss = F.mse_loss(target, critic_value_2)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.learn_step_cntr += 1
        
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        # here update the actor net by policy gradient
        # first fix the critic net
        self.critic_1.eval()
        self.critic_2.eval()
        
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_q1_loss = self.critic_1.forward(state, mu)
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
        

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()
                    
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)


    def save_models(self, best=True):
        self.actor.save_checkpoint(best=best)
        self.target_actor.save_checkpoint(best=best)
        self.critic_1.save_checkpoint(best=best)
        self.critic_2.save_checkpoint(best=best)
        self.target_critic_1.save_checkpoint(best=best)
        self.target_critic_2.save_checkpoint(best=best)

    def load_models(self, load_file_actor = '',load_file_critic_1 ='',load_file_critic_2 =''):
        self.actor.load_checkpoint(load_file = load_file_actor)
        self.target_actor.load_checkpoint(load_file = load_file_actor)
        self.critic_1.load_checkpoint(load_file = load_file_critic_1)
        self.critic_2.load_checkpoint(load_file = load_file_critic_2)
        self.target_critic_1.load_checkpoint(load_file = load_file_critic_1)
        self.target_critic_2.load_checkpoint(load_file = load_file_critic_2)
        
'''
class TwinCriticNetwork(nn.Module):
    def __init__(self, beta, dims, n_actions, name,
                 chkpt_dir='C:\\demo\\IRS_TD3_minimal\\main_foder\\tmp\\TD3', load_file = ''):
        super(TwinCriticNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_TD3')
        self.best_checkpoint_file = os.path.join(chkpt_dir,'best', name+'_TD3')
        self.load_file = 'C:\\demo\\other_branch\\Learning-based_Secure_Transmission_for_RIS_Aided_mmWave-UAV_Communications_with_Imperfect_CSI\\data\\mannal_store\\models\\Critic_UAV_TD3'
        
        ###################################################
        # agent 1
        ###################################################
        input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims = dims[0]
        mid_dims = fc2_dims + dims[1][2] # for cconcat
        n_action = n_actions[0]
        
        self.agent1_fc1 = nn.Linear(*input_dims, fc1_dims)
        f1 = 1./np.sqrt(self.agent1_fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.agent1_fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.agent1_fc1.bias.data, -f1, f1)
        self.agent1_bn1 = nn.LayerNorm(fc1_dims)

        self.agent1_fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1./np.sqrt(self.agent1_fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.agent1_fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.agent1_fc2.bias.data, -f2, f2)
        self.agent1_bn2 = nn.LayerNorm(fc2_dims)

        self.agent1_fc3 = nn.Linear(mid_dims, fc3_dims)
        f3 = 1./np.sqrt(self.agent1_fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.agent1_fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.agent1_fc3.bias.data, -f3, f3)
        self.agent1_bn3 = nn.LayerNorm(fc3_dims)

        self.agent1_fc4 = nn.Linear(fc3_dims, fc4_dims)
        f4 = 1./np.sqrt(self.agent1_fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.agent1_fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.agent1_fc4.bias.data, -f4, f4)
        self.agent1_bn4 = nn.LayerNorm(fc4_dims)

        self.agent1_action_value = nn.Linear(n_action, fc4_dims)
        f5 = 0.003
        self.agent1_q = nn.Linear(fc4_dims, 1)
        T.nn.init.uniform_(self.agent1_q.weight.data, -f5, f5)
        T.nn.init.uniform_(self.agent1_q.bias.data, -f5, f5)
        
        ###################################################
        # agent 2
        ###################################################
        input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims = dims[1]
        mid_dims = fc2_dims + dims[0][2] # for concat
        n_action = n_actions[1]
        
        self.agent2_fc1 = nn.Linear(*input_dims, fc1_dims)
        f1 = 1./np.sqrt(self.agent2_fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.agent2_fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.agent2_fc1.bias.data, -f1, f1)
        self.agent2_bn1 = nn.LayerNorm(fc1_dims)

        self.agent2_fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1./np.sqrt(self.agent2_fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.agent2_fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.agent2_fc2.bias.data, -f2, f2)
        self.agent2_bn2 = nn.LayerNorm(fc2_dims)

        self.agent2_fc3 = nn.Linear(mid_dims, fc3_dims)
        f3 = 1./np.sqrt(self.agent2_fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.agent2_fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.agent2_fc3.bias.data, -f3, f3)
        self.agent2_bn3 = nn.LayerNorm(fc3_dims)

        self.agent2_fc4 = nn.Linear(fc3_dims, fc4_dims)
        f4 = 1./np.sqrt(self.agent2_fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.agent2_fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.agent2_fc4.bias.data, -f4, f4)
        self.agent2_bn4 = nn.LayerNorm(fc4_dims)

        self.agent2_action_value = nn.Linear(n_action, fc4_dims)
        f5 = 0.003
        self.agent2_q = nn.Linear(fc4_dims, 1)
        T.nn.init.uniform_(self.agent2_q.weight.data, -f5, f5)
        T.nn.init.uniform_(self.agent2_q.bias.data, -f5, f5)

        
        
        ###################################################
        # remaining things
        ###################################################        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
#        if torch.cuda.available():
#            import torch.cuda as T
#        else:
#            import torch as T
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, states, actions):
        # unpack
        state1, state2 = states
        action1, action2 = actions
        
        # agent 1
        state_value1 = self.agent1_fc1(state1)
        state_value1 = self.agent1_bn1(state_value1)
        state_value1 = F.relu(state_value1)
        state_value1 = self.agent1_fc2(state_value1)
        state_value1 = self.agent1_bn2(state_value1)
        
        # agent 2
        state_value2 = self.agent2_fc1(state2)
        state_value2 = self.agent2_bn1(state_value2)
        state_value2 = F.relu(state_value2)
        state_value2 = self.agent2_fc2(state_value2)
        state_value2 = self.agent2_bn2(state_value2)
        
        # merge
        state_value = T.cat((state_value1, state_value2), axis=1)
        
        # agent 1
        state_value1 = F.relu(state_value)
        state_value1 = self.agent1_fc3(state_value1)
        state_value1 = self.agent1_bn3(state_value1)
        state_value1 = F.relu(state_value1)
        state_value1 = self.agent1_fc4(state_value1)
        state_value1 = self.agent1_bn4(state_value1)

        action_value1 = F.relu(self.agent1_action_value(action1))
        state_action_value1 = F.relu(T.add(state_value1, action_value1))
        state_action_value1 = self.agent1_q(state_action_value1)
        
        # agent 2
        state_value2 = F.relu(state_value)
        state_value2 = self.agent2_fc3(state_value2)
        state_value2 = self.agent2_bn3(state_value2)
        state_value2 = F.relu(state_value2)
        state_value2 = self.agent2_fc4(state_value2)
        state_value2 = self.agent2_bn4(state_value2)

        action_value2 = F.relu(self.agent2_action_value(action2))
        state_action_value2 = F.relu(T.add(state_value2, action_value2))
        state_action_value2 = self.agent2_q(state_action_value2)

        return state_action_value1, state_action_value2

    def save_checkpoint(self, best=True):
        if best:
            print('... saving best checkpoint ...')
            T.save(self.state_dict(), self.best_checkpoint_file)
        else:
            print('... saving checkpoint ...')
            T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self,load_file = ''):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(load_file))
        
        
class TwinAgent(object):
    # alpha will not be used
    def __init__(self, agent_1, agent_2, alpha, beta, input_dims, dims, tau, env, gamma=0.99,
                 n_actions=[20,2], max_size=1000000, layer1_size=400,
                 layer2_size=300, layer3_size=256, layer4_size=128, batch_size=64, 
                 update_actor_interval=2, noise = 'AWGN', agent_name = 'default', load_file = ''):
        self.load_file = load_file
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.layer3_size = layer3_size
        self.layer4_size = layer4_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.update_actor_iter = update_actor_interval
        
        self.memories = [agent_1.memory, agent_2.memory]

        self.actors = [agent_1.actor, agent_2.actor]
        
        self.target_actors = [agent_1.target_actor, agent_2.target_actor]

        self.critic_1 = TwinCriticNetwork(beta, dims, n_actions, 
                                    name='Critic_1_' + agent_name,chkpt_dir=env.data_manager.store_path)
        self.critic_2 = TwinCriticNetwork(beta, dims, n_actions, 
                                    name='Critic_2_' + agent_name,chkpt_dir=env.data_manager.store_path)


        self.target_critic_1 = TwinCriticNetwork(beta, dims, n_actions, 
                                           name='TargetCritic_1_' + agent_name,chkpt_dir=env.data_manager.store_path)
        self.target_critic_2 = TwinCriticNetwork(beta, dims, n_actions, 
                                           name='TargetCritic_2_' + agent_name,chkpt_dir=env.data_manager.store_path)
        if noise == 'OU':
            noise_1 = OUActionNoise(mu=np.zeros(n_actions[0]))
            noise_2 = OUActionNoise(mu=np.zeros(n_actions[1]))
        elif noise == 'AWGN':
            noise_1 = AWGNActionNoise(mu = np.zeros(n_actions[0]))
            noise_2 = AWGNActionNoise(mu = np.zeros(n_actions[1]))
        self.noises = [noise_1, noise_2]
        # tau = 1 means copy parameters to target
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, greedy=0.5, epsilon = 1, agent_idx=0):

        self.actors[agent_idx].eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actors[agent_idx].device)
        mu = self.actors[agent_idx].forward(observation).to(self.actors[agent_idx].device)
        mu_prime = mu + T.tensor(greedy * self.noises[agent_idx](),
                                 dtype=T.float).to(self.actors[agent_idx].device)
        self.actors[agent_idx].train()
        return mu_prime.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done, agent_idx):
        self.memories[agent_idx].store_transition(state, action, reward, new_state, done)

    def learn(self):
        states, actions, rewards, new_states, dones = [], [], [], [], []
        for i in range(2):
            if self.memories[i].mem_cntr < self.batch_size:
                return
            # the done here is opposite of the done in the env
            state, action, reward, new_state, done = \
                                          self.memories[i].sample_buffer(self.batch_size)
            # trun s, a, r, new_s into tensor
            reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
            done = T.tensor(done).to(self.critic_1.device)
            new_state = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
            action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
            state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
            
            # append
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            new_states.append(new_state)
            dones.append(done)
            
        # trun on evaliation mode of target actor, target critic, critic net
        # fix these three nets
        self.target_actors[0].eval()
        self.target_actors[1].eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        self.critic_1.eval()
        self.critic_2.eval()
    
            
        target_actions_1 = self.target_actors[0].forward(new_states[0])
        target_actions_2 = self.target_actors[1].forward(new_states[1])
        target_actions = [target_actions_1, target_actions_2]
        
        critic_value_1_ = self.target_critic_1.forward(new_states, target_actions)[0]
        critic_value_2_ = self.target_critic_2.forward(new_states, target_actions)[1]
        
        critic_value_1 = self.critic_1.forward(states, actions)[0]
        critic_value_2 = self.critic_2.forward(states, actions)[1]
        
        critic_value_ = T.min(critic_value_1_, critic_value_2_)
        
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic_1.device)
        target = target.view(self.batch_size, 1)
        
        # here update the critic net using mse of (r + gamma * Q_argmax_a*(newstate, a*)) - Q(state, action)
        self.critic_1.train()
        self.critic_2.train()
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        
        critic_1_loss = F.mse_loss(target, critic_value_1)
        critic_2_loss = F.mse_loss(target, critic_value_2)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
            
        self.learn_step_cntr += 1
        
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        # here update the actor net by policy gradient
        # first fix the critic net
        self.critic_1.eval()
        self.critic_2.eval()
        
        mus = []
        for i in range(2):
            self.actors[i].optimizer.zero_grad()
            mu = self.actors[i].forward(states[i])
            mus.append(mu)
            self.actors[i].train()
            
        actor_q1_loss = self.critic_1.forward(states, mus)
        actor_loss_1 = -T.mean(actor_q1_loss[0])
        actor_loss_1.backward()
        self.actors[0].optimizer.step()
        
        actor_q1_loss = self.critic_1.forward(states, mus)
        actor_loss_2 = -T.mean(actor_q1_loss[1])
        actor_loss_2.backward()
        self.actors[1].optimizer.step()

        self.update_network_parameters()
        

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_1_params = self.actors[0].named_parameters()
        actor_2_params = self.actors[1].named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_1_params = self.target_actors[0].named_parameters()
        target_actor_2_params = self.target_actors[1].named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_1_state_dict = dict(actor_1_params)
        actor_2_state_dict = dict(actor_2_params)
        target_actor_1_state_dict = dict(target_actor_1_params)
        target_actor_2_state_dict = dict(target_actor_2_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_1_state_dict:
            actor_1_state_dict[name] = tau*actor_1_state_dict[name].clone() + \
                    (1-tau)*target_actor_1_state_dict[name].clone()
                    
        for name in actor_2_state_dict:
            actor_2_state_dict[name] = tau*actor_2_state_dict[name].clone() + \
                    (1-tau)*target_actor_2_state_dict[name].clone()
                    
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        
        self.target_actors[0].load_state_dict(actor_1_state_dict)
        self.target_actors[1].load_state_dict(actor_2_state_dict)


    def save_models(self, best=True):
        self.actors[0].save_checkpoint(best=best)
        self.actors[1].save_checkpoint(best=best)
        self.target_actors[0].save_checkpoint(best=best)
        self.target_actors[1].save_checkpoint(best=best)
        self.critic_1.save_checkpoint(best=best)
        self.critic_2.save_checkpoint(best=best)
        self.target_critic_1.save_checkpoint(best=best)
        self.target_critic_2.save_checkpoint(best=best)

    def load_models(self, load_file_actor = '',load_file_critic_1 ='',load_file_critic_2 =''):
        self.actors[0].load_checkpoint(load_file = load_file_actor)
        self.actors[1].load_checkpoint(load_file = load_file_actor)
        self.target_actors[0].load_checkpoint(load_file = load_file_actor)
        self.target_actors[1].load_checkpoint(load_file = load_file_actor)
        self.critic_1.load_checkpoint(load_file = load_file_critic_1)
        self.critic_2.load_checkpoint(load_file = load_file_critic_2)
        self.target_critic_1.load_checkpoint(load_file = load_file_critic_1)
        self.target_critic_2.load_checkpoint(load_file = load_file_critic_2)
'''
