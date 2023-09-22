from src.algos.anast.Actor import Actor
import copy
import torch
import random
import torch.nn as nn
import numpy as np
from collections import namedtuple

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ExperienceReplayMemory:
    def __init__(self, params, input_state):

        for key, val in params.items(): setattr(self, key, val)
        self.capacity = self.num_game_iterations
        self.input_state = input_state
        self.reset()

    def reset(self):
        self._states = torch.empty((self.capacity,self.input_state))
        self._actions = torch.empty((self.capacity,1))
        self._rewards = torch.empty((self.capacity,1))
        self._next_states = torch.empty((self.capacity,self.input_state))
        self._dones = torch.empty((self.capacity,1), dtype=torch.bool)
        self.i = 0

    def __len__(self):
        return len(self._states)


class DQN_comm():
    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)

        #self.input_act = self.obs_size
        if (self.reputation_enabled == 0):
            self.input_act = 1
        else: 
            self.input_act = 2
        print("input_act=",self.input_act)
        self.input_mex = self.input_act
        self.input_act = self.input_mex + self.mex_size
        self.other = 0

        self.policy_mex = Actor(params=params, input_size=self.input_mex, output_size=self.mex_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(device)
        self.policy_mex_target = copy.deepcopy(self.policy_mex).to(device)
        self.policy_mex_target.load_state_dict(self.policy_mex.state_dict())

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(device)
        self.policy_act_target = copy.deepcopy(self.policy_act).to(device)
        self.policy_act_target.load_state_dict(self.policy_act.state_dict())

        self.optimizer = torch.optim.RMSprop(self.policy_act.parameters())
        self.memory_mex = ExperienceReplayMemory(params, self.input_mex)
        self.memory_act = ExperienceReplayMemory(params, self.input_act)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.target_net_update_freq = self.target_net_update_freq*2

        self.reputation = torch.Tensor([1.])
        self.old_reputation = self.reputation
        self.previous_action = torch.Tensor([1.])

        self.is_dummy = False
        self.idx = idx
        self.batch_size = self.num_game_iterations

        self.action_selections = [0 for _ in range(self.action_size)]
        self.action_log_frequency = 1.

        self.update_count = 0
        if (self.decaying_epsilon == True):
            self.eps0 = 0.1
            self.final_epsilon = 0.001
            self.epsilon_delta = (self.eps0 - self.final_epsilon)/self.n_episodes
        self.epsilon = self.eps0
        self.r = 1.-np.exp(np.log(self.final_epsilon/self.eps0)/self.n_episodes)

    def reset(self):
        self.memory_act.reset()
        self.memory_mex.reset()
        self.memory_act.i = 0
        self.memory_mex.i = 0
        #self.observed_mult_factors = torch.zeros(self.num_game_iterations)
        self.idx_mf = 0

    def argmax(self, q_values):
        top = torch.Tensor([-10000000])
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return random.choice(ties)
        
    def select_action(self, _eval=False):
        
        self.state_act = self.state_act.view(-1,self.input_act)
        #if (_eval==False):
        #    print("self.state_act=", self.state_act, self.state_act.shape)

        if (_eval == True):
            #print("action selected with argmax bc EVAL=TRUE")
            action = self.argmax(self.policy_act.get_values(state=self.state_act)[0])
            
        elif (_eval == False):
            if torch.rand(1) < self.epsilon:
                action = random.choice([i for i in range(self.action_size)])
            else:
                action = self.argmax(self.policy_act.get_values(state=self.state_act)[0])
                
        return torch.Tensor([action])
    
    def select_message(self, _eval=False):
        
        self.state_message = self.state_message.view(-1,self.input_mex)
        #if (_eval==False):
        #    print("self.state_act=", self.state_act, self.state_act.shape)

        if (_eval == True):
            #print("action selected with argmax bc EVAL=TRUE")
            mex = self.argmax(self.policy_mex.get_values(state=self.state_message)[0])
            
        elif (_eval == False):
            if torch.rand(1) < self.epsilon:
                mex = random.choice([i for i in range(self.mex_size)])
            else:
                mex = self.argmax(self.policy_mex.get_values(state=self.state_message)[0])
                
        return torch.Tensor([mex])
    
    def get_action_distribution(self, state):

        with torch.no_grad():
            out = self.policy_act.get_distribution(state)
            return out
        
    def get_action_values(self, state):

        with torch.no_grad():
            out = self.policy_act.get_values(state)
            return out

    def append_to_replay_mex(self, s, a, r, s_, d):
        self.memory_mex._states[self.memory_mex.i] = s
        self.memory_mex._actions[self.memory_mex.i] = a
        self.memory_mex._rewards[self.memory_mex.i] = r
        self.memory_mex._next_states[self.memory_mex.i] = s_
        self.memory_mex._dones[self.memory_mex.i] = d
        self.memory_mex.i += 1

    def append_to_replay_state(self, s, a, r, s_, d):
        self.memory_act._states[self.memory_act.i] = s
        self.memory_act._actions[self.memory_act.i] = a
        self.memory_act._rewards[self.memory_act.i] = r
        self.memory_act._next_states[self.memory_act.i] = s_
        self.memory_act._dones[self.memory_act.i] = d
        self.memory_act.i += 1

    def prep_minibatch(self, memory):
        batch_state = memory._states
        batch_action = memory._actions.long()
        batch_reward = memory._rewards
        batch_next_state = memory._next_states
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(-1,1)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values
    
    def read_q_matrix(self):
        possible_states = torch.Tensor([[0.], [1.]])
        Q = torch.full((2, 2),  0.)
        for state in possible_states:
            Q[state.long(),:] = self.policy_act.get_values(state.view(-1,1))
        return Q

    def compute_loss(self, policy, policy_target, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars
        #print("batch_state=",batch_state)
    
        current_q_values = policy.get_values(batch_state)
        current_q_values = torch.gather(current_q_values, dim=1, index=batch_action)
        
        #compute target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(policy_target, non_final_next_states)
                dist = policy_target.get_distribution(state=non_final_next_states) #.act(state=non_final_next_states, greedy=False, get_distrib=True)
                max_next_q_values[non_final_mask] = dist.gather(1, max_next_action)
            expected_q_values = batch_reward + self.gamma*max_next_q_values

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, _iter):

        batch_vars_mex = self.prep_minibatch(self.memory_mex)
        batch_vars_act = self.prep_minibatch(self.memory_act)

        loss_mex = self.compute_loss(self.policy_mex, self.policy_mex_target, batch_vars_mex)
        loss_act = self.compute_loss(self.policy_act, self.policy_act_target, batch_vars_act)
        loss = loss_mex + loss_act

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_model(self.policy_mex, self.policy_mex_target)
        self.update_target_model(self.policy_act, self.policy_act_target)

        self.reset()
        self.scheduler.step()
        #print("LR=",self.scheduler.get_last_lr())

        #modif exploration
        self.update_epsilon(_iter)
        #if (self.epsilon >= 0.001):
        #    self.epsilon = self.eps0*self.r**(_iter-1.)
        print("self.epsilon=",self.epsilon)

        return loss.detach()
    
    def update_epsilon(self, _iter):
        #print("update epsilon")
        #self.epsilon = self.eps0*(1.-self.r)**_iter
        if (self.epsilon >= self.final_epsilon): 
            self.epsilon = self.epsilon -self.epsilon_delta #self.eps0*(1.-self.r)**_iter

    def update_target_model(self, policy, policy_target):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            print("updating target model")
            policy_target.load_state_dict(policy.state_dict())

    def get_max_next_state_action(self, policy_target, next_states):
        dist = policy_target.get_distribution(state=next_states) ##.act(state=next_states, greedy=False, get_distrib=True)
        max_vals = dist.max(dim=1)[1].view(-1, 1)
        return max_vals
    
    def MSE(self, x):
        return 0.5 * x.pow(2)