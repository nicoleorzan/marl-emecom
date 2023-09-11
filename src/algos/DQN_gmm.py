from src.algos.anast.Actor import Actor
import copy
import torch
import random
import torch.nn as nn
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.mixture import GaussianMixture as GMM
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


class DQN_gmm():
    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)

        self.n_gmm_components = 3 #number of environments: competitive, mixed, cooperative
        if (self.reputation_enabled == 0):
            self.input_act = self.n_gmm_components
        else: 
            self.input_act = self.n_gmm_components + 1
        print("input_act=",self.input_act)
        self.max_memory_capacity = 10000

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(device)
        self.policy_act_target = copy.deepcopy(self.policy_act).to(device)
        self.policy_act_target.load_state_dict(self.policy_act.state_dict())

        self.optimizer = torch.optim.RMSprop(self.policy_act.parameters())
        self.memory = ExperienceReplayMemory(params, self.input_act)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.reputation = torch.Tensor([1.])
        self.old_reputation = self.reputation
        self.previous_action = torch.Tensor([1.])

        self.is_dummy = False
        self.idx = idx
        self.batch_size = self.num_game_iterations

        self.action_selections = [0 for _ in range(self.action_size)]
        self.action_log_frequency = 1.

        self.update_count = 0
        self.eps0 = self.epsilon
        #print("self.epsilon=", self.epsilon)
        self.r = 0.99

    def reset(self):
        self.memory.reset()
        self.memory.i = 0

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
    
    def add_to_observation_history(self, state_input):
        if (self.reputation_enabled == 0):
            mult_fact = state_input[0]
        else: 
            mult_fact = state_input[1]

        if (hasattr(self, 'obs_history') == False):
            self.obs_history = mult_fact.reshape(1)
        else:
            if (len(self.obs_history) < self.max_memory_capacity):
                self.obs_history = torch.cat((self.obs_history, mult_fact.reshape(1)), 0)
        #print("self.obs_history=", self.obs_history)

    def set_gmm_state(self, state_input):
        #print("state_input=", state_input)
        if (self.reputation_enabled == 0):
            mult_fact = state_input[0]
        else: 
            reputation = state_input[0]
            mult_fact = state_input[1]

        state = torch.zeros(self.n_gmm_components).to(device)

        if (len(self.obs_history) >= self.num_game_iterations*self.n_gmm_components):
            if(len(self.obs_history) < self.max_memory_capacity):
                self.gmm = GMM(n_components = self.n_gmm_components, max_iter=1000, random_state=0, covariance_type = 'full')
                input_ = self.obs_history.reshape(-1, 1)
                self.gmm.fit(input_)

            self.gmm_probs = self.gmm.predict_proba(mult_fact.reshape(1).reshape(-1, 1))[0]
            #print(self.gmm.means_.shape)
            gmm_torch = torch.Tensor(self.gmm.means_)
            #rint("gmm torch=", gmm_torch, gmm_torch.shape)

            self.means = copy.deepcopy(gmm_torch)#self.gmm.means_)
            #print("self.means =",self.means )
            #print("self.gmm_probs=",self.gmm_probs )
            #print("numpy sort=",np.sort(self.means, axis=0))
            self.means = torch.sort(self.means, axis=0)
            means = self.means[0]
            ordering_values = self.means[1]
            #print("self.means =",self.means )
            #print("means=", means)
            #print("ordering valiues-", ordering_values)
            #ordering_values = [torch.where(self.means == i)[0][0] for i in gmm_torch]#self.gmm.means_]
            state = torch.zeros(len(self.gmm_probs)).to(device)
            for i, value in enumerate(ordering_values):
                state[value] = self.gmm_probs[i] 
        if (self.reputation_enabled == True):
            reputation = torch.Tensor([state_input[0]])#.view(-1,1)
            state = torch.cat((reputation, state), 0)
        state = state.view(-1,self.input_act)
        #print("final state=", state)
        #print("final state=", state.view(-1,self.input_act))
        return state

        
    def select_action(self, _eval=False):
        #print(" self.state_act=", self.state_act)
        #print("self.state_act before=", self.state_act.shape)
        self.state_act = self.state_act.view(-1,self.input_act-self.n_gmm_components)
        self.add_to_observation_history(self.state_act)

        self.state_act = self.set_gmm_state(self.state_act)
        #print("self.state_act=", self.state_act.shape)

        if (_eval == True):
            #print("action selected with argmax bc EVAL=TRUE")
            action = self.argmax(self.policy_act.get_values(state=self.state_act)[0])
            
        elif (_eval == False):
            if torch.rand(1) < self.epsilon:
                action = random.choice([i for i in range(self.action_size)])
            else:
                action = self.argmax(self.policy_act.get_values(state=self.state_act)[0])
                
        return torch.Tensor([action])
    
    def get_action_distribution(self, state):
        if (self.reputation_enabled == True):
            state_gmm = torch.zeros((len(self.mult_fact), self.n_gmm_components+1))
        else: 
            state_gmm = torch.zeros((len(self.mult_fact), self.n_gmm_components))

        for i in range(len(self.mult_fact)):
            state_gmm[i,:] = self.set_gmm_state(state[i,:])

        with torch.no_grad():
            out = self.policy_act.get_distribution(state_gmm)
            return out
        
    def get_action_values(self, state):
        if (self.reputation_enabled == True):
            state_gmm = torch.zeros((len(self.mult_fact), self.n_gmm_components+1))
        else: 
            state_gmm = torch.zeros((len(self.mult_fact), self.n_gmm_components))

        for i in range(len(self.mult_fact)):
            state_gmm[i,:] = self.set_gmm_state(state[i,:])

        with torch.no_grad():
            out = self.policy_act.get_values(state_gmm)
            return out

    def append_to_replay(self, s, a, r, s_, d):
        s = self.set_gmm_state(s)
        s_ = self.set_gmm_state(s_)
        self.memory._states[self.memory.i] = s
        self.memory._actions[self.memory.i] = a
        self.memory._rewards[self.memory.i] = r
        self.memory._next_states[self.memory.i] = s_
        self.memory._dones[self.memory.i] = d
        self.memory.i += 1

    def prep_minibatch(self):
        batch_state = self.memory._states
        batch_action = self.memory._actions.long()
        batch_reward = self.memory._rewards
        batch_next_state = self.memory._next_states
        
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

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars
        #print("batch_state=",batch_state)
    
        current_q_values = self.policy_act.get_values(batch_state)
        current_q_values = torch.gather(current_q_values, dim=1, index=batch_action)
        
        #compute target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                dist = self.policy_act_target.get_distribution(state=non_final_next_states) #.act(state=non_final_next_states, greedy=False, get_distrib=True)
                max_next_q_values[non_final_mask] = dist.gather(1, max_next_action)
            expected_q_values = batch_reward + self.gamma*max_next_q_values

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, _iter):

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_model()

        self.reset()
        self.scheduler.step()
        #print("LR=",self.scheduler.get_last_lr())

        #modif exploration
        if (self.epsilon >= 0.01):
            self.epsilon = self.eps0*self.r**(_iter-1.)
        #print("self.epsilon=",self.epsilon)

        return loss.detach()

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            print("updating target model")
            self.policy_act_target.load_state_dict(self.policy_act.state_dict())

    def get_max_next_state_action(self, next_states):
        dist = self.policy_act_target.get_distribution(state=next_states) ##.act(state=next_states, greedy=False, get_distrib=True)
        max_vals = dist.max(dim=1)[1].view(-1, 1)
        return max_vals
    
    def MSE(self, x):
        return 0.5 * x.pow(2)