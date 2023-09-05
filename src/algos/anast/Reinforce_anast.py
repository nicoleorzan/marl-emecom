
from src.algos.anast.Actor import Actor
import torch
from collections import deque
from torch.distributions import Categorical
import numpy as np

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()
        
    def reset(self):
        self._states = torch.empty((self.capacity,1))
        self._actions = torch.empty((self.capacity,1))
        self._rewards = torch.empty((self.capacity,1))
        self._logprobs = torch.empty((self.capacity,1))
        self._next_states = torch.empty((self.capacity,1))
        self._dones = torch.empty((self.capacity,1), dtype=torch.bool)
        self.i = 0

    def __len__(self):
        return len(self._states)

    
class Reinforce():

    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)

        self.input_act = self.obs_size

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(params.device)
    
        self.optimizer = torch.optim.Adam(self.policy_act.parameters(), lr=self.lr_actor)
        self.memory = ExperienceReplayMemory(self.num_game_iterations)
        self.scheduler_act = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.n_update = 0.
        self.baseline = 0.

        self.reputation = torch.Tensor([1.])
        self.old_reputation = self.reputation
        self.previous_action = torch.Tensor([1.])

        self.is_dummy = False
        self.idx = idx
        self.batch_size = self.num_game_iterations

        self.eps = np.finfo(np.float32).eps.item()

        self.eps_batch = 0.00001

    def reset(self):
        self.memory.reset()
        self.memory.i = 0

    def temperature_scaled_softmax(logits, temperature=1.0):
        logits = logits / temperature
        return torch.softmax(logits, dim=0)
            
    def select_action(self, _eval=False):

        self.state_act = self.state_act.view(-1,self.input_act)
        #print("self.state_act=", self.state_act)

        out = self.policy_act.get_distribution(state=self.state_act)
        #print("out=", out)
        dist = Categorical(out)

        if (_eval == True):
            act = torch.argmax(out).detach()
        else:
            act = dist.sample().detach()
        logprob = dist.log_prob(act) # negativi
        #print("act=", act, "logprob=", logprob)
        
        return act, logprob

    def get_action_distribution(self, state):

        with torch.no_grad():
            out = self.policy_act.get_distribution(state)
            return out

    def append_to_replay(self, s, a, r, s_, l, d):
        self.memory._rewards[self.memory.i] = r
        self.memory._logprobs[self.memory.i] = l
        self.memory.i += 1

    def read_distrib(self, possible_states, n_possible_states):
        dist = torch.full((n_possible_states, 2),  0.)
        for state in possible_states:
            dist[state.long(),:] = self.policy_act.get_distribution(state.view(-1,self.input_act))
        return dist

    def update_return(self):

        batch_reward = self.memory._rewards
        #print("batch_reward=", batch_reward)

        R = 0; returns = deque()
        policy_loss = []
        
        for r in list(batch_reward)[::-1]:
            R = r + R*self.gamma
            returns.appendleft(R)
        #print("returns=", returns)

        returns = torch.tensor(returns)
        if (len(returns) > 1):
            returns = (returns - returns.min()) / (returns.max() - returns.min() + self.eps_batch )
            #returns = (returns - returns.mean()) / (returns.std() + self.epsilon)
        #print("norm returns=", returns)

        for log_prob, R in zip(self.memory._logprobs, returns):
            val = -log_prob * R
            policy_loss.append(val.reshape(1))

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.reset()

        return policy_loss.detach()
    
    def update_reward(self):

        batch_reward = self.memory._rewards
        #print("batch_rew=", batch_reward)

        policy_loss = []

        if (len(batch_reward) > 1):
            #batch_reward1 = (batch_reward - batch_reward.min()) / (batch_reward.max() - batch_reward.min() + self.eps_batch)
            #print("batch_rew=", batch_reward1)
            batch_reward = (batch_reward - batch_reward.min()) / (batch_reward.std() + self.eps_batch)
            #print("or batch_rew=", batch_reward)
            #print("logprobs=",self.memory._logprobs)
        for log_prob, rew in zip(self.memory._logprobs, batch_reward):
            #print("-logprob=", -log_prob, ", rew=", rew)
            val = -log_prob * rew
            policy_loss.append(val.reshape(1))

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.reset()

        return policy_loss.detach()