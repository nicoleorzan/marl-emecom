
from src.algos.agent_anast import Agent
from src.nets.Actor import Actor
import torch
from collections import deque
import numpy as np

class Reinforce(Agent):

    def __init__(self, params, idx=0):
        Agent.__init__(self, params, idx)

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, gmm=self.gmm_).to(params.device)
    
        self.optimizer = torch.optim.Adam(self.policy_act.parameters(), lr=self.lr_actor)
        self.scheduler_act = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.n_update = 0.
        self.baseline = 0.
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()

    def update(self):
        
        rew = self.buffer.rewards
        #print("rew=", rew)
        R = 0; returns = deque()
        policy_loss = []
        
        for r in rew[::-1]:
            #print("r=", r)
            R = r + R*self.gamma
            #print("R=", R)
            returns.appendleft(R)

        returns = torch.tensor(returns)
        if (len(returns) > 1):
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        #print("self.buffer.logprobs=",self.buffer.logprobs)
        #print("returns-=", returns)
        #print("ret norm=",returns)

        for log_prob, R in zip(self.buffer.logprobs, returns):
            val = -log_prob * R
            policy_loss.append(val.reshape(1))
        #print("policy_loss=", policy_loss)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.reset()