
from src.algos.agent import Agent
from src.nets.Actor import Actor
import torch
import torch.autograd as autograd
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


class Reinforce(Agent):

    def __init__(self, params, idx=0):
        Agent.__init__(self, params, idx)

        #opt_params = []
        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, gmm=self.gmm_).to(device)
    
        #opt_params_act = {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor}
        #opt_params.append(opt_params_act)
        self.opt_act = torch.optim.Adam(self.policy_act.parameters(), lr=self.lr_actor)
        self.scheduler_act = torch.optim.lr_scheduler.ExponentialLR(self.opt_act, gamma=self.decayRate)

        if (self.is_communicating):
            self.policy_comm = Actor(params=params, input_size=self.input_comm, output_size=self.mex_size, \
                n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, gmm=self.gmm_).to(device)
            #opt_params_comm = {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm}
            #opt_params.append(opt_params_comm)
            self.opt_comm = torch.optim.Adam(self.policy_comm.parameters(), lr=self.lr_actor_comm)
            self.scheduler_comm = torch.optim.lr_scheduler.ExponentialLR(self.opt_comm, gamma=self.decayRate)
        
        if (self.opponent_selection):
            self.policy_opponent_selection = Actor(params=params, input_size=self.n_agents, output_size=self.n_agents, \
                n_hidden=2, hidden_size=self.n_agents, gmm=False).to(device)
            self.opt_opponent = torch.optim.Adam(self.policy_opponent_selection.parameters(), lr=self.lr_opponent)

        #self.optimizer = torch.optim.Adam(opt_params)

        self.htarget = np.log(self.action_size)/2.
        self.n_update = 0.
        self.baseline = 0.

    def append_to_replay(self, s, a, r, s_, l, d):
        self.memory._rewards[self.memory.i] = r
        self.memory._logprobs[self.memory.i] = l
        self.memory.i += 1

    def embed_opponent_idx_act(self, idx):
        out = self.policy_act.embed_opponent_index(idx).t()[0]
        return out
    
    def embed_opponent_idx_comm(self, idx):
        out = self.policy_comm.embed_opponent_index(idx).t()[0]
        return out

    def update(self):
      
        rew_norm = self.buffer.rewards_norm # [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]
        act_logprobs = self.buffer.act_logprobs
        comm_logprobs = self.buffer.comm_logprobs
        opponent_logprobs = self.buffer.opponent_logprobs 

        loss_act = [] #torch.zeros_like(act_logprobs)
        loss_comm = [] #torch.zeros_like(comm_logprobs)
        loss_opponent_choice = []

        entropy = torch.FloatTensor([self.policy_comm.get_dist_entropy(state).detach() for state in self.buffer.states_c])
        self.entropy = entropy
        hloss = (torch.full(entropy.size(), self.htarget) - entropy) * (torch.full(entropy.size(), self.htarget) - entropy)
        
        if (self.opponent_selection):
            for i in range(len(opponent_logprobs)):
                loss_opponent_choice.append(-opponent_logprobs[i] * (rew_norm[i] - self.baseline))
        for i in range(len(act_logprobs)):
            if (self.is_communicating):
                loss_comm.append(-comm_logprobs[i] * (rew_norm[i] - self.baseline) + self.sign_lambda*hloss[i])
            if (self.is_listening and self.n_communicating_agents != 0.):
                #IL PROBLEMA E` QUI`
                #print("act_logprobs[i]=", act_logprobs[i])
                #print("(rew_norm[i] - self.baseline)=", (rew_norm[i] - self.baseline))
                #print("self.list_lambda*self.List_loss_list[i]=", self.list_lambda*self.List_loss_list[i])
                loss_act.append(-act_logprobs[i] * (rew_norm[i] - self.baseline))# + self.list_lambda*self.List_loss_list[i])
            else:
                #print("create loss act")
                loss_act.append(-act_logprobs[i] * (rew_norm[i] - self.baseline))
       
        self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in loss_act])))
        
        #self.optimizer.zero_grad()
        if (self.is_communicating):
            #print("update comm loss")
            self.opt_comm.zero_grad()
            self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in loss_comm])))
            tmp = [torch.ones(a.data.shape) for a in loss_comm]
            autograd.backward(loss_comm, tmp, retain_graph=True)
            self.opt_comm.step()
            self.scheduler_comm.step()

        if (self.opponent_selection):
            self.opt_opponent.zero_grad()
            tmp = [torch.ones(a.data.shape) for a in loss_opponent_choice]
            autograd.backward(loss_opponent_choice, tmp, retain_graph=True)
            self.opt_opponent.step()
        
        self.opt_act.zero_grad()
        tmp1 = [torch.ones(a.data.shape) for a in loss_act]
        autograd.backward(loss_act, tmp1, retain_graph=True)
        self.opt_act.step()
        self.scheduler_act.step()
        print("new lr=",self.scheduler_act.get_lr())
        
        self.n_update += 1.
        self.baseline += (np.mean([i for i in rew_norm]) - self.baseline) / (self.n_update)

        self.reset()