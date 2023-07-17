
from src.algos.agent import Agent
from src.nets.Actor import Actor
import torch
import torch.autograd as autograd
import numpy as np

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

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

    def embed_opponent_idx_act(self, idx):
        #print("embed_opponent_idx, inside reinforce")
        out = self.policy_act.embed_opponent_index(idx).t()[0]
        return out
    
    def embed_opponent_idx_comm(self, idx):
        #print("embed_opponent_idx, inside reinforce")
        out = self.policy_comm.embed_opponent_index(idx).t()[0]
        return out

    def update(self):
        #print("agent=", self.idx)
        #print("\n=====>Update agent", self.idx)
        # I do not normalize rewards here because I already give normalized rewards to the agent
        rew_norm = self.buffer.rewards_norm # [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]
        print("rew norm=", len(rew_norm))
        act_logprobs = self.buffer.act_logprobs
        comm_logprobs = self.buffer.comm_logprobs
        opponent_logprobs = self.buffer.opponent_logprobs 
        #print("act_logprobs=",act_logprobs)
        print("opponent_logprobs=", len(opponent_logprobs))

        loss_act = [] #torch.zeros_like(act_logprobs)
        loss_comm = [] #torch.zeros_like(comm_logprobs)
        loss_opponent_choice = []

        entropy = torch.FloatTensor([self.policy_comm.get_dist_entropy(state).detach() for state in self.buffer.states_c])
        self.entropy = entropy
        hloss = (torch.full(entropy.size(), self.htarget) - entropy) * (torch.full(entropy.size(), self.htarget) - entropy)
        
        if (self.opponent_selection):
            #print("create loss opponent selections")
            #print("np.mean(rew_norm)=", np.mean(rew_norm))
            for i in range(len(opponent_logprobs)):
                #print("-opponent_logprobs[i] * (np.mean(rew_norm) - self.baseline)=",-opponent_logprobs[i] * (np.mean(rew_norm) - self.baseline))
                loss_opponent_choice.append(-opponent_logprobs[i] * (rew_norm[i] - self.baseline))
                #loss_opponent_choice.append(-opponent_logprobs[i] * (np.mean(rew_norm) - self.baseline))
                #print("loss opp=", loss_opponent_choice)
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
            autograd.backward(loss_comm, tmp, retain_graph=True)
            self.opt_opponent.step()
        
        #print("update act loss")
        #print("loss_act=", loss_act)
        self.opt_act.zero_grad()
        tmp1 = [torch.ones(a.data.shape) for a in loss_act]
        autograd.backward(loss_act, tmp1, retain_graph=True)
        self.opt_act.step()
        self.scheduler_act.step()
        #print("optimized")
        
        #self.optimizer.step()

        #diminish learning rate
        self.scheduler.step()
        #print(self.scheduler.get_lr())

        self.n_update += 1.
        self.baseline += (np.mean([i for i in rew_norm]) - self.baseline) / (self.n_update)
        #print("baseline=", self.baseline)

        self.reset()