
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

        opt_params = []
        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, gmm=self.gmm_).to(device)
    
        opt_params_act = {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor}
        opt_params.append(opt_params_act)

        if (self.is_communicating):
            self.policy_comm = Actor(params=params, input_size=self.input_comm, output_size=self.mex_size, \
                n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, gmm=self.gmm_).to(device)
            opt_params_comm = {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm}
            opt_params.append(opt_params_comm)

        self.optimizer = torch.optim.Adam(opt_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.htarget = np.log(self.action_size)/2.
        self.n_update = 0.
        self.baseline = 0.

    def update(self):

        # I do not normalize rewards here because I already give normalized rewards to the agent
        rew_norm = self.buffer.rewards # [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]
        act_logprobs = self.buffer.act_logprobs
        comm_logprobs = self.buffer.comm_logprobs

        entropy = torch.FloatTensor([self.policy_comm.get_dist_entropy(state).detach() for state in self.buffer.states_c])
        self.entropy = entropy
        hloss = (torch.full(entropy.size(), self.htarget) - entropy) * (torch.full(entropy.size(), self.htarget) - entropy)
        for i in range(len(act_logprobs)):
            if (self.is_communicating):
                comm_logprobs[i] = -comm_logprobs[i] * (rew_norm[i] - self.baseline) + self.sign_lambda*hloss[i]
            act_logprobs[i] = -act_logprobs[i] * (rew_norm[i] - self.baseline) 
            if (self.is_listening and self.n_communicating_agents != 0.):
                act_logprobs[i] += self.list_lambda*self.List_loss_list[i]
       
        self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in act_logprobs])))
        
        self.optimizer.zero_grad()
        if (self.is_communicating):
            self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in comm_logprobs])))
            tmp = [torch.ones(a.data.shape) for a in comm_logprobs]
            autograd.backward(comm_logprobs, tmp, retain_graph=True)

        tmp1 = [torch.ones(a.data.shape) for a in act_logprobs]
        autograd.backward(act_logprobs, tmp1, retain_graph=True)
        
        self.optimizer.step()

        #diminish learning rate
        self.scheduler.step()
        #print(self.scheduler.get_lr())

        self.n_update += 1.
        self.baseline += (np.mean([i[0] for i in rew_norm]) - self.baseline) / (self.n_update)

        self.reset()