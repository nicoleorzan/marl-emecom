from src.algos.agent import Agent
from src.nets.Actor import Actor
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch.distributions import Categorical
import numpy as np

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class Reinforce(Agent):

    def __init__(self, params, idx=0):
        Agent.__init__(self, params, idx)

        opt_params = []
        print("self.input_act=",self.input_act)
        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, gmm=self.gmm_).to(device)
    
        opt_params_act = {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor}
        opt_params.append(opt_params_act)

        # Communication Policy
        if (self.is_communicating):
            print("self.input_comm=", self.input_comm)
            self.policy_comm = Actor(params=params, input_size=self.input_comm, output_size=self.mex_size, \
                n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, gmm=self.gmm_).to(device)
            opt_params_comm = {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm}
            opt_params.append(opt_params_comm)

        self.optimizer = torch.optim.Adam(opt_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.htarget = np.log(self.action_size)/2.
        self.n_update = 0.
        self.baseline = 0.

    def act(self, policy, state, greedy=False):
        out = policy.actor(state)
        out = self.softmax(out)
        dist = Categorical(out)

        if (self.random_baseline == True): 
            act = torch.randint(0, self.action_size, (1,))[0]
        else:
            if (greedy):
                act = torch.argmax(out)
            else:
                act = dist.sample()

        logprob = dist.log_prob(act) # negativi

        return act.detach(), logprob, dist.entropy().detach()

    def select_message(self, m_val=None, _eval=False):

        if (_eval == True):
            with torch.no_grad():
                message_out, message_logprob, entropy = self.act(self.policy_comm, self.state)

        elif (_eval == False):
            message_out, message_logprob, entropy = self.act(self.policy_comm, self.state)

            self.buffer.states_c.append(self.state)
            self.buffer.messages.append(message_out)
            if (m_val in self.buffer.messages_given_m):
                self.buffer.messages_given_m[m_val].append(message_out)
            else: 
                self.buffer.messages_given_m[m_val] = [message_out]

            self.buffer.comm_logprobs.append(message_logprob)
            self.buffer.comm_entropy.append(entropy)

        message_out = torch.Tensor([message_out.item()]).long().to(device)
        message_out = F.one_hot(message_out, num_classes=self.mex_size)[0]
        return message_out

    def select_action(self, m_val=None, _eval=False):
            
        self.state_to_act = self.state
        if (self.is_listening):
            self.state_to_act = torch.cat((self.state, self.message_in)).to(device)

        if (_eval == True):
            with torch.no_grad():
                action, action_logprob, entropy = self.act(self.policy_act, self.state_to_act)

        elif (_eval == False):
            #print("input act net=",self.state_to_act)
            action, action_logprob, entropy = self.act(self.policy_act, self.state_to_act)
            
            if (self.is_listening == True and self.n_communicating_agents != 0.):
                state_empty_mex = torch.cat((self.state, torch.zeros_like(self.message_in))).to(device)
                out = self.policy_act.get_values(state_empty_mex).detach()
                dist_empty_mex = self.softmax(out)
                out = self.policy_act.get_values(self.state_to_act)
                dist_mex = self.softmax(out)
                self.List_loss_list.append(-torch.sum(torch.abs(dist_empty_mex - dist_mex)))

            self.buffer.states_a.append(self.state_to_act)
            self.buffer.actions.append(action)
            if (m_val in self.buffer.actions_given_m):
                self.buffer.actions_given_m[m_val].append(action)
            else: 
                self.buffer.actions_given_m[m_val] = [action]

            self.buffer.act_logprobs.append(action_logprob)
            self.buffer.act_entropy.append(entropy)
        
        return action

    def update(self, _iter=None):

        # I do not normalize rewards here because I already give normalized rewards to the agent
        rew_norm = self.buffer.rewards # [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]
        act_logprobs = self.buffer.act_logprobs
        comm_logprobs = self.buffer.comm_logprobs

        print("self.buffer.states_c=", self.buffer.states_c)

        entropy = torch.FloatTensor([self.policy_comm.get_dist_entropy(state).detach() for state in self.buffer.states_c])
        self.entropy = entropy
        hloss = (torch.full(entropy.size(), self.htarget) - entropy) * (torch.full(entropy.size(), self.htarget) - entropy)
        for i in range(len(act_logprobs)):
            if (self.is_communicating):
                comm_logprobs[i] = -comm_logprobs[i] * (rew_norm[i] - self.baseline) + self.sign_lambda*hloss[i]
            act_logprobs[i] = -act_logprobs[i] * (rew_norm[i] - self.baseline)
            #AGAIN, HERE IS THE PROBLEM
            #if (self.is_listening and self.n_communicating_agents != 0.):
            #    act_logprobs[i] = -act_logprobs[i] * (rew_norm[i] - self.baseline) + self.list_lambda*self.List_loss_list[i]
        
        self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in act_logprobs])))
        
        self.optimizer.zero_grad()
        if(self.is_communicating):
            self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in comm_logprobs])))
            tmp = [torch.ones(a.data.shape) for a in comm_logprobs]
            autograd.backward(comm_logprobs, tmp, retain_graph=True)

        tmp1 = [torch.ones(a.data.shape) for a in act_logprobs]
        autograd.backward(act_logprobs, tmp1, retain_graph=True)
        
        self.optimizer.step()

        #diminish learning rate
        self.scheduler.step()

        self.n_update += 1.
        self.baseline += (np.mean([i[0] for i in rew_norm]) - self.baseline) / (self.n_update)

        self.reset()