
from src.algos.buffer import RolloutBufferComm
from src.nets.ActorCritic import ActorCritic
import torch
import copy
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class ReinforceGeneral():

    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)

        self.buffer = RolloutBufferComm()

        self.idx = idx
        self.gmm_ = self.gmm_[idx]
        self.is_communicating = self.communicating_agents[self.idx]
        self.is_listening = self.listening_agents[self.idx]
        print("\nAgent", self.idx)
        print("is communicating?:", self.is_communicating)
        print("is listening?:", self.is_listening)
        print("uncertainty:", self.uncertainties[idx])
        print("gmm=", self.gmm_)
        self.n_communicating_agents = self.communicating_agents.count(1)
        opt_params = []

        self.max_f = max(self.mult_fact)
        self.min_f = min(self.mult_fact)
        # Action Policy
        input_act = self.obs_size
        if (self.gmm_):
            input_act = self.n_gmm_components  #gmm components for the m factor, 1 for the coins I get
        if (self.is_listening):
            input_act += self.mex_size*self.n_communicating_agents

        print("input_act:", input_act)
        self.policy_act = ActorCritic(params=params, input_size=input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, gmm=self.gmm_).to(device)
    
        opt_params_act = {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor}
        opt_params.append(opt_params_act)

        # Communication Policy
        if (self.is_communicating):
            input_comm = self.obs_size
            if (self.gmm_):
                input_comm = self.n_gmm_components #gmm components for the m factor, 1 for the coins I get

            print("input_comm=", input_comm)
            self.policy_comm = ActorCritic(params=params, input_size=input_comm, output_size=self.mex_size, \
                n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, gmm=self.gmm_).to(device)
            print("output comm=", self.mex_size)
            
            opt_params_comm = {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm}
            opt_params.append(opt_params_comm)

        self.optimizer = torch.optim.Adam(opt_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.htarget = np.log(self.action_size)/2.
        self.n_update = 0.
        self.baseline = 0.

        self.eps_norm = 0.0001
        self.comm_loss = True

        self.ent = True
        self.param_entropy = 0.1
        self.entropy = 0.

        self.means = []
        self.probs = []

        #### == this needs to be there, is not a repetition == 
        self.sc = []
        self.sc_m = {}
        self.mutinfo_signaling = []
        self.mutinfo_listening = []
        self.return_episode_norm = 0
        self.return_episode = 0
        #### ==

        self.reset()

        self.saved_losses_comm = []
        self.saved_losses = []

    def reset(self):
        self.comm_logprobs = []
        self.act_logprobs = []
        self.List_loss_list = []
        self.comm_entropy = []
        self.act_entropy = []
        self.rewards = []
        self.mutinfo_signaling_old = self.mutinfo_signaling
        self.mutinfo_listening_old = self.mutinfo_listening
        self.sc_m_old = self.sc_m
        self.sc_old = self.sc
        self.sc = []
        self.sc_m = {}
        self.mutinfo_signaling = []
        self.mutinfo_listening = []
        self.buffer.clear()
        self.reset_episode()

    def reset_episode(self):
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode_old = self.return_episode
        self.return_episode = 0
        self.return_episode_norm = 0

    def set_state(self, state, _eval=False):
        #print("agent=", self.idx)

        # set the internal state of the agent, called self.state_in
        state = torch.FloatTensor(state).to(device)
        if (self.gmm_):
            # If I have uncerainty and use gmm_, the obs of f will become a tensor with probabilities for every posisble m
            self.get_gmm_state(state, _eval)
        else:
            # otherwise I just need to normalize the observed f
            state = (state - self.min_f)/(self.max_f - self.min_f)
            self.state_to_comm = state
        #print("state to comm=",self.state_to_comm)
        self.state_to_act = self.state_to_comm

    def select_message(self, m_val=None, _eval=False):

        if (_eval == True):
            with torch.no_grad():
                message_out, message_logprob, entropy = self.policy_comm.act(self.state_to_comm, self.ent)

        elif (_eval == False):
            message_out, message_logprob, entropy = self.policy_comm.act(self.state_to_comm, self.ent)

            self.buffer.states_c.append(self.state_to_comm)
            self.buffer.messages.append(message_out)
            if (m_val in self.buffer.messages_given_m):
                self.buffer.messages_given_m[m_val].append(message_out)
            else: 
                self.buffer.messages_given_m[m_val] = [message_out]

            self.comm_logprobs.append(message_logprob)
            self.comm_entropy.append(entropy)

        message_out = torch.Tensor([message_out.item()]).long().to(device)
        message_out = F.one_hot(message_out, num_classes=self.mex_size)[0]
        #print("send message=", message_out)
        return message_out

    def random_messages(self, m_val=None):

        message_out = torch.randint(0, self.mex_size, (self.mex_size-1,))[0]
        self.buffer.states_c.append(self.state_to_comm)
        self.buffer.messages.append(message_out)
        if (m_val in self.buffer.messages_given_m):
            self.buffer.messages_given_m[m_val].append(message_out)
        else: 
            self.buffer.messages_given_m[m_val] = [message_out]

        self.comm_logprobs.append(torch.tensor(0.0001))

        message_out = torch.Tensor([message_out.item()]).long().to(device)
        message_out = F.one_hot(message_out, num_classes=self.mex_size)[0]

        return message_out

    def get_message(self, message_in):
        self.message_in = message_in
        self.state_to_act = torch.cat((self.state_to_comm, self.message_in)).to(device)
        #print("listen_message=", self.message_in)
        #print("state to act becomes=", self.state_to_act)

    def select_action(self, m_val=None, _eval=False):
            
        if (_eval == True):
            with torch.no_grad():
                action, action_logprob, entropy = self.policy_act.act(self.state_to_act, self.ent)

        elif (_eval == False):
            #print("input act net=",self.state_to_act)
            action, action_logprob, entropy = self.policy_act.act(self.state_to_act, self.ent)
            
            if (self.is_listening == True and self.n_communicating_agents != 0.):
                state_empty_mex = torch.cat((self.state_to_comm, torch.zeros_like(self.message_in))).to(device)
                dist_empty_mex = self.policy_act.get_distribution(state_empty_mex).detach()
                dist_mex = self.policy_act.get_distribution(self.state_to_act)
                self.List_loss_list.append(-torch.sum(torch.abs(dist_empty_mex - dist_mex)))

            self.buffer.states_a.append(self.state_to_act)
            self.buffer.actions.append(action)
            if (m_val in self.buffer.actions_given_m):
                self.buffer.actions_given_m[m_val].append(action)
            else: 
                self.buffer.actions_given_m[m_val] = [action]

            self.act_logprobs.append(action_logprob)
            self.act_entropy.append(entropy)
        
        return action

    def get_action_distribution(self):

        with torch.no_grad():
            out = self.policy_act.get_distribution(self.state_to_act)
            return out

    def get_message_distribution(self):

        with torch.no_grad():
            out = self.policy_comm.get_distribution(self.state_to_comm)
            return out.detach()

    def get_gmm_state(self, state, _eval=False):

        self.state_to_comm = torch.zeros(self.n_gmm_components).to(device)
        if hasattr(self, 'mf_history'):
            if (len(self.mf_history) < 50000 and _eval == False):
                self.mf_history = torch.cat((self.mf_history, state.reshape(1)), 0)
        else:
            self.mf_history = state.reshape(1)

        if (len(self.mf_history) >= self.n_gmm_components):
            if(len(self.mf_history) < 50000):
                self.gmm = GMM(n_components = self.n_gmm_components, max_iter=1000, random_state=0, covariance_type = 'full')
                input_ = self.mf_history.reshape(-1, 1)
                self.gmm.fit(input_)

            self.gmm_probs = self.gmm.predict_proba(state.reshape(1).reshape(-1, 1))[0]

            self.means = copy.deepcopy(self.gmm.means_)
            #print("means=", self.means)
            self.means = np.sort(self.means, axis=0)
            ##print("means sort=", self.means)
            ordering_values = [np.where(self.means == i)[0][0] for i in self.gmm.means_]
            self.state_to_comm = torch.zeros(len(self.gmm_probs)).to(device)
            #print("self.probs=", self.gmm_probs)
            for i, value in enumerate(ordering_values):
                self.state_to_comm[value] = self.gmm_probs[i] 
            #print(self.state_to_comm.shape)

    def update(self):

        #rewards = self.rewards
        #print("rewards=", rewards)
        # I do not normalize rewards here because I already give normalized rewards to the agent
        rew_norm = self.rewards # [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]

        entropy = torch.FloatTensor([self.policy_comm.get_dist_entropy(state).detach() for state in self.buffer.states_c])
        self.entropy = entropy
        hloss = (torch.full(entropy.size(), self.htarget) - entropy) * (torch.full(entropy.size(), self.htarget) - entropy)
        for i in range(len(self.act_logprobs)):
            if (self.is_communicating):
                self.comm_logprobs[i] = -self.comm_logprobs[i] * (rew_norm[i] - self.baseline) + self.sign_lambda*hloss[i]
            self.act_logprobs[i] = -self.act_logprobs[i] * (rew_norm[i] - self.baseline) 
            if (self.is_listening and self.n_communicating_agents != 0.):
                self.act_logprobs[i] += self.list_lambda*self.List_loss_list[i]
       
        self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in self.act_logprobs])))
        
        self.optimizer.zero_grad()
        if(self.is_communicating):
            self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in self.comm_logprobs])))
            tmp = [torch.ones(a.data.shape) for a in self.comm_logprobs]
            autograd.backward(self.comm_logprobs, tmp, retain_graph=True)

        tmp1 = [torch.ones(a.data.shape) for a in self.act_logprobs]
        autograd.backward(self.act_logprobs, tmp1, retain_graph=True)
        
        self.optimizer.step()

        #diminish learning rate
        self.scheduler.step()

        self.n_update += 1.
        self.baseline += (np.mean([i[0] for i in rew_norm]) - self.baseline) / (self.n_update)

        self.reset()