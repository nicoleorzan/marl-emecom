
from src.algos.buffer import RolloutBufferComm
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

class Agent():

    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)

        self.buffer = RolloutBufferComm()

        self.reputation = 0.5

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

        self.max_f = max(self.mult_fact)
        self.min_f = min(self.mult_fact)
        # Action Policy
        self.input_act = self.obs_size
        if (self.gmm_):
            self.input_act = self.n_gmm_components  #gmm components for the m factor, 1 for the coins I get
        if (self.is_listening):
            self.input_act += self.mex_size*self.n_communicating_agents
    
        # Communication Policy
        if (self.is_communicating):
            self.input_comm = self.obs_size
            if (self.gmm_):
                self.input_comm = self.n_gmm_components #gmm components for the m factor, 1 for the coins I get

        self.max_memory_capacity = 5000

        self.means = []
        self.probs = []

        #### == this needs to be there, is not a repetition == 
        self.return_episode_norm = 0
        self.return_episode = 0
        self.mutinfo_signaling = []
        self.mutinfo_listening = []
        self.sc = []
        self.sc_m = {}
        #### ==

        self.reset()
        self.reset_batch()

        self.saved_losses_comm = []
        self.saved_losses = []
        self.List_loss_list = []

    def reset(self):
        self.buffer.clear()
        self.reset_episode()
        self.mutinfo_signaling_old = self.mutinfo_signaling
        self.mutinfo_listening_old = self.mutinfo_listening
        self.sc_old = self.sc
        self.sc = []
        self.mutinfo_signaling = []
        self.mutinfo_listening = []

    def reset_batch(self):
        self.buffer.clear_batch()
        self.sc_m_old = self.sc_m
        self.sc_m = {}

    def reset_episode(self):
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode_old = self.return_episode
        self.return_episode = 0
        self.return_episode_norm = 0

    def digest_input(self, input):
        obs_m_fact, opponent_idx, reputation = input

        ##### set multiplier factor observation (if uncertainty is modeled, gmm is used)
        digested_m_factor = self.set_mult_fact_obs(obs_m_fact)
        #print("digested_m_factor=",digested_m_factor)
        ##### embed adversary index
        digested_opponent_idx = self.embed_opponent_idx(opponent_idx)
        #print("digested_opponent_idx=",digested_opponent_idx)
        ##### make reputation a tensor
        reputation = torch.Tensor([reputation])
        ##### concatenate all stuff in a single vector
        self.state = torch.cat((digested_m_factor, digested_opponent_idx, reputation), 0)
        #print("self.state=", self.state)      

    def embed_opponent_idx(self, idx):
        pass

    def set_mult_fact_obs(self, obs_m_fact, _eval=False):

        ##### set the iternal state of the agent regarding the observation of the m factor, called self.state_in
        obs_m_fact = torch.FloatTensor(obs_m_fact).to(device)
        if (self.gmm_):
            # If I have uncerainty and use gmm_, the obs of f will become a tensor with probabilities for every posisble m
            obs_m_fact = self.set_gmm_state(obs_m_fact, _eval)
        else:
            # otherwise I just need to normalize the observed f
            obs_m_fact = (obs_m_fact - self.min_f)/(self.max_f - self.min_f)
        return obs_m_fact
    
    def set_gmm_state(self, obs, _eval=False):

        gmm_state = torch.zeros(self.n_gmm_components).to(device)
        if hasattr(self, 'obs_history'):
            if (len(self.obs_history) < 50000 and _eval == False):
                self.obs_history = torch.cat((self.obs_history, obs.reshape(1)), 0)
        else:
            self.obs_history = obs.reshape(1)

        if (len(self.obs_history) >= self.n_gmm_components):
            if(len(self.obs_history) < self.max_memory_capacity):
                self.gmm = GMM(n_components = self.n_gmm_components, max_iter=1000, random_state=0, covariance_type = 'full')
                input_ = self.obs_history.reshape(-1, 1)
                self.gmm.fit(input_)

            self.gmm_probs = self.gmm.predict_proba(obs.reshape(1).reshape(-1, 1))[0]

            self.means = copy.deepcopy(self.gmm.means_)
            self.means = np.sort(self.means, axis=0)
            ordering_values = [np.where(self.means == i)[0][0] for i in self.gmm.means_]
            gmm_state = torch.zeros(len(self.gmm_probs)).to(device)
            for i, value in enumerate(ordering_values):
                gmm_state[value] = self.gmm_probs[i] 

        return gmm_state

    def select_message(self, m_val=None, _eval=False):

        if (_eval == True):
            with torch.no_grad():
                message_out, message_logprob, entropy = self.policy_comm.act(self.state)

        elif (_eval == False):
            message_out, message_logprob, entropy = self.policy_comm.act(self.state)

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

    def get_message(self, message_in):
        self.message_in = message_in

    def select_action(self, m_val=None, _eval=False):

        self.state_to_act = self.state
        if (self.is_listening):
            self.state_to_act = torch.cat((self.state, self.message_in)).to(device)
            
        if (_eval == True):
            with torch.no_grad():
                action, action_logprob, entropy = self.policy_act.act(self.state_to_act)

        elif (_eval == False):
            action, action_logprob, entropy = self.policy_act.act(self.state_to_act)
            
            if (self.is_listening == True and self.n_communicating_agents != 0.):
                state_empty_mex = torch.cat((self.state, torch.zeros_like(self.message_in))).to(device)
                dist_empty_mex = self.policy_act.get_distribution(state_empty_mex).detach()
                dist_mex = self.policy_act.get_distribution(self.state_to_act)
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
    
    def get_action_distribution(self):
        self.state_to_act = self.state
        if (self.is_listening):
            self.state_to_act = torch.cat((self.state, self.message_in)).to(device)

        with torch.no_grad():
            out = self.policy_act.get_distribution(self.state_to_act)
            return out

    def get_message_distribution(self):
        with torch.no_grad():
            out = self.policy_comm.get_distribution(self.state)
            return out.detach()