from src.algos.buffer import RolloutBufferComm
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import torch.nn as nn
import torch
import copy

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

        self.idx = idx
        self.gmm_ = self.gmm_[idx]
        self.is_communicating = self.communicating_agents[self.idx]
        self.is_listening = self.listening_agents[self.idx]

        self.softmax = nn.Softmax(dim=0)

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
        self.List_loss_list = []

    def reset_batch(self):
        self.buffer.clear_batch()
        self.sc_m_old = self.sc_m
        self.sc_m = {}

    def reset_episode(self):
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode_old = self.return_episode
        self.return_episode = 0
        self.return_episode_norm = 0

    def set_observation(self, obs, _eval=False):

        # set the internal state of the agent, called self.state_in
        obs = torch.FloatTensor(obs).to(device)
        if (self.gmm_):
            # If I have uncerainty and use gmm_, the obs of f will become a tensor with probabilities for every posisble m
            self.set_gmm_state(obs, _eval)
        else:
            # otherwise I just need to normalize the observed f
            obs = (obs - self.min_f)/(self.max_f - self.min_f)
            self.state = obs

    def set_gmm_state(self, obs, _eval=False):

        self.state = torch.zeros(self.n_gmm_components).to(device)
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
            self.state = torch.zeros(len(self.gmm_probs)).to(device)
            for i, value in enumerate(ordering_values):
                self.state[value] = self.gmm_probs[i]
    
    def get_message(self, message_in):
        self.message_in = message_in
    
    def get_action_distribution(self):
        self.state_to_act = self.state
        if (self.is_listening):
            self.state_to_act = torch.cat((self.state, self.message_in)).to(device)

        with torch.no_grad():
            out = self.policy_act.get_values(self.state_to_act)
            out = self.softmax(out)
            return out
        
    def get_message_distribution(self):
        with torch.no_grad():
            out = self.policy_comm.get_values(self.state)
            out = self.softmax(out)
            return out.detach()