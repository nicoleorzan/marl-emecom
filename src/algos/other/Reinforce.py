
from src.algos.buffer import RolloutBufferComm
from src.nets.ActorCritic import ActorCritic
import torch
import torch.autograd as autograd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import copy

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class Reinforce():

    def __init__(self, params, idx, gmm_agent):

        for key, val in params.items():  setattr(self, key, val)

        self.idx = idx
        self.gmm_ = gmm_agent

        self.buffer = RolloutBufferComm()

        # get observations size
        self.input_size = params.obs_size
        #if (self.gmm_):
        #    self.input_size = self.n_gmm_components
        # get message
        if ("listening_agents" in params):
            if (self.listening_agents[self.idx]):
                self.input_size += self.mex_size*self.communicating_agents.count(1)
        print("input size=", self.input_size)
        self.policy = ActorCritic(params=params, input_size=self.input_size, output_size=self.action_size, \
            n_hidden=self.n_hidden, hidden_size=self.hidden_size, gmm=self.gmm_).to(device)

        self.optimizer = torch.optim.Adam([
             {'params': self.policy.actor.parameters(), 'lr': params.lr_actor},
             {'params': self.policy.critic.parameters(), 'lr': params.lr_critic} 
             ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)
        
        self.train_returns_norm = []
        self.train_returns = []
        self.return_episode = 0
        self.return_episode_norm= 0
        self.tmp_actions = []
        self.coop = []
        self.saved_losses = []
        self.mutinfo_listening = []

        self.reset()

        self.eps_norm = 0.0001

        self.idx = idx
        self.gmm_ = self.policy.gmm_

        self.z_value = 1
        self.min_mult = min(self.mult_fact)
        self.max_mult = max(self.mult_fact)
        self.min_observable_mult = self.min_mult - self.z_value*self.uncertainties[idx]
        self.max_observable_mult = self.max_mult + self.z_value*self.uncertainties[idx]
        self.means = np.zeros(len(self.mult_fact))
        self.probs = np.zeros(len(self.mult_fact))
        
    def reset(self):
        self.logprobs = []
        self.rewards = []
        self.buffer.clear()
        self.mutinfo_listening_old = self.mutinfo_listening
        self.mutinfo_listening = []

    def reset_episode(self):
        self.return_episode_old = self.return_episode
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode = 0
        self.return_episode_norm = 0
        self.tmp_actions_old = self.tmp_actions
        self.tmp_actions = []

    def select_action(self, state, eval=False):

        state = torch.FloatTensor(state).to(device)
        if (self.gmm_):
            state = self.get_gmm_state(state, eval)
        else:
            state[1] = self.normalize_m_factor(state[1])
            
        if (eval == True):
            with torch.no_grad():
                action, action_logprob = self.policy.act(state)
        elif (eval == False):
            action, action_logprob = self.policy.act(state)

            self.logprobs.append(action_logprob)
        self.buffer.actions.append(action)

        return action

    def normalize_m_factor(self, obs_multiplier):
        
        obs_multiplier_norm = (obs_multiplier - self.min_observable_mult)/(self.max_observable_mult - self.min_observable_mult)
        if (obs_multiplier_norm < 0.):
            obs_multiplier_norm = torch.Tensor([0.])
        elif (obs_multiplier_norm > 1.):
            obs_multiplier_norm = torch.Tensor([1.])
        return obs_multiplier_norm

    def get_action_distribution(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if (self.gmm_):
                state = self.get_gmm_state(state, eval=True)
            else:
                state[1] = self.normalize_m_factor(state[1])
            out = self.policy.get_distribution(state)

            return out

    def update(self):

        rewards =  self.rewards
        rew_norm = [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]

        for i in range(len(self.logprobs)):
            self.logprobs[i] = -self.logprobs[i] * rew_norm[i]

        self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in self.logprobs])))
        
        self.optimizer.zero_grad()
        tmp = [torch.ones(a.data.shape) for a in self.logprobs]
        autograd.backward(self.logprobs, tmp, retain_graph=True)
        self.optimizer.step()

        self.scheduler.step()

        self.reset()

    def get_gmm_state(self, state, eval=False):

        if hasattr(self, 'mf_history'):
            if (len(self.mf_history) < 50000 and eval == False):
                self.mf_history = torch.cat((self.mf_history, state[1].reshape(1)), 0)
        else:
            self.mf_history = state[1].reshape(1)

        if (len(self.mf_history) >= len(self.mult_fact)):
            if(len(self.mf_history) < 50000):
                self.gmm = GMM(n_components = len(self.mult_fact), max_iter=1000, random_state=0, covariance_type = 'full')
                input_ = self.mf_history.reshape(-1, 1)
                self.gmm.fit(input_)
            self.gmm_probs = self.gmm.predict_proba(state[1].reshape(1).reshape(-1, 1))[0]

            self.means = copy.deepcopy(self.gmm.means_)
            self.means = np.sort(self.means, axis=0)
            ordering_values = [np.where(self.means == i)[0][0] for i in self.gmm.means_]
            self.probs = torch.zeros(len(self.gmm_probs)).to(device)
            for i, value in enumerate(ordering_values):
                self.probs[value] = self.gmm_probs[i] 
            state_in = self.probs
        else: 
            state_in = torch.zeros(len(self.mult_fact)).to(device)

        return state_in