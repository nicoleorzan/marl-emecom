
import torch
import torch.autograd as autograd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import torch.nn.functional as F
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

    def __init__(self, model, optimizer, params, idx):

        for key, val in params.items():  setattr(self, key, val)
    
        self.policy = model.to(device)
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decayRate)
        
        self.train_returns_norm = []
        self.return_episode = 0
        self.return_episode_norm= 0
        self.tmp_actions = []
        self.coop = []
        self.saved_losses = []

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
            # normalize m factor
            #print("state1 before", state)
            state[1] = self.normalize_m_factor(state[1])
            #print("state1 after", state)
        if (eval == True):
            with torch.no_grad():
                action, action_logprob = self.policy.act(state)
        elif (eval == False):
            action, action_logprob = self.policy.act(state)

            self.logprobs.append(action_logprob)

        return action

    def normalize_m_factor(self, obs_multiplier):
        
        obs_multiplier_norm = (obs_multiplier - self.min_observable_mult)/(self.max_observable_mult - self.min_observable_mult)
        if (obs_multiplier_norm < 0.):
            obs_multiplier_norm = torch.Tensor([0.])
        elif (obs_multiplier_norm > 1.):
            obs_multiplier_norm = torch.Tensor([1.])
        return obs_multiplier_norm

    def get_distribution(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if (self.gmm_):
                state = self.get_gmm_state(state, eval=True)
            else:
                #print("state1 before", state)
                state[1] = self.normalize_m_factor(state[1])
                #print("state1 after", state)
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
            p = torch.Tensor(self.gmm.predict(state[1].reshape(1).reshape(-1, 1))).long()
            #print("self.gmm.means=", self.gmm.means_)
            #print("shape=", self.gmm.means_.shape)
            #print("prediction=", p)
            value_to_feed = self.gmm.means_[p]
            #print("value_to_feed=", value_to_feed)
            self.gmm_probs = self.gmm.predict_proba(state[1].reshape(1).reshape(-1, 1))[0]
            #print("probs=", self.gmm_probs)
            #state_in = torch.FloatTensor(np.array(self.gmm_probs))
            #state_in = F.one_hot(p, num_classes=len(self.mult_fact))[0].to(torch.float32)
            #print("state=", state_in)


            self.means = copy.deepcopy(self.gmm.means_)
            self.means = np.sort(self.means, axis=0)
            #print("means ordered=", means_ordered)
            ordering_values = [np.where(self.means == i)[0][0] for i in self.gmm.means_]
            #print("sort values=", ordering_values)
            self.probs = torch.zeros(len(self.gmm_probs)).to(device)
            for i, value in enumerate(ordering_values):
                self.probs[value] = self.gmm_probs[i] 
            #print("sort probs=", ordered_probs)
            state_in = self.probs
            #print("state=", state_in)

        else: 
            state_in = torch.zeros(len(self.mult_fact)).to(device)

        return state_in