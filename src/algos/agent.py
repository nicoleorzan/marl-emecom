
from src.algos.buffer import RolloutBufferComm
import torch
import copy
import torch.nn.functional as F
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

        #self.reputation = 0.5 # change this based on initial probs
        if (params.binary_reputation == True):
            #print("rep is binary")
            #self.reputation = float(np.random.binomial(1,0.5))
            self.reputation = torch.Tensor([1.0])
        self.old_reputation = self.reputation
        #print("rep=", self.reputation)

        self.is_dummy = False
        self.previous_action = torch.Tensor([1.])

        self.idx = idx
        self.gmm_ = self.gmm_
        self.is_communicating = self.communicating_agents[self.idx]
        self.is_listening = self.listening_agents[self.idx]

        print("\nAgent", self.idx)
        print("is communicating?:", self.is_communicating)
        print("is listening?:", self.is_listening)
        print("uncertainty:", self.uncertainties[idx])
        print("gmm=", self.gmm_)

        self.buffer = RolloutBufferComm()

        self.is_uncertain = torch.Tensor([0.])
        if (self.uncertainties[idx] != 0.):
            self.is_uncertain = torch.Tensor([1.])
        self.n_communicating_agents = self.communicating_agents.count(1)
        #print("self.n_communicating_agents=",self.n_communicating_agents)

        self.max_f = max(self.mult_fact)
        self.min_f = min(self.mult_fact)
        # Action Policy
        self.input_act = self.obs_size
        print("obs_size=", self.obs_size)
        if (self.gmm_):
            self.input_act = self.n_gmm_components  #gmm components for the m factor, 1 for the coins I get
        if (self.get_index == True):
            self.input_act += self.n_agents
        if (self.get_opponent_is_uncertain == True):
            self.input_act += 1 # we just need a 0/1 variable
        print("input size act=", self.input_act)
        if (self.is_listening and self.n_communicating_agents != 0):
            self.input_act += self.mex_size #*self.n_communicating_agents IN THIS CASE ONLY 2 AGENT INTERACT WITH EACH OTHER!
    
        # Communication Policy
        if (self.is_communicating):
            self.input_comm = self.obs_size
            print("0 self.input_comm ",  self.input_comm)
            if (self.get_index == True):
                self.input_comm += self.n_agents
                print("1 self.input_comm ",  self.input_comm)
            if (self.get_opponent_is_uncertain == True):
                self.input_comm += 1 # we just need a 0/1 variable
            if (self.gmm_):
                self.input_comm = self.n_gmm_components #gmm components for the m factor, 1 for the coins I get
                print("2 self.input_comm ",  self.input_comm)
            print("input size comm=", self.input_comm)

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
        self.sc_for_opponent = {}
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

        obs_m_fact, opponent_reputation = input

        ##### set multiplier factor observation (if uncertainty is modeled, gmm is used)
        digested_m_factor = self.set_mult_fact_obs(obs_m_fact)

        opponent_reputation = torch.Tensor([opponent_reputation])
        my_reputation = torch.Tensor([self.reputation])
        self.state_act = torch.cat((digested_m_factor, opponent_reputation, my_reputation), 0)
        #print("self.state_act=", self.state_act)

        if (self.is_communicating):
            self.state_comm = torch.cat((digested_m_factor, opponent_reputation, my_reputation), 0)

    def digest_input_no_reputation(self, input):
        obs_m_fact, opponent_is_uncertain = input
        digested_m_factor = self.set_mult_fact_obs(obs_m_fact)
        self.state_act = torch.cat((digested_m_factor, opponent_is_uncertain), 0)
        #print("self.state_act=", self.state_act)

        if (self.is_communicating):
            self.state_comm = torch.cat((digested_m_factor, opponent_is_uncertain), 0)
            #print("self.state_comm=", self.state_comm)

    # ===============
    # Abstract functions implemented in the specific algorithmic models
    def embed_opponent_idx_act(self, idx):
        pass

    def embed_opponent_idx_comm(self, idx):
        pass
    # ===============

    def set_mult_fact_obs(self, obs_m_fact, _eval=False):

        ##### set the iternal state of the agent regarding the observation of the m factor, called self.state_in
        obs_m_fact = torch.FloatTensor(obs_m_fact).to(device)
        if (self.gmm_):
            # If I have uncerainty and use gmm_, the obs of f will become a tensor with probabilities for every posisble m
            obs_m_fact = self.set_gmm_state(obs_m_fact, _eval)
        else:
            # otherwise I just need to normalize the observed f
            obs_m_fact = (obs_m_fact - self.min_f)/(self.max_f - self.min_f + 0.0001)
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
    
    def select_opponent(self, reputations, _eval=False):

        if (_eval == True):
            with torch.no_grad():
                opponent_out, opponent_logprob, entropy = self.policy_opponent_selection.act(reputations)

        elif(_eval == False):
            opponent_out, opponent_logprob, entropy = self.policy_opponent_selection.act(reputations)
            #("opponent_logprob=",opponent_logprob)
            if (opponent_out != self.idx):
                self.buffer.reputations.append(reputations)
                self.buffer.opponent_choices.append(opponent_out)
                self.buffer.opponent_logprobs.append(opponent_logprob)

        return opponent_out

    def select_message(self, m_val=None, _eval=False):
        #print("state for mex=", self.state_comm)

        if (_eval == True):
            with torch.no_grad():
                message_out, message_logprob, entropy = self.policy_comm.act(self.state_comm)

        elif (_eval == False):
            #print("self.state_comm", self.state_comm)
            #print(self.policy_comm.act(self.state_comm))
            message_out, message_logprob, entropy = self.policy_comm.act(self.state_comm)

            self.buffer.states_c.append(self.state_comm)
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

    def select_action(self, epsilon = 0., m_val=None, _eval=False):         
        #print("qui")   
        
        state_to_act = self.state_act
        print("agent=", self.idx, "is selecting action")
        if (self.is_listening and self.n_communicating_agents != 0):
            state_to_act = torch.cat((self.state_act, self.message_in)).to(device)
        print("state_act=", state_to_act)
            
        if (_eval == True):
            with torch.no_grad():
                action, action_logprob, entropy = self.policy_act.act(state=state_to_act)

        elif (_eval == False):
            print("eval false")
            action, action_logprob, entropy, distrib = self.policy_act.act(state=state_to_act, greedy=False, get_distrib=True)
            print('action=', action, "distrib=", distrib)
            if (self.is_listening == True and self.n_communicating_agents != 0.):
                #print("self.state_act=", self.state_act)
                state_empty_mex = torch.cat((self.state_act.detach(), torch.zeros_like(self.message_in))).to(device)
                #print("state_empty_mex=", state_empty_mex)
                dist_empty_mex = self.policy_act.get_distribution(state_empty_mex).detach()
                #print("dist_empty_mex=", dist_empty_mex)
                #dist_mex = self.policy_act.get_distribution(state_to_act.detach()) #IL PROBLEMA E` QUI`
                #print("dist_mex=", dist_mex)
                self.List_loss_list.append(-torch.sum(torch.abs(dist_empty_mex - distrib)))

            self.buffer.states_a.append(state_to_act)
            self.buffer.actions.append(action)
            if (m_val in self.buffer.actions_given_m):
                self.buffer.actions_given_m[m_val].append(action)
            else: 
                self.buffer.actions_given_m[m_val] = [action]

            self.buffer.act_logprobs.append(action_logprob)
            self.buffer.act_entropy.append(entropy)
        print("return")
        return action, action_logprob
    
    def get_action_distribution(self):

        if (self.is_listening == True and self.n_communicating_agents != 0.):
            self.state_act = torch.cat((self.state_act, self.message_in)).to(device)

        with torch.no_grad():
            out = self.policy_act.get_distribution(self.state_act)
            return out

    def get_message_distribution(self):
        with torch.no_grad():
            out = self.policy_comm.get_distribution(self.state_comm)
            return out.detach()