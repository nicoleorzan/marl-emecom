
from src.algos.buffer import RolloutBufferComm, RolloutBuffer
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

    #def __init__(self, params, idx=0):
    def __init__(self, params, is_communicating, is_listening, gmm_, n_communicating_agents):

        for key, val in params.items(): setattr(self, key, val)

        self.is_communicating = is_communicating
        self.is_listening = is_listening
        self.gmm_ = gmm_
        self.n_communicating_agents = n_communicating_agents
        self.max_f = max(self.mult_fact)
        self.min_f = min(self.mult_fact)

        self.buffer = RolloutBufferComm()

        opt_params = []

        # Action Policy
        input_act = self.obs_size
        if (self.gmm_):
            input_act = self.n_gmm_components  #gmm components for the m factor, 1 for the coins I get
        if (self.partner_selection):
            input_act += self.n_agents # add one hot encoding of partner agent index
        if (self.is_listening):
            input_act += self.mex_size*self.n_communicating_agents

        print("input_act:", input_act)
        self.policy_act = ActorCritic(params=params, input_size=input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, mask=None, gmm=self.gmm_).to(device)
    
        opt_params_act = {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor}
        opt_params.append(opt_params_act)

        # Communication Policy
        if (self.is_communicating):
            input_comm = self.obs_size
            if (self.gmm_):
                input_comm = self.n_gmm_components #gmm components for the m factor, 1 for the coins I get

            #print("input_comm=", input_comm)
            self.policy_comm = ActorCritic(params=params, input_size=input_comm, output_size=self.mex_size, \
                n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, mask=None, gmm=self.gmm_).to(device)
            #print("output comm=", self.mex_size)
            
            opt_params_comm = {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm}
            opt_params.append(opt_params_comm)

        self.optimizer = torch.optim.Adam(opt_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.htarget = np.log(self.action_size)/2.
        self.n_update = 0.
        self.baseline = 0.

        self.eps_norm = 0.0001
        self.comm_loss = True

        #self.entropy = 0.

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
        self.reset_batch()

        self.saved_losses_comm = []
        self.saved_losses = []

    def reset(self):
        self.List_loss_list = []
        self.rewards = []
        self.mutinfo_signaling_old = self.mutinfo_signaling
        self.mutinfo_listening_old = self.mutinfo_listening
        self.sc_old = self.sc
        self.sc = []
        self.mutinfo_signaling = []
        self.mutinfo_listening = []
        self.buffer.clear()
        self.reset_episode()

    def reset_batch(self):
        self.sc_m_old = self.sc_m
        self.sc_m = {}
        self.buffer.clear_batch()

    def reset_episode(self):
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode_old = self.return_episode
        self.return_episode = 0
        self.return_episode_norm = 0

    def set_state(self, state, _eval=False):

        # set the internal state of the agent, called self.state_in
        state = torch.FloatTensor(state).to(device)
        if (self.gmm_):
            # If I have uncerainty and use gmm_, the obs of f will become a tensor with probabilities for every posisble m
            self.get_gmm_state(state, _eval)
        else:
            # otherwise I just need to normalize the observed f
            state = (state - self.min_f)/(self.max_f - self.min_f)
            self.state_to_comm = state
        self.state_to_act = self.state_to_comm

    def select_message(self, m_val=None, _eval=False):

        if (_eval == True):
            with torch.no_grad():
                message_out, message_logprob, entropy = self.policy_comm.act(self.state_to_comm)

        elif (_eval == False):
            message_out, message_logprob, entropy = self.policy_comm.act(self.state_to_comm)

            self.buffer.states_c.append(self.state_to_comm)
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

    def random_messages(self, m_val=None):

        message_out = torch.randint(0, self.mex_size, (self.mex_size-1,))[0]
        self.buffer.states_c.append(self.state_to_comm)
        self.buffer.messages.append(message_out)
        if (m_val in self.buffer.messages_given_m):
            self.buffer.messages_given_m[m_val].append(message_out)
        else: 
            self.buffer.messages_given_m[m_val] = [message_out]

        self.buffer.comm_logprobs.append(torch.tensor(0.0001))

        message_out = torch.Tensor([message_out.item()]).long().to(device)
        message_out = F.one_hot(message_out, num_classes=self.mex_size)[0]

        return message_out

    def get_message(self, message_in):
        self.message_in = message_in
        self.state_to_act = torch.cat((self.state_to_comm, self.message_in)).to(device)

    def select_action(self, partner_id = None, m_val=None, _eval=False):

        if (partner_id is not None):
            self.state_to_act = torch.cat((partner_id, self.state_to_act)).to(device)
        #print("state to act=", self.state_to_act)
            
        if (_eval == True):
            with torch.no_grad():
                action, action_logprob, entropy = self.policy_act.act(self.state_to_act)

        elif (_eval == False):
            #print("input act net=",self.state_to_act)
            action, action_logprob, entropy = self.policy_act.act(self.state_to_act)
            
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

            self.buffer.act_logprobs.append(action_logprob)
            self.buffer.act_entropy.append(entropy)
        
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
            self.means = np.sort(self.means, axis=0)
            ordering_values = [np.where(self.means == i)[0][0] for i in self.gmm.means_]
            self.state_to_comm = torch.zeros(len(self.gmm_probs)).to(device)
            for i, value in enumerate(ordering_values):
                self.state_to_comm[value] = self.gmm_probs[i] 

    def update(self):

        # I do not normalize rewards here because I already give normalized rewards to the agent
        rew_norm = self.rewards # [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]
        if (self.rewards != []):
            print("rew norm=", rew_norm)
            print("actions=", self.buffer.actions)
            print("logp=", self.buffer.act_logprobs)

            entropy = torch.FloatTensor([self.policy_comm.get_dist_entropy(state).detach() for state in self.buffer.states_c])
            self.entropy = entropy
            hloss = (torch.full(entropy.size(), self.htarget) - entropy) * (torch.full(entropy.size(), self.htarget) - entropy)
            for i in range(len(self.buffer.act_logprobs)):
                if (self.is_communicating):
                    self.buffer.comm_logprobs[i] = -self.buffer.comm_logprobs[i] * (rew_norm[i] - self.baseline) + self.sign_lambda*hloss[i]
                self.buffer.act_logprobs[i] = -self.buffer.act_logprobs[i] * (rew_norm[i] - self.baseline) 
                if (self.is_listening and self.n_communicating_agents != 0.):
                    self.buffer.act_logprobs[i] += self.list_lambda*self.List_loss_list[i]
        
            self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in self.buffer.act_logprobs])))
            
            self.optimizer.zero_grad()
            if(self.is_communicating):
                self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in self.buffer.comm_logprobs])))
                tmp = [torch.ones(a.data.shape) for a in self.buffer.comm_logprobs]
                autograd.backward(self.buffer.comm_logprobs, tmp, retain_graph=True)

            tmp1 = [torch.ones(a.data.shape) for a in self.buffer.act_logprobs]
            autograd.backward(self.buffer.act_logprobs, tmp1, retain_graph=True)
            
            self.optimizer.step()

            #diminish learning rate
            self.scheduler.step()

            self.n_update += 1.
            self.baseline += (np.mean([i[0] for i in rew_norm]) - self.baseline) / (self.n_update)

        self.reset()




class Reinforce():

    def __init__(self, params, input_size, output_size, hidden_size, mask=None):

        for key, val in params.items(): setattr(self, key, val)

        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(params=params, input_size=input_size, output_size=output_size, \
            n_hidden=1, hidden_size=hidden_size, mask=mask).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': params.lr_actor},
            ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.n_update = 0.
        self.baseline = 0.

        self.eps_norm = 0.0001
        self.return_episode_norm = 0
        self.return_episode = 0

        self.saved_losses = []

        self.reset()

    def reset(self):
        self.rewards = []
        self.buffer.clear()
        self.reset_episode()

    def reset_batch(self):
        self.buffer.clear_batch()

    def reset_episode(self):
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode_old = self.return_episode
        self.return_episode = 0
        self.return_episode_norm = 0

    def select_action(self, _input, _eval=False):

        if (_eval == True):
            with torch.no_grad():
                out, logprob, entropy = self.policy.act(_input)

        elif (_eval == False):
            out, logprob, entropy = self.policy.act(_input)

            self.buffer.states.append(_input)
            self.buffer.actions.append(out)
            
            self.buffer.logprobs.append(logprob)
            self.buffer.entropy.append(entropy)

        return out

    def get_action_distribution(self, _input):

        with torch.no_grad():
            out = self.policy.get_distribution(_input)
            return out

    def update(self):

        rew_norm = self.rewards
        if (self.rewards != []):
            print("rew norm=", rew_norm)
            print("actions=", self.buffer.actions)
            print("logp=", self.buffer.logprobs)

            entropy = torch.FloatTensor([self.policy.get_dist_entropy(state).detach() for state in self.buffer.states]) #
            self.entropy = entropy
            #hloss = (torch.full(entropy.size(), self.htarget) - entropy) * (torch.full(entropy.size(), self.htarget) - entropy)
            for i in range(len(self.buffer.logprobs)):
                self.buffer.logprobs[i] = -self.buffer.logprobs[i] * (rew_norm[i] - self.baseline) 
        
            self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in self.buffer.logprobs])))
            
            self.optimizer.zero_grad()
            tmp1 = [torch.ones(a.data.shape) for a in self.buffer.logprobs]
            autograd.backward(self.buffer.logprobs, tmp1, retain_graph=True)
            self.optimizer.step()

            #diminish learning rate
            self.scheduler.step()

            self.n_update += 1.
            self.baseline += (np.mean([i[0] for i in rew_norm]) - self.baseline) / (self.n_update)

        self.reset()