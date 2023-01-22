
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

class ReinforceComm():

    def __init__(self, params, idx=0, gmm_agent=False):

        for key, val in params.items(): setattr(self, key, val)

        self.buffer = RolloutBufferComm()

        self.idx = idx
        self.gmm_ = gmm_agent
    
        # Communication Policy and Action Policy
        input_comm = self.obs_size
        if (self.gmm_):
            input_comm = self.n_gmm_components
        self.policy_comm = ActorCritic(params=params, input_size=input_comm, output_size=self.mex_size, \
            n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, gmm=self.gmm_).to(device)

        input_act = self.obs_size# + self.n_agents*self.mex_size
        if (self.gmm_):
            input_act = self.n_gmm_components# + self.n_agents*self.mex_size
        if ("listening_agent" in params):
            if (self.listening_agents[self.idx]):
                input_act += self.mex_size*self.communcating_agents.count(1)
    
        self.policy_act = ActorCritic(params=params, input_size=input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, gmm=self.gmm_).to(device)

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm},
                        {'params': self.policy_comm.critic.parameters(), 'lr': self.lr_critic_comm},
                        {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy_act.critic.parameters(), 'lr': self.lr_critic}])
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
        self.sc_old = self.sc
        self.sc = []
        self.mutinfo_signaling = []
        self.mutinfo_listening = []
        self.buffer.clear()

    def reset_episode(self):
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode_old = self.return_episode
        self.return_episode = 0
        self.return_episode_norm = 0

    def select_message(self, state, eval=False):

        state = torch.FloatTensor(state).to(device)
        if (self.gmm_):
            self.get_gmm_state(state, eval)
            # creates new state called state_in
        else: 
            self.state_in = state

        if (eval == True):
            with torch.no_grad():
                message, message_logprob, entropy = self.policy_comm.act(self.state_in, self.ent)

        elif (eval == False):
            message, message_logprob, entropy = self.policy_comm.act(self.state_in, self.ent)

            self.buffer.states_c.append(self.state_in)
            self.buffer.messages.append(message)

            self.comm_logprobs.append(message_logprob)
            self.comm_entropy.append(entropy)

        message = torch.Tensor([message.item()]).long().to(device)
        message = F.one_hot(message, num_classes=self.mex_size)[0]
        return message

    def random_messages(self, state):

        state = torch.FloatTensor(state).to(device)
        if (self.gmm_):
            self.get_gmm_state(state, eval)
        else: 
            self.state_in = state
        message = torch.randint(0, self.mex_size, (self.mex_size-1,))[0]

        self.buffer.states_c.append(self.state_in)
        self.buffer.messages.append(message)

        self.comm_logprobs.append(torch.tensor(0.0001))

        message = torch.Tensor([message.item()]).long().to(device)
        message = F.one_hot(message, num_classes=self.mex_size)[0]

        return message

    def select_action(self, state, message=None, eval=False):

        state = torch.FloatTensor(state).to(device)
        #if (self.gmm_):
        #    self.get_gmm_state(state, eval)
        if (self.gmm_ == False): 
            self.state_in = state
        # otherwise I already crated state_in with gmm, in the comm policy
    
        if (message is not None):
            state = torch.cat((self.state_in, message)).to(device)
            
        if (eval == True):
            with torch.no_grad():
                #state_mex = torch.cat((self.state_in, message)).to(device)
                action, action_logprob, entropy = self.policy_act.act(state, self.ent)

        elif (eval == False):
            #state_mex = torch.cat((self.state_in, message)).to(device)
            action, action_logprob, entropy = self.policy_act.act(state, self.ent)
            
            if (self.comm_loss == True and self.listening_agents[self.idx] == True):
                state_no_mex = torch.cat((self.state_in, torch.zeros_like(message))).to(device)
                dist_nomex = self.policy_act.get_distribution(state_no_mex).detach()
                dist_mex = self.policy_act.get_distribution(state)
                self.List_loss_list.append(-torch.sum(torch.abs(dist_nomex - dist_mex)))

            self.buffer.states_a.append(state)
            self.buffer.actions.append(action)

            self.act_logprobs.append(action_logprob)
            self.act_entropy.append(entropy)
        
        return action

    def get_action_distribution(self, state, message=None):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if (self.gmm_):
                self.get_gmm_state(state, eval)
            else: 
                self.state_in = state
            if (message is not None):
                self.state_in = torch.cat((self.state_in, message))
            out = self.policy_act.get_distribution(self.state_in)

            return out

    def get_message_distribution(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if (self.gmm_):
                self.get_gmm_state(state, eval)
            else: 
                self.state_in = state
            out = self.policy_comm.get_distribution(self.state_in)

            return out.detach()

    def update(self):

        rewards = self.rewards
        rew_norm = [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]

        entropy = torch.FloatTensor([self.policy_comm.get_dist_entropy(state).detach() for state in self.buffer.states_c])
        self.entropy = entropy
        hloss = (torch.full(entropy.size(), self.htarget) - entropy) * (torch.full(entropy.size(), self.htarget) - entropy)
        for i in range(len(self.comm_logprobs)):
            self.comm_logprobs[i] = -self.comm_logprobs[i] * (rew_norm[i] - self.baseline) + self.sign_lambda*hloss[i]
            self.act_logprobs[i] = -self.act_logprobs[i] * (rew_norm[i] - self.baseline) 
            if (self.listening_agents[self.idx] == 1.):
                self.act_logprobs[i] += self.list_lambda*self.List_loss_list[i]
       
        self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in self.comm_logprobs])))
        self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in self.act_logprobs])))

        self.optimizer.zero_grad()
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

    def get_gmm_state(self, state, eval=False):

        if hasattr(self, 'mf_history'):
            if (len(self.mf_history) < 50000 and eval == False):
                self.mf_history = torch.cat((self.mf_history, state[1].reshape(1)), 0)
        else:
            self.mf_history = state[1].reshape(1)

        if (len(self.mf_history) >= self.n_gmm_components):
            if(len(self.mf_history) < 50000):
                self.gmm = GMM(n_components = self.n_gmm_components, max_iter=1000, random_state=0, covariance_type = 'full')
                input_ = self.mf_history.reshape(-1, 1)
                self.gmm.fit(input_)

            self.gmm_probs = self.gmm.predict_proba(state[1].reshape(1).reshape(-1, 1))[0]

            self.means = copy.deepcopy(self.gmm.means_)
            self.means = np.sort(self.means, axis=0)
            ordering_values = [np.where(self.means == i)[0][0] for i in self.gmm.means_]
            self.probs = torch.zeros(len(self.gmm_probs)).to(device)
            for i, value in enumerate(ordering_values):
                self.probs[value] = self.gmm_probs[i] 
            self.state_in = self.probs
        else: 
            self.state_in = torch.zeros(len(self.mult_fact)).to(device)