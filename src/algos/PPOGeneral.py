
from src.algos.buffer import RolloutBufferComm
from src.nets.ActorCritic import ActorCritic
import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture as GMM

#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class PPOGeneral():

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
        self.policy_act_old = copy.deepcopy(self.policy_act).to(device)
        self.policy_act_old.load_state_dict(self.policy_act.state_dict())
    
        #opt_params_act = {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor},
        #                 {'params': self.policy_act.critic.parameters(), 'lr': self.lr_critic},
        opt_params.append({'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor})
        opt_params.append({'params': self.policy_act.critic.parameters(), 'lr': self.lr_critic})
        
        #opt_params.append(opt_params_act)

        # Communication Policy
        if (self.is_communicating):
            input_comm = self.obs_size
            if (self.gmm_):
                input_comm = self.n_gmm_components #gmm components for the m factor, 1 for the coins I get

            self.policy_comm = ActorCritic(params=params, input_size=input_comm, output_size=self.mex_size, \
                n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, gmm=self.gmm_).to(device)
            self.policy_comm_old = copy.deepcopy(self.policy_comm).to(device)
            self.policy_comm_old.load_state_dict(self.policy_comm.state_dict())

            print("input_comm=", input_comm)
            print("output comm=", self.mex_size)
            
            #opt_params_comm = {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm}
            #opt_params.append(opt_params_comm)
            opt_params.append({'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm})
            opt_params.append({'params': self.policy_comm.critic.parameters(), 'lr': self.lr_critic_comm})

        self.optimizer = torch.optim.Adam(opt_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.htarget = np.log(self.action_size)/2.
        self.n_update = 0.
        self.baseline = 0.

        self.eps_norm = 0.0001
        self.comm_loss = True

        self.ent = True

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
        
        self.MseLoss = nn.MSELoss()

    def reset(self):
        self.comm_logprobs = []
        self.act_logprobs = []
        self.List_loss_list = []
        self.comm_entropy = []
        self.act_entropy = []
        self.rewards = []
        self.is_terminals = []
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

            self.buffer.comm_logprobs.append(message_logprob)
            self.buffer.comm_entropy.append(entropy)

        message_out = torch.Tensor([message_out.item()]).long().to(device)
        message_out = F.one_hot(message_out, num_classes=self.mex_size)[0]
        return message_out

    def get_message(self, message_in):
        self.message_in = message_in
        self.state_to_act = torch.cat((self.state_to_comm, self.message_in)).to(device)

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

    def eval_mex(self, state_c):
    
        messages_logprobs = []
        with torch.no_grad():
            state_c = torch.FloatTensor(state_c).to(device)
            for mex in range(self.mex_size):
                m = torch.Tensor([mex])
                _, mex_logprob, _= self.policy_comm_old.evaluate(state_c, m)
                messages_logprobs.append(mex_logprob.item())

        return messages_logprobs

    def eval_action(self, state_a, message):
    
        actions_logprobs = []
        with torch.no_grad():
            state_a = torch.FloatTensor(state_a).to(device)
            for action in range(self.action_size):
                a = torch.Tensor([action])
                _, action_logprob, _= self.policy_act_old.evaluate(state_a, message, a)
                actions_logprobs.append(action_logprob.item())

        return actions_logprobs

    def update(self):
        rewards = []
        discounted_reward = 0
        #for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        if (self.policy_act.input_size == 1):
            old_states_a = torch.stack(self.buffer.states_a, dim=0).detach().to(device)
            old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
            old_logprobs_act = torch.stack(self.buffer.act_logprobs, dim=0).detach().to(device)
        else:
            old_states_a = torch.squeeze(torch.stack(self.buffer.states_a, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            old_logprobs_act = torch.squeeze(torch.stack(self.buffer.act_logprobs, dim=0)).detach().to(device)

        if (self.is_communicating):
            if (self.policy_comm.input_size == 1):
                old_states_c = torch.stack(self.buffer.states_c, dim=0).detach().to(device)
                old_messages = torch.stack(self.buffer.messages, dim=0).detach().to(device)
                old_logprobs_comm = torch.stack(self.buffer.comm_logprobs, dim=0).detach().to(device)
            else:
                old_states_c = torch.squeeze(torch.stack(self.buffer.states_c, dim=0)).detach().to(device)
                old_messages = torch.squeeze(torch.stack(self.buffer.messages, dim=0)).detach().to(device)
                old_logprobs_comm = torch.squeeze(torch.stack(self.buffer.comm_logprobs, dim=0)).detach().to(device)
    
        for _ in range(self.K_epochs):

            if (self.is_communicating):
                logprobs_comm, dist_entropy_comm, state_values_comm = self.policy_comm.evaluate(old_states_c, old_messages)
                state_values_comm = torch.squeeze(state_values_comm)
                ratios_comm = torch.exp(logprobs_comm - old_logprobs_comm.detach())
                mutinfo = torch.tensor(self.buffer.mut_info, dtype=torch.float32).to(device)
                advantages_comm = rewards - state_values_comm.detach()
                surr1c = ratios_comm*advantages_comm
                surr2c = torch.clamp(ratios_comm, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_comm

                loss_c = (-torch.min(surr1c, surr2c) + \
                    self.c3*self.MseLoss(state_values_comm, rewards) - \
                    self.c4*dist_entropy_comm)

            #print(old_states_a[0], old_actions[0])
            logprobs_act, dist_entropy_act, state_values_act = self.policy_act.evaluate(old_states_a, old_actions)
            state_values_act = torch.squeeze(state_values_act)
            ratios_act = torch.exp(logprobs_act - old_logprobs_act.detach())
            advantages_act = rewards - state_values_act.detach()
            surr1a = ratios_act*advantages_act
            surr2a = torch.clamp(ratios_act, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_act

            loss_a = (-torch.min(surr1a, surr2a) + \
                self.c1*self.MseLoss(state_values_act, rewards) - \
                self.c2*dist_entropy_act)
            #print("loss_a=", loss_a)

            if (self.is_communicating):
                loss_a += loss_c
                #self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in self.comm_logprobs])))
                #tmp = [torch.ones(a.data.shape) for a in self.comm_logprobs]
                #autograd.backward(self.comm_logprobs, tmp, retain_graph=True)
            self.saved_losses.append(torch.mean(loss_a.detach()))

            self.optimizer.zero_grad()
            loss_a.mean().backward()
            self.optimizer.step()
            #print(self.scheduler.get_lr())

        self.scheduler.step()

        # Copy new weights into old policy
        self.policy_act_old.load_state_dict(self.policy_act.state_dict())
        if (self.is_communicating):
            self.policy_comm_old.load_state_dict(self.policy_comm.state_dict())

        # clear buffer
        self.buffer.clear()
        self.reset()