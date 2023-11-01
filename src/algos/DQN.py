from src.algos.agent import Agent
from src.nets.Actor import Actor
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import random
from torch.distributions import Categorical
import numpy as np
import copy

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class ExperienceReplayMemory:
    def __init__(self, params, input_act, input_comm=0):

        for key, val in params.items(): setattr(self, key, val)
        self.capacity = self.batch_size #dqn_capacity
        self.input_act = input_act
        self.input_comm = input_comm
        self.reset()

    def reset(self):
        self._states_comm = torch.empty((self.capacity,self.input_comm))
        self._states_act = torch.empty((self.capacity,self.input_act))
        self._messages = torch.empty((self.capacity,1))
        self._actions = torch.empty((self.capacity,1))
        self._rewards = torch.empty((self.capacity,1))
        #self._next_states = torch.empty((self.capacity,self.input_state))
        self._dones = torch.empty((self.capacity,1), dtype=torch.bool)
        self.i = 0

    def __len__(self):
        return len(self._states)
    

class DQN(Agent):

    def __init__(self, params, idx=0):
        Agent.__init__(self, params, idx)

        opt_params = []
        print("self.input_act=",self.input_act)
        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, gmm=self.gmm_).to(device)
        self.policy_act_target = copy.deepcopy(self.policy_act).to(device)
        self.policy_act_target.load_state_dict(self.policy_act.state_dict())

        opt_params_act = {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor}
        opt_params.append(opt_params_act)

        # Communication Policy
        if (self.is_communicating):
            print("self.input_comm=", self.input_comm)
            self.policy_comm = Actor(params=params, input_size=self.input_comm, output_size=self.mex_size, \
                n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, gmm=self.gmm_).to(device)
            self.policy_comm_target = copy.deepcopy(self.policy_comm).to(device)
            self.policy_comm_target.load_state_dict(self.policy_comm.state_dict())
            opt_params_comm = {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm}
            opt_params.append(opt_params_comm)
            self.memory = ExperienceReplayMemory(params, self.input_act, self.input_comm)
        else: 
            self.memory = ExperienceReplayMemory(params, self.input_act)

        self.optimizer = torch.optim.Adam(opt_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.htarget = np.log(self.action_size)/2.
        self.n_update = 0.
        self.baseline = 0.

        self.update_count = 0
        self.eps0 = 0.1
        self.final_epsilon = 0.001
        self.epsilon_delta = (self.eps0 - self.final_epsilon)/self.n_epochs
        self.epsilon = self.eps0
        self.r = 1.-np.exp(np.log(self.final_epsilon/self.eps0)/self.n_epochs)

        self.message_out = 0

    def append_to_replay(self, s, m, s1, a, r, d):
        #print("agent=", self.idx)
        #print("self.input_act=",self.input_act)
        #print("s=",s, "s1=", s1)
        if (self.is_communicating):
            self.memory._states_comm[self.memory.i] = s
            #self.memory._states_act[self.memory.i] = s1
            self.memory._messages[self.memory.i] = m
        if (self.is_listening):
            self.memory._states_act[self.memory.i] = s1
        else:
            self.memory._states_act[self.memory.i] = s
        
        self.memory._actions[self.memory.i] = a
        self.memory._rewards[self.memory.i] = r
        self.memory._dones[self.memory.i] = d
        self.memory.i += 1

    def argmax(self, q_values):
        top = torch.Tensor([-10000000])
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return random.choice(ties)

    def act(self, policy, state, input_size, greedy=False):
        state = state.view(-1,input_size)
        values = policy.get_values(state=state)[0]

        if (greedy == True):
            action = self.argmax(values)
            
        elif (greedy == False):
            if torch.rand(1) < self.epsilon:
                action = random.choice([i for i in range(self.action_size)])
            else:
                action = self.argmax(values)

        dist = Categorical(logits=values)
        entropy = dist.entropy().detach()
                
        return torch.Tensor([action]), entropy
       
    def select_message(self, m_val=None, _eval=False):

        if (_eval == True):
            with torch.no_grad():
                message_out, mex_entropy = self.act(self.policy_comm, self.state, self.input_comm)

        elif (_eval == False):
            message_out, mex_entropy = self.act(self.policy_comm, self.state, self.input_comm)

            self.buffer.states_c.append(self.state)
            self.buffer.messages.append(message_out)
            self.buffer.comm_entropy.append(mex_entropy)
            if (m_val in self.buffer.messages_given_m):
                self.buffer.messages_given_m[m_val].append(message_out)
            else: 
                self.buffer.messages_given_m[m_val] = [message_out]

        message_out = torch.Tensor([message_out.item()]).long().to(device)
        self.message_out = message_out
        message_out = F.one_hot(message_out, num_classes=self.mex_size)[0]
        #print("message_out=", message_out)
        return message_out

    def select_action(self, m_val=None, _eval=False):
            
        self.state_to_act = self.state
        if (self.is_listening):
            self.state_to_act = torch.cat((self.state, self.message_in)).to(device)
        #print("state_to_act agent", self.idx, "=", self.state_to_act)

        if (_eval == True):
            with torch.no_grad():
                action, act_entropy = self.act(self.policy_act, self.state_to_act, self.input_act)

        elif (_eval == False):
            #print("input act net=",self.state_to_act)
            action, act_entropy = self.act(self.policy_act, self.state_to_act, self.input_act)
            
            if (self.is_listening == True and self.n_communicating_agents != 0.):
                state_empty_mex = torch.cat((self.state, torch.zeros_like(self.message_in))).to(device)
                out = self.policy_act.get_values(state_empty_mex).detach()
                dist_empty_mex = self.softmax(out)
                out = self.policy_act.get_values(self.state_to_act)
                dist_mex = self.softmax(out)
                self.List_loss_list.append(-torch.sum(torch.abs(dist_empty_mex - dist_mex)))

            self.buffer.states_a.append(self.state_to_act)
            self.buffer.actions.append(action)
            self.buffer.act_entropy.append(act_entropy)
            if (m_val in self.buffer.actions_given_m):
                self.buffer.actions_given_m[m_val].append(action)
            else: 
                self.buffer.actions_given_m[m_val] = [action]

        self.action = action[0]
        #print("action agent", self.idx, "=", self.action)
        return self.action
    
    def compute_loss(self):

        batch_state_act = self.memory._states_act
        batch_state_comm = self.memory._states_comm
        batch_messages = self.memory._messages.long()
        batch_action = self.memory._actions.long()
        #print("batch_action=",batch_action)
        batch_reward = self.memory._rewards
        #print('batch_state_act=',batch_state_act)
    
        current_q_values_act = self.policy_act.get_values(batch_state_act)
        current_q_values_act = torch.gather(current_q_values_act, dim=1, index=batch_action)
        #print("current_q_values_act=", current_q_values_act.shape)

        if (self.is_communicating):
            #for state in batch_state_comm:
            #    print("state=", state)
            entropy = torch.FloatTensor([self.policy_comm.get_dist_entropy(state).detach() for state in batch_state_comm])
            #print("entropy=", entropy)
            self.entropy = entropy
            hloss = (torch.full(entropy.size(), self.htarget) - entropy) * (torch.full(entropy.size(), self.htarget) - entropy)
            #print("hloss=", hloss.shape)
            current_q_values_comm = self.policy_comm.get_values(batch_state_comm)
            #print("PRIMA: current_q_values_comm=",current_q_values_comm)
            #print("batch_messages=",batch_messages)
            current_q_values_comm = torch.gather(current_q_values_comm, dim=1, index=batch_messages)
            #print("DOPO: current_q_values_comm=",current_q_values_comm)
            #print("current_q_values_comm", current_q_values_comm.shape)

        #compute target
        with torch.no_grad():
            expected_q_values_act = batch_reward # + self.gamma*max_next_q_values
            if (self.is_communicating):
                expected_q_values_comm = batch_reward
                diff_comm = (expected_q_values_comm - current_q_values_comm) + self.sign_lambda*hloss
                print("diff_comm=", diff_comm.shape)
                loss_comm = self.MSE(diff_comm)
                loss_comm = loss_comm.mean()

        diff_act = (expected_q_values_act - current_q_values_act)
        loss = self.MSE(diff_act)
        loss = loss.mean()
        self.saved_losses.append(loss.detach())

        if (self.is_communicating):
            loss = loss + loss_comm

        return loss

    def update(self):

        entropy = torch.FloatTensor([self.policy_comm.get_dist_entropy(state).detach() for state in self.buffer.states_c])
        self.entropy = entropy

        loss = self.compute_loss()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_model()

        self.reset()
        self.memory.reset()
        self.memory.i = 0
        self.scheduler.step()
        print("LR=",self.scheduler.get_last_lr())

        #modif exploration
        self.update_epsilon()
        #print("self.epsilon=",self.epsilon)

        return loss.detach()
    
    def update_epsilon(self):
        if (self.epsilon >= self.final_epsilon): 
            self.epsilon = self.epsilon -self.epsilon_delta

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            print("updating target model")
            self.policy_act_target.load_state_dict(self.policy_act.state_dict())
    
    def MSE(self, x):
        return 0.5 * x.pow(2)