
from dis import disco
from src.nets.ActorCriticRNNcomm import ActorCriticRNNcomm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBufferComm:
    def __init__(self, recurrent = False):
        self.states_c = []
        self.states_a = []
        self.messages = []
        self.actions = []
        self.act_logprobs = []
        self.comm_logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.mut_info = []
        self.recurrent = recurrent
        if self.recurrent:
            self.hstates_c = []
            self.cstates_c = []
            self.hstates_a = []
            self.cstates_a = []
        
    def clear(self):
        del self.states_c[:]
        del self.states_a[:]
        del self.messages[:]
        del self.actions[:]
        del self.act_logprobs[:]
        del self.comm_logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        if self.recurrent:
            del self.hstates_c[:]
            del self.cstates_c[:]
            del self.hstates_a[:]
            del self.cstates_a[:]

    def __print__(self):
        print("states_c=", len(self.states_c))
        print("states_a=", len(self.states_a))
        print("messages_out=", len(self.messages))
        print("actions=", len(self.actions))
        print("act logprobs=", len(self.act_logprobs))
        print("mex_logprobs=", len(self.comm_logprobs))
        print("rewards=", len(self.rewards))
        print("is_terminals=", len(self.is_terminals))
        if self.recurrent:
            print("hstates_c=", len(self.hstates_c))
            print("cstates_c=", len(self.cstates_c))
            print("hstates_a=", len(self.hstates_a))
            print("cstates_a=", len(self.cstates_a))


class PPOcomm_recurrent():

    def __init__(self, params):

        for key, val in params.items(): setattr(self, key, val)
        
        self.hloss_lambda = 0.01
        self.htarget = np.log(self.action_size)/2.

        self.buffer = RolloutBufferComm()
    
        # Communication and Action Policy
        self.policy = ActorCriticRNNcomm(params).to(device)
        self.optimizer = torch.optim.Adam([{'params': self.policy.parameters(), 'lr': params.lr} ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.policy_old = copy.deepcopy(self.policy).to(device)

        self.MseLoss = nn.MSELoss()

        self.train_returns = []
        self.tmp_return = 0
        self.train_actions = []
        self.tmp_actions = []
        self.coop = []

    def reset(self):
        self.policy.reset_state()
        self.tmp_return = 0
        self.tmp_actions = []

    def select_message(self, state):
        #print("SELECT MESSAGE")
        #print("state=", state)

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            self.policy_old.observe(state)
            message, message_logprob = self.policy_old.speak()

            self.buffer.hstates_c.append(self.policy_old.hState)
            self.buffer.cstates_c.append(self.policy_old.cState)
            self.buffer.states_c.append(state)
            self.buffer.messages.append(message)
            self.buffer.comm_logprobs.append(message_logprob)

        #message = torch.Tensor([message.item()]).long()
        #message = F.one_hot(message, num_classes=self.mex_size)[0]
        return message

    def random_messages(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            message = torch.randint(0, self.mex_size, (self.mex_size-1,))[0]

            self.buffer.hstates_c.append(self.policy_old.hState)
            self.buffer.cstates_c.append(self.policy_old.cState)
            self.buffer.states_c.append(state)
            self.buffer.messages.append(message)
            self.buffer.comm_logprobs.append(torch.tensor(0.0001))

        #message = torch.Tensor([message.item()]).long()
        #message = F.one_hot(message, num_classes=self.mex_size)[0]
        return message

    def select_action(self, state):
        #print("SELECT ACTION")
    
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            self.policy_old.observe(state)
            action, action_logprob = self.policy_old.act()
            self.buffer.hstates_a.append(self.policy_old.hState)
            self.buffer.cstates_a.append(self.policy_old.cState)

            self.buffer.states_a.append(state)
            self.buffer.actions.append(action)
            self.buffer.act_logprobs.append(action_logprob)

        return action.item()

    def eval_messages(self, state):
    
        messages_logprobs = []
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            for mex in range(self.mex_size):
                m = torch.Tensor([mex])
                _, mex_logprob, _= self.policy_old.evaluate_mex(state, m)
                messages_logprobs.append(mex_logprob.item())

        return messages_logprobs

    def eval_action(self, state, message):
        # the state here is composed as: state, (hstate, cstate) ???
    
        actions_logprobs = []
        with torch.no_grad():
            #state = torch.FloatTensor(state).to(device) ???
            for action in range(self.action_size):
                a = torch.Tensor([action])
                _, action_logprob, _= self.policy_old.evaluate_act(state, message, a)
                actions_logprobs.append(action_logprob.item())

        return actions_logprobs

    def update(self):
        #print("UPDATE")
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards_comm = torch.unsqueeze(rewards, dim=1).repeat(1, self.communication_loops).view(-1)
        #print("rewards=", rewards)
        #print("rewards_comm=", rewards_comm)

        old_states_c = torch.squeeze(torch.stack(self.buffer.states_c, dim=0)).detach().to(device)
        old_hstates_c = torch.squeeze(torch.stack(self.buffer.hstates_c, dim=0)).detach().to(device)
        old_cstates_c = torch.squeeze(torch.stack(self.buffer.cstates_c, dim=0)).detach().to(device)
        old_states_c = (old_states_c, (old_hstates_c, old_cstates_c))

        old_states_a = torch.squeeze(torch.stack(self.buffer.states_a, dim=0)).detach().to(device)
        old_hstates_a = torch.squeeze(torch.stack(self.buffer.hstates_a, dim=0)).detach().to(device)
        old_cstates_a = torch.squeeze(torch.stack(self.buffer.cstates_a, dim=0)).detach().to(device)
        old_states_a = (old_states_a, (old_hstates_a, old_cstates_a))

        old_messages = torch.squeeze(torch.stack(self.buffer.messages, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs_act = torch.squeeze(torch.stack(self.buffer.act_logprobs, dim=0)).detach().to(device)
        old_logprobs_comm = torch.squeeze(torch.stack(self.buffer.comm_logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):

            logprobs_comm, dist_entropy_mex, state_values_comm = self.policy.evaluate_mex(old_states_c, old_messages)
            logprobs_act, dist_entropy_act, state_values_act = self.policy.evaluate_act(old_states_a, old_actions)

            state_values_act = torch.squeeze(state_values_act)
            state_values_comm = torch.squeeze(state_values_comm)

            ratios_act = torch.exp(logprobs_act - old_logprobs_act.detach())
            ratios_comm = torch.exp(logprobs_comm - old_logprobs_comm.detach())
            #print("rewards=", rewards.shape)

            advantages_act = rewards - state_values_act.detach()
            # here I have to understand if it is better to use rewards or the entropy distribution as "communication reward"
            advantages_comm = rewards_comm - state_values_comm.detach() # -dist_entropy_mex - state_values_comm.detach()

            surr1_a = ratios_act*advantages_act
            surr2_a = torch.clamp(ratios_act, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_act
            surr_a = torch.min(surr1_a, surr2_a)
            surr1_c = ratios_comm*advantages_comm
            surr2_c = torch.clamp(ratios_comm, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_comm
            surr_c = torch.min(surr1_c, surr2_c)
            #print("surr_c = ", surr_c.shape)
            #print("surr_a = ", surr_a.shape)

            loss_a = (-surr_a + self.c1*self.MseLoss(state_values_act, rewards) + self.c2*dist_entropy_act)
            loss_c = (-surr_c + self.c3*self.MseLoss(state_values_comm, rewards_comm) + self.c4*dist_entropy_mex)
            #print("loss_a=", loss_a.shape)
            #print("loss_c=", loss_c.shape)

            loss = loss_a.mean() + loss_c.mean()

            # add term to compute signaling entropy loss
            #entropy = torch.FloatTensor([self.policy_old.get_actions_entropy(state).detach() for state in old_states])
            #hloss =  (torch.full(entropy.size(), self.htarget) - entropy)* (torch.full(entropy.size(), self.htarget) - entropy)

            #loss = loss + self.hloss_lambda*hloss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.scheduler.step()
            #print(self.scheduler.get_lr())

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()