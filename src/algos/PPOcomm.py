
from src.algos.buffer import RolloutBufferComm
from src.nets.ActorCritic import ActorCritic
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

class PPOcomm():

    def __init__(self, params):

        for key, val in params.items(): setattr(self, key, val)
        
        self.hloss_lambda = 0.01
        self.htarget = np.log(self.action_size)/2.

        self.buffer = RolloutBufferComm()
    
        # Communication Policy and Action Policy
        input_comm = self.obs_size
        output_comm = self.mex_size
        self.policy_comm = ActorCritic(params, input_comm, output_comm).to(device)
        input_act = self.obs_size + self.n_agents*self.mex_size
        output_act = self.action_size
        self.policy_act = ActorCritic(params, input_act, output_act).to(device)

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy_comm.critic.parameters(), 'lr': self.lr_critic},
                        {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy_act.critic.parameters(), 'lr': self.lr_critic}])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.policy_comm_old = copy.deepcopy(self.policy_comm).to(device)
        self.policy_act_old = copy.deepcopy(self.policy_act).to(device)

        self.MseLoss = nn.MSELoss()

        self.train_returns = []
        self.tmp_return = 0
        self.train_actions = []
        self.tmp_actions = []
        self.coop = []

    def reset(self):
        self.tmp_return = 0
        self.tmp_actions = []

    def select_message(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            message, message_logprob = self.policy_comm_old.act(state)

            self.buffer.states_c.append(state)
            self.buffer.messages.append(message)
            self.buffer.comm_logprobs.append(message_logprob)

        message = torch.Tensor([message.item()]).long()
        message = F.one_hot(message, num_classes=self.mex_size)[0]
        return message

    def random_messages(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            message = torch.randint(0, self.mex_size, (self.mex_size-1,))[0]

            self.buffer.states_c.append(state)
            self.buffer.messages.append(message)
            self.buffer.comm_logprobs.append(torch.tensor(0.0001))

        message = torch.Tensor([message.item()]).long()
        message = F.one_hot(message, num_classes=self.mex_size)[0]
        return message

    def select_action(self, state, message):
    
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            state_mex = torch.cat((state, message))
            action, action_logprob = self.policy_act_old.act(state_mex)

            self.buffer.states_a.append(state_mex)
            self.buffer.actions.append(action)
            self.buffer.act_logprobs.append(action_logprob)

        return action.item()

    def eval_mex(self, state):
    
        messages_logprobs = []
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            for mex in range(self.mex_size):
                m = torch.Tensor([mex])
                _, mex_logprob, _= self.policy_comm_old.evaluate(state, m)
                messages_logprobs.append(mex_logprob.item())

        return messages_logprobs

    def eval_action(self, state, message):
    
        actions_logprobs = []
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            for action in range(self.action_size):
                a = torch.Tensor([action])
                _, action_logprob, _= self.policy_act_old.evaluate(state, message, a)
                actions_logprobs.append(action_logprob.item())

        return actions_logprobs

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):#, reversed(self.buffer.mut_info)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        if (self.policy_comm.input_size == 1):
            old_states_c = torch.stack(self.buffer.states_c, dim=0).detach().to(device)
            old_states_a = torch.stack(self.buffer.states_a, dim=0).detach().to(device)
            old_messages = torch.stack(self.buffer.messages, dim=0).detach().to(device)
            old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
            old_logprobs_act = torch.stack(self.buffer.act_logprobs, dim=0).detach().to(device)
            old_logprobs_comm = torch.stack(self.buffer.comm_logprobs, dim=0).detach().to(device)
        else:
            old_states_c = torch.squeeze(torch.stack(self.buffer.states_c, dim=0)).detach().to(device)
            old_states_a = torch.squeeze(torch.stack(self.buffer.states_a, dim=0)).detach().to(device)
            old_messages = torch.squeeze(torch.stack(self.buffer.messages, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            old_logprobs_act = torch.squeeze(torch.stack(self.buffer.act_logprobs, dim=0)).detach().to(device)
            old_logprobs_comm = torch.squeeze(torch.stack(self.buffer.comm_logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
  
            logprobs_comm, dist_entropy_mex, state_values_comm = self.policy_comm.evaluate(old_states_c, old_messages)
            logprobs_act, dist_entropy_act, state_values_act = self.policy_act.evaluate(old_states_a, old_actions)

            state_values_act = torch.squeeze(state_values_act)
            state_values_comm = torch.squeeze(state_values_comm)

            ratios_act = torch.exp(logprobs_act - old_logprobs_act.detach())
            ratios_comm = torch.exp(logprobs_comm - old_logprobs_comm.detach())

            advantages_act = rewards - state_values_act.detach()
            # here I have to understand if it is better to use rewards or the entropy distribution as "communication reward"
            advantages_comm = -dist_entropy_mex - state_values_comm.detach()

            surr1 = ratios_act*advantages_act + ratios_comm*advantages_comm
            surr2 = torch.clamp(ratios_act, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_act
            surr3 = torch.clamp(ratios_comm, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_comm
            
            surr12 = torch.min(surr1, surr2)

            loss = (-torch.min(surr12, surr3) + \
                self.c1*self.MseLoss(state_values_act, rewards) + self.c2*dist_entropy_act + \
                self.c3*self.MseLoss(state_values_comm, rewards) + self.c4*dist_entropy_mex)

            # add term to compute signaling entropy loss
            #entropy = torch.FloatTensor([dist_entropy_act])
            entropy = torch.FloatTensor([self.policy_act_old.get_dist_entropy(state).detach() for state in old_states_a])
            hloss =  (torch.full(entropy.size(), self.htarget) - entropy)* (torch.full(entropy.size(), self.htarget) - entropy)

            loss = loss + self.hloss_lambda*hloss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.scheduler.step()
            #print(self.scheduler.get_lr())

        # Copy new weights into old policy
        self.policy_comm_old.load_state_dict(self.policy_comm.state_dict())
        self.policy_act_old.load_state_dict(self.policy_act.state_dict())

        # clear buffer
        self.buffer.clear()