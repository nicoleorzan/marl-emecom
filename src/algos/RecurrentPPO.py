
from dis import disco
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBuffer:
    def __init__(self, recurrent = False):
        self.actions = []
        self.states = []
        self.hstates = []
        self.cstates = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.recurrent = recurrent
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.hstates[:]
        del self.cstates[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def __print__(self):
        print("states=", len(self.states))
        if self.recurrent:
            print("hstates=", len(self.hstates))
            print("cstates=", len(self.cstates))
        print("actions=", len(self.actions))
        print("logprobs=", len(self.logprobs))
        print("rewards=", len(self.rewards))
        print("is_terminals=", len(self.is_terminals))

class RecurrentPPO():

    def __init__(self, model, optimizer, params):

        # absorb all parameters to self
        for key, val in params.items():  setattr(self, key, val)

        self.optimizer = optimizer

        self.buffer = RolloutBuffer()
    
        self.policy = model.to(device)
        self.policy_old = copy.deepcopy(model).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.train_returns = []
        self.tmp_return = 0
        self.train_actions = []
        self.tmp_actions = []
        self.coop = []

    def reset(self):
        self.policy.init_hidden()
        self.tmp_return = 0
        self.tmp_actions = []

    def select_action(self, state):

        #print("state=", state)
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            self.policy_old.observe(state)
            action, action_logprob = self.policy_old.act()
            #print("action=", action)

            if self.recurrent:
                hstate = self.policy_old.hState
                cstate = self.policy_old.cState
                self.buffer.hstates.append(hstate)
                self.buffer.cstates.append(cstate)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

        return action

    def eval_action(self, state, done=False):
    
        actions_logprobs = []
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            for action in range(self.action_size):
                a = torch.Tensor([action])
                action_logprob, _, _ = self.policy_old.evaluate(state, a)
                actions_logprobs.append(action_logprob.item())

        return actions_logprobs

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        #print("\nrewards=", rewards)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        #print("self.policy.input_size=", self.policy.input_size)
        if (self.policy.input_size == 1):
            old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
            old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
            old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
            if self.recurrent:
                old_hstates = torch.stack(self.buffer.hstates, dim=0).detach().to(device)
                old_cstates = torch.stack(self.buffer.cstates, dim=0).detach().to(device)
                old_states = (old_states, (old_hstates, old_cstates))
        else:
            old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
            if self.recurrent:
                old_hstates = torch.squeeze(torch.stack(self.buffer.hstates, dim=0)).detach().to(device)
                old_cstates = torch.squeeze(torch.stack(self.buffer.cstates, dim=0)).detach().to(device)
                old_states = (old_states, (old_hstates, old_cstates))

        #print("actions=", old_actions.shape)
        #print("states=", old_states.shape)

        for _ in range(self.K_epochs):
            #print("old states=", old_states)
            #print("old_states=", old_states)
            #print("old_actions=", old_actions)

            logprobs, dist_entropy, state_values = self.policy.evaluate(old_states, old_actions)
            #print("logprobs, dist_entropy, state_values", logprobs, dist_entropy, state_values)

            state_values = torch.squeeze(state_values)
            #print("states values act", state_values.shape)
            #print("rewards=", rewards.shape)
            #print("state_values=", state_values)
            #print("rewards=", rewards)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            #print("ratios=", ratios)

            advantages = rewards - state_values.detach()
            surr1 = ratios*advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages

            #print("surr1, surr2",surr1, surr2)
            loss = (-torch.min(surr1, surr2) + self.c1*self.MseLoss(state_values, rewards) + self.c2*dist_entropy)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()