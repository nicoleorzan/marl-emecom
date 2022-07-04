
from dis import disco
import torch
import copy
import torch.nn as nn
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
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def __print__(self):
        print("states=", len(self.states))
        print("actions=", len(self.actions))
        print("logprobs=", len(self.logprobs))
        print("rewards=", len(self.rewards))
        print("is_terminals=", len(self.is_terminals))

class PPO():

    def __init__(self, model, optimizer, lr_actor, lr_critic, gamma, K_epochs, eps_clip, c1, c2):

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.c1 = c1 
        self.c2 = c2

        self.buffer = RolloutBuffer()
    
        self.policy = model.to(device)
        self.optimizer = optimizer

        self.policy_old = copy.deepcopy(model).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.train_returns = []
        self.tmp_return = 0
        self.train_actions = []
        self.tmp_actions = []
        self.coop = []

    def select_action(self, state):
    
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
        return action.item()

    def eval_action(self, state, done=False):
    
        actions_logprobs = []
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            for action in range(self.action_dim):
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

        #print("self.policy.input_dim=", self.policy.input_dim)
        if (self.policy.input_dim == 1):
            old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
            old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
            old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
        else:
            old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        #print("actions=", old_actions.shape)
        #print("states=", old_states.shape)

        for _ in range(self.K_epochs):
            #print("old states=", old_states)
            #print("old_states=", old_states.shape)
            #print("old_actions=", old_actions.shape)

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