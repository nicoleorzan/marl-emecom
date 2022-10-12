
from src.algos.buffer import RolloutBuffer
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

class PPO():

    """
    PPO with ability of handing recurrent net or simple nets

    Non-recurrent networks need to provide:
    - an "act" function, that takes as input a states and returns an action
    - an "evaluate" function, that takes as input states and actions and returns the value function
    Recurrent networks need to provide:
    - a "reset" function, that initializes hidden states
    - an "observe" function, that takes as input the state
    - an "act" function, that samples an action
    - an "evaluate" function, that takes as input states (which are couples of states and hidden states) and actions and returns the value function
    
    """

    def __init__(self, model, optimizer, params):

        for key, val in params.items():  setattr(self, key, val)

        self.buffer = RolloutBuffer(self.recurrent)
    
        self.policy = model.to(device)
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decayRate)
        
        self.policy_old = copy.deepcopy(model).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.train_returns = []
        self.tmp_return = 0
        self.train_actions = []
        self.tmp_actions = []
        self.coop = []

    def reset(self):
        if self.recurrent:
            self.policy.reset_state()
        self.tmp_return = 0
        self.tmp_actions = []

    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)

            if self.recurrent:
                self.policy_old.observe(state)
                action, action_logprob = self.policy_old.act()
            else:
                action, action_logprob = self.policy_old.act(state)

            if self.recurrent:
                self.buffer.hstates.append(self.policy_old.hState)
                self.buffer.cstates.append(self.policy_old.cState)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob.detach())

        return action

    def eval_action(self, state):
    
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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
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

    
        for _ in range(self.K_epochs):

            logprobs, dist_entropy, state_values = self.policy.evaluate(old_states, old_actions)
            #print("logporbs=", logprobs)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios*advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
            #print("state values=", state_values)

            loss = (-torch.min(surr1, surr2) + self.c1*self.MseLoss(state_values, rewards) + \
                self.c2*dist_entropy)
            #print("loss=", loss)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            #print(self.scheduler.get_lr())

        self.scheduler.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()