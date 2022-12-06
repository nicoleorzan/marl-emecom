
import torch
import torch.autograd as autograd
import numpy as np

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class Reinforce():

    def __init__(self, model, optimizer, params):

        for key, val in params.items():  setattr(self, key, val)
    
        self.policy = model.to(device)
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decayRate)
        
        self.train_returns = []
        self.return_episode = 0
        self.return_episode_norm= 0
        self.tmp_actions = []
        self.coop = []
        self.saved_losses = []

        self.reset()

        self.eps_norm = 0.0001

    def reset(self):
        self.logprobs = []
        self.rewards = []

    def reset_episode(self):
        self.return_episode_old = self.return_episode
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode = 0
        self.return_episode_norm = 0
        self.tmp_actions_old = self.tmp_actions
        self.tmp_actions = []

    def select_action(self, state, eval=False):

        if (eval == True):
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy.act(state)

        elif (eval == False):

            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy.act(state)

            self.logprobs.append(action_logprob)

        return action

    def get_distribution(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            out = self.policy.get_distribution(state)

            return out


    def update(self):

        rewards =  self.rewards
        rew_norm = [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]

        for i in range(len(self.logprobs)):
            self.logprobs[i] = -self.logprobs[i] * rew_norm[i]

        self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in self.logprobs])))
        self.optimizer.zero_grad()
        tmp = [torch.ones(a.data.shape) for a in self.logprobs]
        autograd.backward(self.logprobs, tmp, retain_graph=True)
        self.optimizer.step()

        self.scheduler.step()
        print(self.scheduler.get_lr())

        self.reset()