
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
        self.coop = []
        self.reset()
        self.saved_losses = []

    def reset(self):
        self.logprobs = []
        self.rewards = []

    def reset_episode(self):
        self.return_episode = 0
        self.tmp_actions = []

    def select_action(self, state):

        state = torch.FloatTensor(state).to(device)
        action, action_logprob = self.policy.act(state)

        self.logprobs.append(action_logprob)

        return action

    def update(self):

        #rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        #print("\nUpdate")
        rewards =  self.rewards
        rew_norm = [(i - min(rewards))/(max(rewards) - min(rewards)) for i in rewards]
        #print("logps=", self.logprobs)
        #print("rews=", rewards)

        for i in range(len(self.logprobs)):
            self.logprobs[i] = -self.logprobs[i] * rew_norm[i] #rewards[i]

        #print("\nlogp", self.logprobs)
        #print("loss=", np.mean([i.detach() for i in self.logprobs]))
        self.saved_losses.append(np.mean([i.detach() for i in self.logprobs]))
        self.optimizer.zero_grad()
        tmp = [torch.ones(a.data.shape) for a in self.logprobs]
        autograd.backward(self.logprobs, tmp, retain_graph=True)
        self.optimizer.step()

        self.reset()