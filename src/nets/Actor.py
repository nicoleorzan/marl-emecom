
import torch.nn as nn
import torch
from torch.distributions import Categorical

class Actor(nn.Module):

    def __init__(self, params, input_size, output_size, n_hidden, hidden_size, gmm=False):
        super(Actor, self).__init__()

        # absorb all parameters to self
        for key, val in params.items():  setattr(self, key, val)

        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=0)

        if (self.n_hidden == 2):
            self.actor = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_size)#,
                #nn.Softmax(dim=0)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_size)#,
                #nn.Softmax(dim=0)
            )

    def act(self, state, greedy=False):
        out = self.actor(state)
        out = self.softmax(out)
        dist = Categorical(out)

        if (self.random_baseline == True): 
            act = torch.randint(0, self.action_size, (1,))[0]
        else:
            if (greedy):
                act = torch.argmax(out)
            else:
                act = dist.sample()

        logprob = dist.log_prob(act) # negativi

        return act.detach(), logprob, dist.entropy().detach()
    
    def get_dist_entropy(self, state):
        out = self.actor(state)
        dist = Categorical(out)
        return dist.entropy()

    def get_distribution(self, state, greedy=False):
        out = self.actor(state)
        out = self.softmax(out)
        return out
    
    def get_values(self, state, greedy=False):
        out = self.actor(state)
        return out