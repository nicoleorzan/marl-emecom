
import torch.nn as nn
from torch.distributions import Categorical

# check hidden layer
class Actor(nn.Module):

    def __init__(self, params, input_size, output_size, n_hidden, hidden_size, gmm=False):
        super(Actor, self).__init__()

        for key, val in params.items():  setattr(self, key, val)

        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.gmm_ = gmm
        self.softmax = nn.Softmax(dim=0)

        if (self.n_hidden == 2):
            self.actor = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_size)
            )

        else:
            self.actor = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_size)
            )

    def get_values(self, state):
        out = self.actor(state)
        return out
    
    def get_dist_entropy(self, state):
        print("state=", state)
        out = self.actor(state)
        print("out=", out)
        dist = Categorical(logits=out)
        return dist.entropy()