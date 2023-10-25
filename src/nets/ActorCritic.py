
import torch.nn as nn
import torch
from torch.distributions import Categorical, Normal

# check hidden layer
class ActorCritic(nn.Module):

    def __init__(self, params, input_size, output_size, n_hidden, hidden_size, gmm=False):
        super(ActorCritic, self).__init__()

        for key, val in params.items():  setattr(self, key, val)

        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.gmm_ = gmm

        if (self.n_hidden == 2):
            self.actor = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=0)
            )
            self.critic = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 1),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=0)
            )
            self.critic = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 1),
            )

    def reset_state(self):
        pass

    def act(self, state, ent=False, greedy=False):

        out = self.actor(state)
        dist = Categorical(out)

        if (self.random_baseline == True): 
            act = torch.randint(0, self.action_size, (1,))[0]
        else:
            if (greedy):
                act = torch.argmax(out)
            else:
                act = dist.sample()

        logprob = dist.log_prob(act) # negativi

        if (ent):
            return act.detach(), logprob, dist.entropy().detach()

        return act.detach(), logprob #.detach() --> moved from here be necessary for REINFORCE,put it in PPO code

    def get_dist_entropy(self, state):
        out = self.actor(state)
        dist = Categorical(out)
        return dist.entropy()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprob, dist_entropy, state_values

    def get_distribution(self, state):
        out = self.actor(state)
        return out