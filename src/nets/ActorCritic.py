
import torch.nn as nn
import torch
from torch.distributions import Categorical, Normal

# check hidden layer
class ActorCritic(nn.Module):

    def __init__(self, params, comm=False):
        super(ActorCritic, self).__init__()

        # absorb all parameters to self
        for key, val in params.items():  setattr(self, key, val)

        self.comm = comm

        if (self.comm): 
            # in the case of communication, the input is not only the coins 
            # but also the messages sent by the other agents
            self.input_size = self.obs_size + self.n_agents*self.mex_size
        else: 
            self.input_size = self.obs_size

        self.actor = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.action_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )

    def reset_state(self):
        pass

    def act(self, state, greedy=False):

        out = self.actor(state)
        dist = Categorical(logits=out)

        if (greedy):
            act = torch.argmax(out)
        else:
            act = dist.sample()

        logprob = dist.log_prob(act)

        return act.detach(), logprob.detach()

    def get_dist_entropy(self, state):
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)  # here I changed probs with logits!!!
        dist_entropy = dist.entropy()
        return dist_entropy
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)  # here I changed probs with logits!!!
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprob, dist_entropy, state_values

    def get_distribution(self, state, greedy=False):
        out = self.actor(state)
        m = nn.Softmax(dim=0)
        out = m(out)

        return out




class ActorCriticContinuous(nn.Module):

    def __init__(self, params):
        super(ActorCriticContinuous, self).__init__()

        # absorb all parameters to self
        for key, val in params.items():  setattr(self, key, val)

        self.min_var = 0.001

        self.policy = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU())

        self.critic = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )

        self.mean = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid())
        self.var = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.ReLU())

    def act(self, state):

        logits = self.policy(state)
        mean = self.mean(logits)
        var = self.var(logits) + self.min_var

        dist = Normal(mean, var)

        act = dist.rsample()

        if (act < 0.):
            act = torch.Tensor([0.])
        elif (act > 1.):
            act = torch.Tensor([1.])

        logprobs = dist.log_prob(act)

        return act.detach(), logprobs.detach()

    def evaluate(self, state, action):

        logits = self.policy(state)
        mean = self.mean(logits)
        var = self.var(logits) + self.min_var

        dist = Normal(mean, var)

        action_logprob = dist.log_prob(action)

        dist_entropy = dist.entropy()

        state_values = self.critic(state)

        return action_logprob, dist_entropy, state_values