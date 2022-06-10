
import torch.nn as nn
import torch
from torch.distributions import Categorical, Normal

class ActorCriticDiscrete(nn.Module):

    def __init__(self, input_dim, action_dim):
        super(ActorCriticDiscrete, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = 64
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim)#,
        )
        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def act(self, state, greedy=False):

        out = self.actor(state)
        dist = Categorical(logits=out)

        if (greedy):
            act = torch.argmax(out)
        else:
            act = dist.sample()

        logprob = dist.log_prob(act)

        return act.detach(), logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)  # here I changed probs with logits!!!
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprob, dist_entropy, state_values



class ActorCriticContinuous(nn.Module):

    def __init__(self, input_dim, action_dim):
        super(ActorCriticContinuous, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = 64
        self.action_dim = action_dim
        self.min_var = 0.0001

        self.policy = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU())

        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.mean = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid())
        self.var = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
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