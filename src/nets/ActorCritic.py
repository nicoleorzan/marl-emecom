
import torch.nn as nn
import torch
from torch.distributions import Categorical, Normal

# check hidden layer
class ActorCritic(nn.Module):

    def __init__(self, params, input_size, output_size):
        super(ActorCritic, self).__init__()

        # absorb all parameters to self
        for key, val in params.items():  setattr(self, key, val)

        self.input_size = input_size
        self.output_size = output_size
        self.bottleneck_size = 8

        """if (self.comm == True): 
            self.actor = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(), # try out other activation function?
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=0)
            )
            self.critic = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(), # try out other activation function?
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 1),
        )"""
        
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
        #dist = Categorical(logits=out)
        dist = Categorical(out)

        if (self.random_baseline == True): 
            act = torch.randint(0, self.action_size, (1,))[0]
        else:
            if (greedy):
                act = torch.argmax(out)
            else:
                act = dist.sample()

        logprob = dist.log_prob(act) # negativi
        #print("logp=", logprob)

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

    def get_distribution(self, state, greedy=False):
        out = self.actor(state)
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