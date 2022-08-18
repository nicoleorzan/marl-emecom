
import torch.nn as nn
import torch
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn.functional as F
#import torch.autograd as autograd

#https://github.com/pliang279/Competitive-Emergent-Communication/blob/master/chatbots.py

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

class ActorCriticRNN(nn.Module):

    def __init__(self, params):
        super(ActorCriticRNN, self).__init__()

        # absorb all parameters to self
        for key, val in params.items():  setattr(self, key, val)

        self.hState = torch.Tensor()
        self.cState = torch.Tensor()
        self.softmax = nn.Softmax(dim=0)

        # net that embeds the actions f all other agents
        self.input_size = self.obs_size + self.n_agents*self.action_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.actor_linear = nn.Linear(self.hidden_size, self.action_size)
        self.critic_linear = nn.Linear(self.hidden_size, 1)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.train()
        self.evalFlag = False
        self.useGPU = False
        self.reset_state()

    def reset_state(self):
        self.hState = Variable(torch.zeros(self.hidden_size))
        self.cState = Variable(torch.zeros(self.hidden_size))

    # observe actions of all other agents
    def observe(self, input):
        # embed and pass through LSTM
        out = F.relu(self.fc1(input))
        # now pass it through rnn
        self.hState, self.cState = self.lstm(out, (self.hState, self.cState))

    def act(self):
        # compute softmax and choose a new action

        out = self.softmax(self.actor_linear(self.hState))
        if self.evalFlag:
            action = out.argmax()
            return action
        else:
            dist = Categorical(out)
            action = dist.sample()
            act_logprob = dist.log_prob(action)

        return action, act_logprob
    
    def evaluate(self, state, action):
        # I expect a state composed as (state, (hstate, cstate))

        state, (hstate, cstate) = state
        self.hState = hstate
        self.cState = cstate
        self.observe(state)
        out = self.softmax(self.actor_linear(self.hState))
        dist = Categorical(out)
        
        action_logprob = dist.log_prob(action)

        dist_entropy = dist.entropy().mean()
        state_values = self.critic_linear(self.hState)

        return action_logprob, dist_entropy, state_values

    def get_distribution(self, state):
        self.observe(state)
        out = self.softmax(self.actor_linear(self.hState))

        return out

    """ def performBackward(self):
        if self.useGPU:
            tmp = [torch.ones(a.data.shape).cuda() for a in self.actionLosses]
        else:
            tmp = [torch.ones(a.data.shape) for a in self.actionLosses]
            autograd.backward(self.actionLosses, tmp, retain_graph=True)"""

    # switch mode to evaluate
    def eval(self): self.evalFlag = True

    # switch mode to train
    def train(self): self.evalFlag = False