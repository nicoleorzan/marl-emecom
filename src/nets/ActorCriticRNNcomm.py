
import torch.nn as nn
import torch
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn.functional as F

#https://github.com/pliang279/Competitive-Emergent-Communication/blob/master/chatbots.py

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

class ActorCriticRNNcomm(nn.Module):

    def __init__(self, params):
        super(ActorCriticRNNcomm, self).__init__()

        # absorb all parameters to self
        for key, val in params.items():  setattr(self, key, val)

        self.hState = torch.Tensor()
        self.cState = torch.Tensor()
        self.softmax = nn.Softmax(dim=0)

        # net that embeds the actions of all other agents
        self.input_size = self.obs_size + self.n_agents*self.mex_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.mex_linear = nn.Linear(self.hidden_size, self.mex_size)
        self.mex_critic_linear = nn.Linear(self.hidden_size, 1)

        self.act_linear = nn.Linear(self.hidden_size, self.action_size)
        self.act_critic_linear = nn.Linear(self.hidden_size, 1)

        self.mex_linear.weight.data = norm_col_init(
            self.mex_linear.weight.data, 0.01)
        self.mex_linear.bias.data.fill_(0)
        self.mex_critic_linear.weight.data = norm_col_init(
            self.mex_critic_linear.weight.data, 1.0)
        self.mex_critic_linear.bias.data.fill_(0)

        self.act_linear.weight.data = norm_col_init(
            self.act_linear.weight.data, 0.01)
        self.act_linear.bias.data.fill_(0)
        self.act_critic_linear.weight.data = norm_col_init(
            self.act_critic_linear.weight.data, 1.0)
        self.act_critic_linear.bias.data.fill_(0)

        self.train()
        self.evalFlag = False
        self.reset_state()

    def reset_state(self):
        self.hState = Variable(torch.zeros(self.hidden_size))
        self.cState = Variable(torch.zeros(self.hidden_size))  

    # observe input = concat(coins, messages of all other agents)
    def observe(self, input):
        # embed and pass through LSTM
        out = F.relu(self.fc1(input))
        # now pass it through rnn
        self.hState, self.cState = self.lstm(out, (self.hState, self.cState))

    # speak a token
    def speak(self):
        # compute softmax and choose a new action

        out = self.softmax(self.mex_linear(self.hState))
        # if evaluating
        if self.evalFlag:
            _, message = out.max(1)
        else:
            dist = Categorical(out)
            message = dist.sample()
            mex_logprob = dist.log_prob(message)

        return message, mex_logprob

    # given the sequence of messages, take an action
    def act(self):

        out = self.softmax(self.act_linear(self.hState))
        # if evaluating
        if self.evalFlag:
            _, action = out.max(1)
        else:
            dist = Categorical(out)
            action = dist.sample()
            act_logprob = dist.log_prob(action)

        return action, act_logprob

    def evaluate_mex(self, state, mex):
        #print("EVALUATE MESSAGE")
        # I expect a state composed of the hidden states too

        state, (hstate, cstate) = state
        #print("state=", state.shape)
        #print("hstate=", hstate.shape)
        self.hState = hstate
        self.cState = cstate
        self.observe(state)
        out = self.softmax(self.mex_linear(self.hState))
        dist = Categorical(out)
        
        mex_logprob = dist.log_prob(mex)

        mex_dist_entropy = dist.entropy().mean()
        mex_state_values = self.mex_critic_linear(self.hState)

        return mex_logprob, mex_dist_entropy, mex_state_values

    def evaluate_act(self, state, action):
        # I expect a state composed of the hidden states too

        state, (hstate, cstate) = state
        self.hState = hstate
        self.cState = cstate
        self.observe(state)
        out = self.softmax(self.act_linear(self.hState))
        dist = Categorical(out)
        
        act_logprob = dist.log_prob(action)

        act_dist_entropy = dist.entropy().mean()
        act_state_values = self.act_critic_linear(self.hState)

        return act_logprob, act_dist_entropy, act_state_values

    def get_actions_entropy(self, state):
        state, (hstate, cstate) = state
        self.hState = hstate
        self.cState = cstate
        self.observe(state)
        out = self.softmax(self.act_linear(self.hState))
        dist = Categorical(out)

        return dist.entropy().mean()

    # switch mode to evaluate
    def eval(self): self.evalFlag = True

    # switch mode to train
    def train(self): self.evalFlag = False