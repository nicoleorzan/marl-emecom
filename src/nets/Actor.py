
import torch.nn as nn
import torch
from torch.distributions import Categorical

class Actor(nn.Module):

    def __init__(self, params, input_size, output_size, n_hidden, hidden_size, gmm=False):
        super(Actor, self).__init__()

        # absorb all parameters to self
        for key, val in params.items():  setattr(self, key, val)

        #self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=0)

        #print("embedding dim=", self.embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=self.embedding_dim)

        # inputs: 
        # multiplication factor (scalar) already considered
        # possible message (already considered)
        # opponent index (embedded) new -> NO!!
        # opponent reputation (scalar) new
        self.input_size = input_size + 1

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

    def embed_opponent_index(self, idx):
        #print("embed_opponent_idx, inside actor")
        # turn index into one-hot encoding and then embed it
        idx_one_hot = torch.nn.functional.one_hot(torch.Tensor([idx]).long(), num_classes=self.n_agents)[0]
        #print("idx_one_hot=", idx_one_hot)
        self.emb = self.embedding(idx_one_hot)
        return self.emb

    def act(self, state, greedy=False, get_distrib=False):
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

        if (get_distrib == True):
            return act.detach(), logprob, dist.entropy().detach(), out
        
        return act.detach(), logprob, dist.entropy().detach()
    
    def get_dist_entropy(self, state):
        out = self.actor(state)
        out = self.softmax(out)
        dist = Categorical(out)
        return dist.entropy()

    def get_distribution(self, state, greedy=False):
        out = self.actor(state)
        out = self.softmax(out)
        return out
    
    def get_values(self, state, greedy=False):
        out = self.actor(state)
        return out