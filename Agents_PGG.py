import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Identify which device should be used by torch for the ANN calculations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentNet(nn.Module):
    def __init__(self, n_players, n_coins):
        super(AgentNet, self).__init__()
        self.n_players = n_players
        self.n_coins = n_coins
        self.n_actions = 2 # 0=keep coins, 1=give coins
        self.h_size = int(n_coins)
        self.input_size = self.n_coins + 1 # input is one-hot encoding of amount of coins + value of the multiplicative factor
        self.fc1 = nn.Linear(self.input_size, self.h_size) 
        self.fc2 = nn.Linear(self.h_size, self.n_actions)
        self.epsilon = 0.1
        self._gamma = 0.99

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out
