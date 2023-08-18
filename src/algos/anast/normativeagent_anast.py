
import torch
import random
from src.algos.buffer import RolloutBuffer

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class NormativeAgent():

    def __init__(self, params, idx):

        self.buffer = RolloutBuffer()

        for key, val in params.items(): setattr(self, key, val)

        self.reputation = torch.Tensor([1.0])
        self.old_reputation = self.reputation
        self.idx = idx
        self.is_dummy = True

        self.previous_action = torch.Tensor([1.])
        
        print("\nNormative agent", self.idx)

        self.return_episode_norm = 0
        self.return_episode_old_norm = torch.Tensor([0.])
        self.return_episode = 0

    def reset(self):
        self.buffer.clear()

    def digest_input_anast(self, input):
        opponent_reputation, opponent_previous_action = input
        self.opponent_reputation = opponent_reputation
        self.opponent_previous_action = opponent_previous_action
    
    def select_opponent(self, reputations):
        return random.randint(0, self.n_agents-1)

    def select_action(self, m_val=None, _eval=False):
        if (hasattr(self, 'state_act')):
            self.opponent_reputation = self.state_act[0]
        action = torch.Tensor([0.])

        if (self.opponent_reputation >= self.other_reputation_threshold): # and the reputation of my opponent is big enough
            action = torch.Tensor([1.]) # I will play cooperatively

        self.buffer.actions.append(action[0])
        if (m_val in self.buffer.actions):
                self.buffer.actions[m_val].append(action[0])
        else: 
            self.buffer.actions = [action[0]]
        return action[0]
        
    def update(self):
        pass

    def get_action_distribution(self):
        return self.reputation