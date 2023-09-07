
import torch
import random

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

        #self.buffer = RolloutBuffer()

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
        pass
        #self.buffer.clear()

    def digest_input_anast(self, input):
        #opponent_reputation, opponent_previous_action = input
        self.opponent_reputation = input #opponent_reputation
        #self.opponent_previous_action = opponent_previous_action
    
    def select_opponent(self, reputations):
        return random.randint(0, self.n_agents-1)

    def select_action(self, _eval=False):
        
        #if (hasattr(self, 'state_act')):
        self.opponent_reputation = self.state_act[0]
        
        action = torch.Tensor([0.])
        if (self.opponent_reputation == torch.Tensor([1.])): # if the reputation of my opponent is 1
            action = torch.Tensor([1.]) # I will play cooperatively
        
        return action
        
    def update(self, epoch=None):
        pass

    def update1(self):
        pass

    def get_action_distribution(self):
        return self.reputation