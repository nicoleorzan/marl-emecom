
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

    def select_action(self, state_act, _eval=False):
        if (self.reputation_enabled == 0):
            print("CANNOT USE DUMMY AGENTS, THERE IS NO REPUTATION")
            return
        
        if (self.reputation_enabled == 1):
            if (len(self.mult_fact) > 1):
                self.opponent_reputation = state_act[0]
                self.mf = state_act[1]
            else: 
                self.opponent_reputation = state_act[0]
        else:
            print("CANNOT USE DUMMY AGENTS, THERE IS NO REPUTATION")
            return

        action = torch.Tensor([0.])
        if (len(self.mult_fact) == 1):
            if (self.opponent_reputation == torch.Tensor([1.])): 
                action = torch.Tensor([1.]) # I will play cooperatively
        else: 
            if (self.mf >= 1. and self.opponent_reputation == torch.Tensor([1.])): # and the reputation of my opponent is big enough
                action = torch.Tensor([1.]) # I will play cooperatively

        return action
    
    def select_reputation_assignment(self, rep_state):
        other_rep = torch.Tensor([rep_state[0]])
        other_action = torch.Tensor([rep_state[1]])
        reputation = other_rep
        #print("other rep=", other_rep, "other_action", other_action)

        if (self.mf >= 1.):
            if (other_action == torch.Tensor([1.0])):
                if (other_rep == torch.Tensor([1.0])):
                    reputation = torch.Tensor([1.0])
                else: 
                    reputation = torch.Tensor([0.0])
            else: 
                if (other_rep == torch.Tensor([1.0])):
                    reputation = torch.Tensor([0.0])
                else: 
                    reputation = torch.Tensor([1.0])
        #print("rep=", reputation)
        return reputation

        
    def update(self, _iter=None):
        pass

    def update1(self):
        pass

    def get_action_distribution(self):
        return self.reputation