
from src.algos.buffer import RolloutBuffer
import torch

class Agent():

    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)

        self.reputation = torch.Tensor([1.0])
        self.old_reputation = self.reputation

        self.is_dummy = False
        self.idx = idx
        print("\nAgent", self.idx)

        self.buffer = RolloutBuffer()

        # Action Policy
        self.input_act = self.obs_size
    
        self.max_memory_capacity = 5000

        self.reset()

    def reset(self):
        self.buffer.clear()
        self.previous_action = torch.Tensor([1.])
        self.return_episode = 0

    def digest_input_anast(self, input):
        opponent_reputation, opponent_previous_action = input

        opponent_reputation = torch.Tensor([opponent_reputation])
        my_reputation = torch.Tensor([self.reputation])
        my_previous_action = torch.Tensor([self.previous_action])
        self.state_act = torch.cat((opponent_reputation, my_reputation, opponent_previous_action, my_previous_action), 0)

    def select_action(self, _eval=False):
        
        state_to_act = self.state_act
    
        if (_eval == True):
            with torch.no_grad():
                action, action_logprob, entropy = self.policy_act.act(state=state_to_act)

        elif (_eval == False):
            action, action_logprob, entropy, distrib = self.policy_act.act(state=state_to_act, greedy=False, get_distrib=True)
            
            self.buffer.states.append(state_to_act)
            self.buffer.actions.append(action)
        
            self.buffer.logprobs.append(action_logprob)
        #print(" action, distrib=", action, distrib)
        return action, distrib
    
    def get_action_distribution(self):

        with torch.no_grad():
            out = self.policy_act.get_distribution(self.state_act)
            return out
        
    def select_opponent(self, reputations, _eval=False):

        if (_eval == True):
            with torch.no_grad():
                opponent_out, opponent_logprob, entropy = self.policy_opponent_selection.act(reputations)

        elif(_eval == False):
            opponent_out, opponent_logprob, entropy = self.policy_opponent_selection.act(reputations)
            if (opponent_out != self.idx):
                self.buffer.reputations.append(reputations)
                self.buffer.opponent_choices.append(opponent_out)
                self.buffer.opponent_logprobs.append(opponent_logprob)

        return opponent_out