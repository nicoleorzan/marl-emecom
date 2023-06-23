
import torch
from src.algos.buffer import RolloutBufferComm

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

        self.buffer = RolloutBufferComm()

        for key, val in params.items(): setattr(self, key, val)

        self.reputation = 1.0
        self.idx = idx
        self.is_communicating = self.communicating_agents[self.idx]
        self.is_listening = self.listening_agents[self.idx]

        self.is_dummy = True

        if (self.is_communicating != 0):
            self.mex = torch.zeros(self.mex_size).long()
        
        print("\nNormative agent", self.idx)

        self.return_episode_norm = 0
        self.return_episode_old_norm = torch.Tensor([0.])
        self.return_episode = 0

        self.buffer = RolloutBufferComm()


    def reset(self):
        self.buffer.clear()

    def reset_batch(self):
        self.buffer.clear_batch()

    def reset_episode(self):
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode_old = self.return_episode
        self.return_episode_norm = 0
        self.return_episode = 0

    def digest_input(self, input):
        obs_m_fact, opponent_reputation = input
        self.obs_m_fact = obs_m_fact[0]
        self.opponent_reputation = opponent_reputation
    
    def digest_input_with_idx(self, input):
        obs_m_fact, opponent_idx, opponent_reputation = input
        self.obs_m_fact = obs_m_fact[0]
        self.opponent_reputation = opponent_reputation

    def set_mult_fact_obs(self, obs_m_fact, _eval=False):
        pass
    
    def set_gmm_state(self, obs, _eval=False):
        pass

    def select_message(self, m_val=None, _eval=False):
        self.buffer.messages.append(self.mex)
        if (m_val in self.buffer.messages_given_m):
                self.buffer.messages_given_m[m_val].append(self.mex)
        else: 
            self.buffer.messages_given_m[m_val] = [self.mex]
        return self.mex

    def get_message(self, message_in):
        self.message_in = message_in

    def select_action(self, m_val=None, _eval=False):
        action = torch.Tensor([0.])
        if (self.obs_m_fact > 2): # coopreative env
            action = torch.Tensor([1.])
            return action[0]
        elif (self.obs_m_fact > 1 and self.obs_m_fact < 2): # if we are playing in a mixed-motive environment
            if (self.opponent_reputation >= 0.8): # and the reputation of my opponent is big enough
                action = torch.Tensor([1.]) # I will play cooperatively

        self.buffer.actions.append(action[0])
        if (m_val in self.buffer.actions_given_m):
                self.buffer.actions_given_m[m_val].append(action[0])
        else: 
            self.buffer.actions_given_m[m_val] = [action[0]]
        return action[0]
        
    def update(self):
        pass

    def get_action_distribution(self):
        return self.reputation