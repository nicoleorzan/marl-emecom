
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

    def __init__(self, idx):

        self.buffer = RolloutBufferComm()

        self.reputation = 1.0
        self.idx = idx
        self.is_communicating = 0
        self.is_listening = 0
        self.n_playing_agents = 2

        self.is_dummy = True
        
        print("\nNormative agent", self.idx)

        self.return_episode_norm = 0
        self.return_episode_old_norm = torch.Tensor([0.])
        self.return_episode = 0

    def reset_batch(self):
        pass

    def reset_episode(self):
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode_old = self.return_episode
        self.return_episode_norm = 0
        self.return_episode = 0

    def digest_input(self, input):
        obs_m_fact, opponent_reputation = input
        self.obs_m_fact = obs_m_fact[0]
        self.opponent_reputation = opponent_reputation

    def set_mult_fact_obs(self, obs_m_fact, _eval=False):
        pass
    
    def set_gmm_state(self, obs, _eval=False):
        pass

    def select_message(self, m_val=None, _eval=False):
        pass

    def get_message(self, message_in):
        self.message_in = message_in

    def select_action(self, m_val=None, _eval=False):
        action = torch.Tensor([0.])
        if (self.obs_m_fact > 1):
            if (self.opponent_reputation >= 0.5):
                action = torch.Tensor([1.])
        return action
        
    def update(self):
        pass

    def get_action_distribution(self):
        return self.reputation