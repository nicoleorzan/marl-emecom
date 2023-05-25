
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

class DummyAgent():

    def __init__(self, prob_coop, idx):

        self.buffer = RolloutBufferComm()

        self.prob_coop = prob_coop
        self.reputation = prob_coop
        self.idx = idx
        self.is_communicating = 0
        self.is_listening = 0

        self.is_dummy = True
        
        print("\nDummy agent", self.idx)
        print("with prob coop=", self.prob_coop)

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
        pass

    def set_mult_fact_obs(self, obs_m_fact, _eval=False):
        pass
    
    def set_gmm_state(self, obs, _eval=False):
        pass

    def select_message(self, m_val=None, _eval=False):
        pass

    def get_message(self, message_in):
        self.message_in = message_in

    def select_action(self, m_val=None, _eval=False):
        action = torch.bernoulli(torch.Tensor([self.prob_coop]))
        return action[0]

    def update(self):
        pass

    def get_action_distribution(self):
        return self.reputation