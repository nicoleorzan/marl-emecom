
import torch
from src.algos.ReinforceGeneral import ReinforceGeneral, Reinforce
import torch.nn.functional as F

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    

class Agent():

    def __init__(self, params, idx=0):

        self.idx = idx
        
        for key, val in params.items(): setattr(self, key, val)

        self.define_agent_characteristics()
        print("\nAgent", self.idx)
        print("is communicating?:", self.is_communicating)
        print("is listening?:", self.is_listening)
        print("uncertainty:", self.uncertainties[self.idx])
        print("gmm=", self.gmm_)

        self.reinforce = ReinforceGeneral(params, self.is_communicating, self.is_listening, self.gmm_, self.n_communicating_agents)

        input_part_sel = self.obs_size
        if (self.gmm_):
            input_part_sel = self.n_gmm_components

        if (self.partner_selection):
            print("Defining partner selection module...")
            mask = torch.ones(self.n_agents)
            mask[idx] = 0
            print("mask=", mask)
            print("input=", self.n_agents + input_part_sel)
            self.partner_selection_reinforce = Reinforce(params=params, input_size=self.n_agents + input_part_sel, # I select my partner based on the agents id's and the observation of the environment I made 
                                               output_size=self.n_agents, hidden_size=10, mask=mask)
        
        if (self.punishment):
            print("Defining punishment module...")
            self.punishment_reinforce = Reinforce(params, input_size=self.n_agents+1, # one hot encoding of the agent index + agent action 
                                        output_size=1, hidden_size=10)

        self.reputation = 1. # between 0 and 1 or discrete?
        self.partner_id = None

        self.return_episode_norm = 0
        self.return_episode = 0

        self.reset()

    def define_agent_characteristics(self):
        self.gmm_ = self.gmm_[self.idx]
        self.is_communicating = self.communicating_agents[self.idx]
        self.is_listening = self.listening_agents[self.idx]
        self.n_communicating_agents = self.communicating_agents.count(1)

    def reset(self):
        #self.rewards = []
        self.rew_norm = []
        self.reinforce.reset()
        if (self.partner_selection):
            self.partner_selection_reinforce.reset()
        if (self.punishment):
            self.punishment_reinforce.reset()
    
    def reset_batch(self):
        self.reinforce.reset_batch()

    def reset_episode(self):
        self.chosen_agent = 0
        self.reinforce.reset_episode()

        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode_old = self.return_episode
        self.return_episode = 0
        self.return_episode_norm = 0

    def set_reward(self, rew, normalizer):
        #self.rewards.append(rew)
        self.rew_norm.append(rew/normalizer)
        self.return_episode_norm += rew/normalizer
        self.return_episode =+ rew

        self.reinforce.rewards = self.rew_norm
        if (self.partner_selection and self.chosen_agent):
            self.partner_selection_reinforce.rewards = self.rew_norm
        if (self.punishment):
            self.punishment_reinforce.rewards = self.rew_norm

    def set_state(self, state, _eval=False):
        self.reinforce.set_state(state, _eval)

    def select_partner(self, reputations):
        self.chosen_agent = 1
        _input = torch.cat((reputations, self.reinforce.state_to_comm)).to(device)
        act = self.partner_selection_reinforce.select_action(_input)
        partner_id = int(act)
        #print("partner_id=", partner_id)
        #self.partner_id = torch.Tensor([partner_id])
        #print("torch.Tensor([partner_id]).to(torch.int64)=", torch.Tensor([partner_id]).to(torch.int64))
        self.partner_id = F.one_hot(torch.Tensor([partner_id]).to(torch.int64), num_classes=self.n_agents)[0]
        #print("partner id one hot=", self.partner_id)
       
        return partner_id

    def select_action(self, m_val=None, _eval=False):
        #print("self.partner_id=", self.partner_id)
        #print("self.reinforce.state_to_comm=", self.reinforce.state_to_comm)
        act = self.reinforce.select_action(partner_id = self.partner_id, m_val = m_val, _eval =_eval)
        return act

    def update(self):
        print("==>General update")
        self.reinforce.update()
        if (self.partner_selection):
            print("==>Partner selection update")
            self.partner_selection_reinforce.update()
        if (self.punishment):
            print("==>Punishment update")
            self.punishment_reinforce.update()

        self.reset()

    def get_action_distribution(self):
        return self.reinforce.get_action_distribution()