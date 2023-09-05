import functools
from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
import supersuit as ss
from pettingzoo.utils import parallel_to_aec
import torch



# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

# azione 1 e` cooperativa

def env(config):
    env = raw_env(config)
    # This wrapper is only for environments which print results to the terminal
    # env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    if (config.fraction == False):
        env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    env_ = ss.pettingzoo_env_to_vec_env_v1(env_)
    env_ = ss.concat_vec_envs_v1(env_, 1, base_class="stable_baselines3")
    return env

def raw_env(config):
    env = parallel_env(config)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {
        'render.modes': ['human'], 
        "name": "prisoner_dilemma"
        }

    def __init__(self, config):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.

        In config we need:
        - n_agents
        - n_game_iterations
        '''

        for key, val in config.items(): setattr(self, key, val)

        self.possible_agents = ["agent_" + str(r) for r in range(self.n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.infos = {agent: {} for agent in self.agents}

        self.n_actions = 2 # give money (1), keep money (0)
        # self.obs_space_size = 2 # I can observe the amount of money I have, and the multiplicative factor (with uncertaity)

        # c is fixed, b can change
        self.c = torch.Tensor([self.c_value])
        self.d = torch.Tensor([self.d_value])
        self.b = torch.Tensor([self.b_value])
        self.mat = torch.Tensor([[self.c+self.d, self.b+self.c],[self.d, self.b]])
        #self.mat = torch.Tensor([[torch.Tensor([0.]), self.b],[-self.c, self.b-self.c]])
        print("DD=", self.mat[0,0])
        print("Dc=", self.mat[0,1])
        print("Cd=", self.mat[1,0])
        print("CC=", self.mat[1,1])
        
        self.mv = torch.max(self.mat)
        print("mv=", self.mv)
        print("mat=", self.mat)
        print("norm mat=", self.mat/self.mv)

    def set_active_agents(self, idxs):
        self.active_agents = ["agent_" + str(r) for r in idxs]
        self.n_active_agents = len(idxs)

    # this cache ensures that same space object is returned for the same agent
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return Discrete(self.obs_space_size)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if (self.fraction == True):
            return Box(low=torch.Tensor([-0.001]), high=torch.Tensor([1.001]))
        else:
            return Discrete(self.n_actions)

    def close(self):
        pass
           
    def reset(self, mult_in=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.active_agents}

        self.state = {agent: None for agent in self.active_agents}

        self.num_moves = 0

        observations = {agent: torch.Tensor([0.]) for agent in self.active_agents}

        return observations

    def get_coins(self):
        return self.coins
    
    def step(self, actions):
        '''
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        '''

        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        rewards = {}
    
        ag0 = self.active_agents[0]
        ag1 = self.active_agents[1]

        #print("actions=", actions)
        if (actions[ag0] == 0 and actions[ag1] == 0):
            rewards[ag0] = self.mat[0,0]
            rewards[ag1] = self.mat[0,0]
        elif (actions[ag0] == 0 and actions[ag1] == 1):
            rewards[ag0] = self.mat[0,1]
            rewards[ag1] = self.mat[1,0] 
        elif (actions[ag0] == 1 and actions[ag1] == 0):
            rewards[ag0] = self.mat[1,0]
            rewards[ag1] = self.mat[0,1]
        elif (actions[ag0] == 1 and actions[ag1] == 1):
            rewards[ag0] = self.mat[1,1]
            rewards[ag1] = self.mat[1,1]
        #print("rewards=", rewards)

        self.num_moves += 1
        env_done = self.num_moves >= self.num_game_iterations

        observations = {agent: torch.Tensor([0.]) for agent in self.active_agents}

        if env_done:
            self.agents = []

        return observations, rewards, env_done, self.infos