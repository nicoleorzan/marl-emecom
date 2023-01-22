import functools
from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
import random
from torch.distributions import uniform, normal
import torch
# azione 1 e` cooperativa

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

def env(n_agents, threshold, num_iterations=1, uncertainties=None, fraction=False, comm=False):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(n_agents, threshold, num_iterations, uncertainties, fraction, comm)
    # This wrapper is only for environments which print results to the terminal
    # env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    if (fraction == False):
        env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(n_agents, threshold, num_iterations, uncertainties, fraction, comm):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = parallel_env(n_agents, threshold, num_iterations, uncertainties, fraction, comm)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {
        'render.modes': ['human'], 
        "name": "pgg_parallel_v0"
        }

    def __init__(self, n_agents, threshold, num_iterations=1, uncertainties=None, fraction=False, comm=False):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.n_agents = n_agents
        self.fraction = fraction
        self.threshold = threshold
        self.num_iterations = num_iterations
        self.possible_agents = ["agent_" + str(r) for r in range(self.n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.comm = comm

        if (uncertainties is not None):
            assert (self.n_agents == len(uncertainties))
            self.uncertainties = {}
            for idx, agent in enumerate(self.agents):
                self.uncertainties[agent] = uncertainties[idx]
        else: 
            self.uncertainties = {agent: 0 for agent in self.agents}
        self.n_actions = 2 # give money, keep money
        self.obs_space_size = 2 # I can observe the amount of money I have (precisely), and the multiplicative fctor (with uncertaity)

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        if (self.fraction == True):
            self._action_spaces = {agent: Box(low=np.array([0.],dtype=np.float32), high=np.array([1.],dtype=np.float32), dtype=np.float32) for agent in self.possible_agents}            
        else:
            self._action_spaces = {agent: Discrete(self.n_actions) for agent in self.possible_agents}
        self._observation_spaces = {agent: Discrete(self.obs_space_size) for agent in self.possible_agents}

        self.uncertainty_min = torch.Tensor([0.0000001])

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
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

    def assign_coins(self):
        #agent get a random amount of coin that represents part of the state
        #if coins is not None:
        #    self.coins = coins
        #else:
        a = torch.randint(low=0, high=10, size=(self.n_agents,))
        while (all(elem == a[0] for elem in a)):
            a = torch.randint(low=0, high=10, size=(self.n_agents,))
        sum_a = torch.sum(a)
        coins = a/sum_a
        self.coins = {agent: coins[idx]*1.5 for idx, agent in enumerate(self.agents)}
        #print("coins=", self.coins)


    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        if (self.comm):
            self.comm_step = True
        else:
            self.comm_step = False

        self.assign_coins()

        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
 
        observations = {}
        for agent in self.agents:
            d = normal.Normal(torch.Tensor([self.coins[agent]]),  self.uncertainties[agent]+self.uncertainty_min) # is not var, is std. wrong name I put
            obs_coins = d.sample()
            observations[agent] = obs_coins.to(device)
        return observations    

    def step(self, actions):

        '''
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        '''
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}

        num_contributors = torch.sum(torch.Tensor([actions[agent] for agent in self.agents]))
        #print("coins=", self.coins)
        #print("actions=", actions)

        for agent in self.agents:
            # defect
            if (actions[agent] == 0 and num_contributors >= self.threshold):
                    rewards[agent] = 1. + self.coins[agent]
            elif (actions[agent] == 0 and num_contributors < self.threshold):
                rewards[agent] = self.coins[agent]
            # cooperate
            elif (actions[agent] == 1 and (num_contributors-1) >= self.threshold-1):
                rewards[agent] = 1.
            elif (actions[agent] == 1 and (num_contributors-1) < self.threshold-1):
                rewards[agent] = 0.
            #self.coins[agent] = rewards[agent]

        self.num_moves += 1
        env_done = self.num_moves >= self.num_iterations
        # The dones dictionary must be updated for all players.
        self.dones = {agent: self.num_moves >= self.num_iterations for agent in self.agents}
        #print(self.dones)

        observations = {}
        #print("rewards=", rewards)

        # assign new amoung of coins
        if (self.num_iterations > 1):

            # assign new amount of coins for next round
            self.assign_coins()

            observations = {}
            for agent in self.agents:
                d = normal.Normal(torch.Tensor([self.coins[agent]]), self.uncertainties[agent]+self.uncertainty_min) # is not var, is std. wrong name I put
                observations[agent] = d.sample.to(device)

            #print("coins=", self.coins)

        
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, env_done, infos
