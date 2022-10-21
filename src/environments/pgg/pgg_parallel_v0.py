import functools
from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
#import numpy as np
import random
import src.analysis.utils as U
from torch.distributions import uniform, normal
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
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(config)
    # This wrapper is only for environments which print results to the terminal
    # env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    if (config.fraction == False):
        env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(config):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = parallel_env(config)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {
        'render.modes': ['human'], 
        "name": "pgg_parallel_v0"
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
        - mult_factor (list with two numers, bounduaries or the same: [0., 5.] or [1., 1.])
        - uncertainies (list with uncertainty for every agent)
        '''

        for key, val in config.items(): setattr(self, key, val)

        self.z_value = torch.Tensor([4.]).to(device) # max numer of sigma that I want to check if I am away from the mean
         
        if hasattr(self.mult_fact, '__len__'):
            self.min_mult = self.mult_fact[0]
            self.max_mult = self.mult_fact[1]
        else: 
            self.min_mult = self.mult_fact
            self.max_mult = self.mult_fact

        self.possible_agents = ["agent_" + str(r) for r in range(self.n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        if (self.uncertainties is not None):
            assert (self.n_agents == len(self.uncertainties))
            self.uncertainties_dict = {}
            self.min_observable_mult = {}
            self.max_observable_mult = {}
            for idx, agent in enumerate(self.agents):
                self.uncertainties_dict[agent] = self.uncertainties[idx]
                self.min_observable_mult[agent] = self.min_mult - \
                    self.z_value*self.uncertainties_dict[agent]
                self.max_observable_mult[agent] = self.max_mult + \
                    self.z_value*self.uncertainties_dict[agent]
        else: 
            self.uncertainties_dict = {agent: 0 for agent in self.agents}
        self.uncertainty_eps = torch.Tensor([0.00001])
        self.n_actions = 2 # give money, keep money
        self.obs_space_size = 2 # I can observe the amount of money I have (precisely), and the multiplicative fctor (with uncertaity)

        self.current_multiplier = 0

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

    def assign_coins_fixed(self):
        self.coins = {}
        self.normalized_coins = {}
        for agent in self.agents:
            self.coins[agent] = 4. # what they have
            self.normalized_coins[agent] = 0. # what they see


    def assign_coins_uniform(self):
        self.coins = {}
        self.normalized_coins = {}
        d = uniform.Uniform(self.coins_min, self.coins_max)
        for agent in self.agents:
            self.coins[agent] = d.sample()
            self.normalized_coins[agent] = (self.coins[agent] - self.coins_min)/(self.coins_max - self.coins_min)

    def assign_coins_normal(self):
        self.coins = {}
        self.normalized_coins = {}
        d = normal.Normal(torch.Tensor([self.coins_mean]), torch.Tensor([self.coins_var]+self.uncertainty_eps)) # is not var, is std. wrong name I put
        for agent in self.agents:
            coin = d.sample()
            while coin < 0.:
                coin = d.sample()
            self.coins[agent] = coin
            self.normalized_coins[agent] = (self.coins[agent] - self.coins_min)/(self.coins_max - self.coins_min)
        
    def observe(self):

        self.observations = {}
        for agent in self.agents:
            d = normal.Normal(torch.Tensor([self.current_multiplier]), torch.Tensor([self.uncertainties_dict[agent]+self.uncertainty_eps])) # is not var, is std. wrong name I put
            obs_multiplier = d.sample()#np.random.normal(self.current_multiplier, self.uncertainties_dict[agent], 1)[0]
            if obs_multiplier < 0.:
                obs_multiplier = torch.Tensor([0.])

            # normalize the observed mutiplier (after this has been observed with uncertainty)
            if self.normalize_nn_inputs == True:
                if (self.min_mult == self.max_mult):
                    obs_multiplier_norm = torch.Tensor([0.])
                else:
                    obs_multiplier_norm = (obs_multiplier - self.min_observable_mult[agent])/(self.max_observable_mult[agent] - self.min_observable_mult[agent])
                self.observations[agent] = torch.Tensor((self.normalized_coins[agent], obs_multiplier_norm)).to(device) 
            # or if I don't normalize
            else:
                self.observations[agent] = torch.Tensor((self.coins[agent], obs_multiplier)).to(device)       
        
    def reset(self, coins=None, unc=None, mult_in=None, seed=123):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        """
        self.seed = seed
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        if (mult_in is not None):
            self.current_multiplier = mult_in
        else:
            if hasattr(self.mult_fact, '__len__'):
                #self.current_multiplier = random.uniform(self.min_mult, self.max_mult)
                #self.current_multiplier = random.sample([0,10], 1)[0]
                self.current_multiplier = torch.Tensor(random.sample(self.mult_fact,1)).to(device)
            else: 
                self.current_multiplier = self.mult_fact

        self.state = {agent: None for agent in self.agents}

        if (coins is not None):
            self.assign_coins2(coins)
        else:
            self.assign_coins_fixed()

        if (unc is not None):
            for agent in self.agents:
                self.uncertainties_dict[agent] = unc
        
        self.num_moves = 0

        self.observe()

        return self.observations

    def get_coins(self):
        return self.coins

    def communication_rewards(self, messages, actions):

        mut_infos = {}
        for _, agent in self.agents.dict():
            mut_info = 0
            for _, agent1 in self.agents.dict():
                mut_info += U.calc_mutinfo(actions[agent], messages[agent1], self.n_actions, self.n_messages)
            mut_infos[agent] = mut_info
         
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
        
        # this means that the actions are not the amount of coins, 
        # but the percentage of coins that agents put
        common_pot = torch.sum(torch.Tensor([self.coins[agent]*actions[agent] for agent in self.agents])).to(device)

        for agent in self.agents:
            rewards[agent] = common_pot/self.n_agents*self.current_multiplier + \
                (self.coins[agent]-self.coins[agent]*actions[agent])
            #self.coins[agent] = rewards[agent]

        self.num_moves += 1
        env_done = self.num_moves >= self.num_game_iterations
        # The dones dictionary must be updated for all players.
        self.dones = {agent: self.num_moves >= self.num_game_iterations for agent in self.agents}

        observations = {}
        #print("rewards=", rewards)

        # assign new amoung of coins
        if (self.num_game_iterations > 1):

            # assign new amount of coins for next round
            self.assign_coins_fixed()

            self.observe()

        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, env_done, infos
