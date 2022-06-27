import functools
from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec, aec_to_parallel
import numpy as np
import random

# azione 1 e` cooperativa

def env(n_agents, coins_per_agent, num_iterations=1, mult_fact=None, uncertainties=None, fraction=False, comm=False):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(n_agents, coins_per_agent, num_iterations, mult_fact, uncertainties, fraction, comm)
    # This wrapper is only for environments which print results to the terminal
    # env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    if (fraction == False):
        env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(n_agents, coins_per_agent, num_iterations, mult_fact, uncertainties, fraction, comm):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = parallel_env(n_agents, coins_per_agent, num_iterations, mult_fact, uncertainties, fraction, comm)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {
        'render.modes': ['human'], 
        "name": "pgg_parallel_v0"
        }

    def __init__(self, n_agents, coins_per_agent, num_iterations, mult_fact=None, uncertainties=None, fraction=False, comm=False):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.n_agents = n_agents
        self.coins_per_agent = coins_per_agent
        self.fraction = fraction
        self.mult_fact = mult_fact if mult_fact != None else 1
        if hasattr(mult_fact, '__len__'):
            self.min_mult = mult_fact[0]
            self.max_mult = mult_fact[1]
        else: 
            self.min_mult = mult_fact
            self.max_mult = mult_fact
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
            return Box(low=np.array([-0.001],dtype=np.float32), high=np.array([1.001],dtype=np.float32))            
        else:
            return Discrete(self.n_actions)

    def close(self):
        pass

    def reset(self, seed=123):
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
        if (self.comm):
            self.comm_step = True
        else:
            self.comm_step = False

        if hasattr(self.mult_fact, '__len__'):
            self.current_multiplier = random.uniform(self.min_mult, self.max_mult)
        else: 
            self.current_multiplier = self.mult_fact

        self.state = {agent: None for agent in self.agents}
        # every agent has the same amount of coins
        self.coins = {agent: self.coins_per_agent for agent in self.agents} 
        self.num_moves = 0

        self.observations = {}
        for agent in self.agents:
            obs_multiplier = np.random.normal(self.current_multiplier, self.uncertainties[agent], 1)[0]
            if obs_multiplier < 0.:
                obs_multiplier = 0.
            self.observations[agent] = np.array((self.coins[agent], obs_multiplier))

        return self.observations    

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
        common_pot = np.sum([self.coins[agent]*actions[agent] for agent in self.agents])

        for agent in self.agents:
            rewards[agent] = common_pot/self.n_agents*self.current_multiplier + \
                (self.coins[agent]-self.coins[agent]*actions[agent])
            self.coins[agent] = rewards[agent]

        self.num_moves += 1
        env_done = self.num_moves >= self.num_iterations
        # The dones dictionary must be updated for all players.
        self.dones = {agent: self.num_moves >= self.num_iterations for agent in self.agents}

        observations = {}
        for agent in self.agents:
            obs_multiplier = np.random.normal(self.current_multiplier, self.uncertainties[agent], 1)[0]
            if obs_multiplier < 0.:
                obs_multiplier = 0.
            observations[agent] = np.array((self.coins[agent], obs_multiplier))

        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, env_done, infos
