import functools
from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
import numpy as np
import random
import src.analysis.utils as U

# azione 1 e` cooperativa

def env(n_agents, coins_per_agent, num_iterations=1):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(n_agents, coins_per_agent, num_iterations)
    # This wrapper is only for environments which print results to the terminal
    # env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(n_agents, num_iterations):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = parallel_env(n_agents, num_iterations)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {
        'render.modes': ['human'], 
        "name": "pgg_parallel_v0"
        }

    def __init__(self, n_agents, num_iterations):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.n_agents = n_agents
        self.num_iterations = num_iterations
        self.possible_agents = ["agent_" + str(r) for r in range(self.n_agents)]
        self.agents = self.possible_agents[:]
        self.uncertainties = {}
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

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

    def assign_coins_normal(self):
        self.coins = {}
        for agent in self.agents:
            coin = np.random.normal(self.coins_mean, self.coins_var, 1)[0]
            if coin < 0.:
                coin = np.random.normal(self.coins_mean, self.coins_var, 1)[0]
            self.coins[agent] = coin

    def assign_coins_uniform(self, _min, _max):
        self.coins = {}
        for agent in self.agents:
            coin = np.random.uniform(_min, _max, 1)[0]
            self.coins[agent] = coin
        
    def reset(self, coins='uniform', unc=None, mult_in=None, seed=123):
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

        self.current_multiplier = mult_in
        self.state = {agent: None for agent in self.agents}

        if (unc is not None):
            for agent in self.agents:
                self.uncertainties[agent] = unc

        if (coins == 'uniform'):
            self.assign_coins_uniform(_min=1, _max=3)
        
        self.num_moves = 0

        self.observations = {}
        for agent in self.agents:
            obs_multiplier = np.random.normal(self.current_multiplier, self.uncertainties[agent], 1)[0]
            if obs_multiplier < 0.:
                obs_multiplier = 0.
            self.observations[agent] = np.array((self.coins[agent], obs_multiplier))

        return self.observations  

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
        common_pot = np.sum([self.coins[agent]*actions[agent] for agent in self.agents])

        for agent in self.agents:
            rewards[agent] = common_pot/self.n_agents*self.current_multiplier + \
                (self.coins[agent]-self.coins[agent]*actions[agent])
            #self.coins[agent] = rewards[agent]

        self.num_moves += 1
        env_done = self.num_moves >= self.num_iterations
        # The dones dictionary must be updated for all players.
        self.dones = {agent: self.num_moves >= self.num_iterations for agent in self.agents}

        observations = {}
        #print("rewards=", rewards)

        # assign new amoung of coins
        if (self.num_iterations > 1):

            # assign new amount of coins for next round
            self.assign_coins()

            observations = {}
            for agent in self.agents:
                obs_multiplier = np.random.normal(self.current_multiplier, self.uncertainties[agent], 1)[0]
                if obs_multiplier < 0.:
                    obs_multiplier = 0.
                observations[agent] = np.array((self.coins[agent], obs_multiplier))

            #print("coins=", self.coins)


        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, env_done, infos
