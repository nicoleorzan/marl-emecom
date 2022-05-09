import functools
from gym.spaces import Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
import numpy as np

KEEP = 0 # give my money to pot
GIVE = 1 # keep my money
MOVES = ["GIVE", "KEEP"]
NUM_ITERS = 1

def make_env(n_agents, n_total_coins):
    print("env1")
    '''
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_env(n_agents, n_total_coins)
    # This wrapper is only for environments which print results to the terminal
    #env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(n_agents, n_total_coins):
    print("rew_env1")
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = parallel_env(n_agents, n_total_coins)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self, n_agents, n_total_coins):
        print("parallel class")
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.n_agents = n_agents
        self.n_total_coins = n_total_coins
        self.possible_agents = ["player_" + str(r) for r in range(self.n_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.n_actions = 2 # give money, keep money
        self.obs_space_size = 2 # I can observe the amount of money I have (precisely), and the multiplicative fctor (with uncertaity)

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self._action_spaces = {agent: Discrete(self.n_actions) for agent in self.possible_agents}
        self._observation_spaces = {agent: Discrete(self.obs_space_size) for agent in self.possible_agents}

    
    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return Discrete(self.obs_space_size)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.n_actions)

    """def render(self, mode="human"):
        
        #Renders the environment. In human mode, it can print to terminal, open
        #up a graphical window, or open up some other display that a human can see and understand.
        
        if len(self.agents) == 2:
            string = ("Current state: Agent1: {} , Agent2: {}".format(MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]))
        else:
            string = "Game over"
        print(string)"""
    
    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''
        # observation of one agent is the number of coins he possesses
        return self.coins[agent]

    def close(self):
        pass

    def reset(self):
        '''
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        '''
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        # every agent has the same amount of coins
        self.coins = {agent: int(self.n_total_coins/self.n_agents) for agent in self.agents}
        self.num_moves = 0
        observations = {agent: None for agent in self.agents}
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
        common_pot = np.sum([self.coins[agent] for agent in self.n_players if self.state[agent] == 1])

        for idx, agent in enumerate(self.agents):
            if (actions[idx] == 1):
                self.rewards[agent] = common_pot/self.n_players*self.multiplier
            else:
                self.rewards[agent] = common_pot/self.n_players*self.multiplier + self.coins[agent]

        self.num_moves += 1
        env_done = self.num_moves >= NUM_ITERS
        dones = {agent: env_done for agent in self.agents}

         # current observation is just the other player's most recent action
        observations = {self.agents[i]: None for i in range(len(self.agents))}

        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, dones, infos