#import gym
from gym.spaces import Discrete, Box
import numpy as np
import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

# azione 1 e` cooperativa

def env(n_agents, threshold, num_iterations=1, uncertainties=None, fraction=False, comm=False):
    '''
    The env function often wraps the environment in wrappers by default.
    '''
    env = raw_env(n_agents, threshold, num_iterations, uncertainties, fraction, comm)
    # This wrapper is only for environments which print results to the terminal
    #env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    if (fraction == False):
        env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    '''
    metadata = {
        'render.modes': ['human'], 
        "name": "pgg_v1",
        "is_parallelizable": True
        }

    def __init__(self, n_agents, threshold, num_iterations, uncertainties, fraction, comm):
        '''
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.n_agents = n_agents
        self.threshold = threshold # numero minimo di agenti che devono contribuire k la public pot valga
        self.fraction = fraction
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
        self.obs_space_size = 2 # I can observe the amount of money I have (precisely)

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        if (self.fraction == True):
            self._action_spaces = {agent: Box(low=np.array([0.],dtype=np.float32), high=np.array([1.],dtype=np.float32), dtype=np.float32) for agent in self.possible_agents}            
        else:
            self._action_spaces = {agent: Discrete(self.n_actions) for agent in self.possible_agents}
        self._observation_spaces = {agent: Discrete(self.obs_space_size) for agent in self.possible_agents}
        self.reset()

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
    
    def observe(self, agent):

        obs_coins = np.random.normal(self.coins[agent], self.uncertainties[agent], 1)[0]
        # observation of one agent is the number of coins he possesses
        return np.array(([obs_coins]))

    def close(self):
        pass

    def reset(self, seed=123):

        self.seed = seed
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        if (self.comm):
            self.comm_step = True
        else:
            self.comm_step = False

        self.state = {agent: None for agent in self.agents}
        # agent get a random amount of coin that represents part of the state
        self.coins = {agent: np.random.uniform(0,5) for agent in self.agents} 
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):

        if self.dones[self.agent_selection]:

            return self._was_done_step(action)
        agent = self.agent_selection

        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action
        # collect reward if it is the last agent to act

        if self._agent_selector.is_last():

            num_contributors = np.sum([self.state[agent] for agent in self.agents])

            for agent in self.agents:
                if (self.state[agent] == 0 and num_contributors >= self.threshold):
                     self.rewards[agent] = 1.
                elif (self.state[agent] == 0 and num_contributors < self.threshold):
                    self.rewards[agent] = 0.

                elif (self.state[agent] == 1 and num_contributors >= self.threshold):
                    self.rewards[agent] = 1. - self.coins[agent]
                elif (self.state[agent] == 1 and num_contributors < self.threshold):
                    self.rewards[agent] = - self.coins[agent]

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >= self.num_iterations for agent in self.agents}

        else: 
            # no rewards are allocated until all players take an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()
