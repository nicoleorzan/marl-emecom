#import gym
from gym.spaces import Discrete
import numpy as np
import functools
import random
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers


KEEP = 0 # give my money to pot
GIVE = 1 # keep my money
MOVES = ["GIVE", "KEEP"]
NUM_ITERS = 1

def env(n_agents, n_total_coins):
    '''
    The env function often wraps the environment in wrappers by default.
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

class raw_env(AECEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    '''
    metadata = {
        'render.modes': ['human'], 
        "name": "pgg_v0",
        "is_parallelizable": True
        }

    def __init__(self, n_agents, n_total_coins):
        '''
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.n_agents = n_agents
        self.n_total_coins = n_total_coins
        self.multiplier = 5
        self.possible_agents = ["agent_" + str(r) for r in range(self.n_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.n_actions = 2 # give money, keep money
        self.obs_space_size = 2 # I can observe the amount of money I have (precisely), and the multiplicative fctor (with uncertaity)

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
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
        obs_multiplier = np.random.normal(0, self.uncertainties[agent], 1)[0]
        return np.array((self.coins[agent], self.n_agents, obs_multiplier))

    def close(self):
        pass

    def reset(self, seed=123):
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
        self.seed = seed
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        #print("self agents=", self.agents)
        #print("cumul rews=",self._cumulative_rewards)
        #print("self.dones=", self.dones)
        #print("self.infos=", self.infos)
        self.state = {agent: None for agent in self.agents}
        # every agent has the same amount of coins
        self.coins = {agent: int(self.n_total_coins/self.n_agents) for agent in self.agents} 
        self.uncertainties = {agent: random.uniform(0,1) for agent in self.agents} 
        #print("Coins",self.coins)
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''

        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent, or if there are no more done agents, to the next live agent
            return self._was_done_step(action)
        agent = self.agent_selection
        #print("agent=", agent, "self._agent_selector.is_last()=",self._agent_selector.is_last())

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            #print('is last')

            common_pot = np.sum([self.coins[agent] for agent in self.agents if self.state[agent] == 1])

            for agent in self.agents:
                if (self.state[agent] == 1):
                    self.rewards[agent] = common_pot/self.n_agents*self.multiplier
                else:
                    self.rewards[agent] = common_pot/self.n_agents*self.multiplier + self.coins[agent]

            self.num_moves += 1

            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}
            #print("self.num_moves", self.num_moves, "NUM_ITERS=",NUM_ITERS)
            #print("self.dones=", self.dones)

        else: 
            # no rewards are allocated until all players take an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        #print("agent after should be=",self.agent_selection)
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
        #print("cumul rews=",self._cumulative_rewards)
        

