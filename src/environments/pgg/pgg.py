#import gym
from gym.spaces import Discrete, Box
import numpy as np
import functools
import random
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

# azione 1 e` cooperativa

def env(n_agents, coins_per_agent, num_iterations=1, mult_fact=None, uncertainties=None, fraction=False):
    '''
    The env function often wraps the environment in wrappers by default.
    '''
    env = raw_env(n_agents, coins_per_agent, num_iterations, mult_fact, uncertainties, fraction)
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
        "name": "pgg_v0",
        "is_parallelizable": True
        }

    def __init__(self, n_agents, coins_per_agent, num_iterations, mult_fact, uncertainties, fraction):
        '''
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.n_agents = n_agents
        self.coins_per_agent = coins_per_agent
        self.fraction = fraction
        self.mult_fact = mult_fact if mult_fact != None else 1
        #if hasattr(mult_factor, '__len__'):
        #    self.mult_factors = mult_factor
        self.num_iterations = num_iterations
        self.possible_agents = ["agent_" + str(r) for r in range(self.n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

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
            self._action_spaces = {agent: Box(low=np.array([-0.001],dtype=np.float32), high=np.array([1.001],dtype=np.float32), dtype=np.float32) for agent in self.possible_agents}            
        else:
            self._action_spaces = {agent: Discrete(self.n_actions) for agent in self.possible_agents}
        self._observation_spaces = {agent: Discrete(self.obs_space_size) for agent in self.possible_agents}
        self.current_multiplier = 0
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
        obs_multiplier = np.random.normal(self.current_multiplier, self.uncertainties[agent], 1)[0]
        if obs_multiplier < 0.:
            obs_multiplier = 0.
        return np.array((self.coins[agent], obs_multiplier))

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

        if hasattr(self.mult_fact, '__len__'):
            self.current_multiplier = random.choice(self.mult_fact)
        else: 
            self.current_multiplier = self.mult_fact
        #print("self agents=", self.agents)
        #print("cumul rews=",self._cumulative_rewards)
        #print("self.dones=", self.dones)
        #print("self.infos=", self.infos)
        self.state = {agent: None for agent in self.agents}
        # every agent has the same amount of coins
        self.coins = {agent: self.coins_per_agent for agent in self.agents} 
        #self.uncertainties = {agent: 0 for agent in self.agents} #{agent: random.uniform(0,1) for agent in self.agents} 
        #print("Coins",self.coins)
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

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

            if self.fraction == True: 
                # this means that the actions are not the amount of coins, 
                # but the percentage of coins that agents put
                common_pot = np.sum([self.coins[agent]*self.state[agent] for agent in self.agents])

                for agent in self.agents:
                    self.rewards[agent] = common_pot/self.n_agents*self.current_multiplier + \
                        (self.coins[agent]-self.coins[agent]*self.state[agent])
                    self.coins[agent] = self.rewards[agent]
            else: 
                # agents only decide if to put the coins or not
                common_pot = np.sum([self.coins[agent] for agent in self.agents if self.state[agent] == 1])

                for agent in self.agents:
                    if (self.state[agent] == 1):
                        self.rewards[agent] = common_pot/self.n_agents*self.current_multiplier
                    else:
                        self.rewards[agent] = common_pot/self.n_agents*self.current_multiplier + self.coins[agent]
                    self.coins[agent] = self.rewards[agent]

            self.num_moves += 1

            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >= self.num_iterations for agent in self.agents}
            #self.dones = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}
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
        

