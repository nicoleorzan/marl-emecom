import numpy as np

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import 

class Scenario(BaseScenario):
    def make_world(self, N=3, coins=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        world.agents = [Agent() for i in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.coins = coins