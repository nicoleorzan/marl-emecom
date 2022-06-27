import numpy as np
import itertools

n_agents = 3
multiplier = 1.3
print("Multiplier=", multiplier)
possible_actions = ["C", "D"]
possible_scenarios = [''.join(i) for i in itertools.product(possible_actions, repeat = n_agents)]
print(possible_scenarios)

starting_coins_per_agent = 4
coins_per_agent = np.zeros((len(possible_scenarios)))
coins_per_agent.fill(starting_coins_per_agent)
print("coins per agent=", coins_per_agent)

returns = np.zeros((len(possible_scenarios))) # TUTTI I POSSIBILI RITORNI CHE UN AGENTE PUO OTTENERE

for idx_scenario, scenario in enumerate(possible_scenarios):
    common_pot = np.sum([coins_per_agent[i] for i in range(n_agents) if scenario[i] == "C"])

    if (scenario[0] == 'C'):
        returns[idx_scenario] = common_pot/n_agents*multiplier
    else: 
        returns[idx_scenario] = common_pot/n_agents*multiplier + coins_per_agent[0]
    print("scenario=", scenario, "common_pot=", common_pot, "return=", returns[idx_scenario])
#print("coins per agent", coins_per_agent)