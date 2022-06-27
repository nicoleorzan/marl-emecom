import numpy as np
import itertools

n_agents = 3
threshold = 2
n_experiments = 1000
possible_actions = ["C", "D"]
possible_agents = ["agent_" + str(r) for r in range(n_agents)]
possible_scenarios = [''.join(i) for i in itertools.product(possible_actions, repeat = n_agents)]
print(possible_scenarios)

print("Reward function 1")

returns = np.zeros((n_experiments, len(possible_scenarios))) # TUTTI I RITORNI CHE UN AGENTE PUO OTTENERE NEI DIVERSI SCENARI
for experiment in range(n_experiments):
    list_coins = [np.random.uniform(0,1) for i in range(n_agents)]
    coins_per_agent = np.array(list_coins)

    for idx_scenario, scenario in enumerate(possible_scenarios):
        if (scenario[0] == 'D'):
            if (scenario.count('C') >= threshold):
                returns[experiment, idx_scenario] = 1 + coins_per_agent[0]
            elif (scenario.count('C') < threshold):
                returns[experiment, idx_scenario] = coins_per_agent[0]
        elif (scenario[0] == 'C'):

            if (scenario.count('C') >= threshold):
                returns[experiment, idx_scenario] = 1
            elif (scenario.count('C') < threshold ):
                returns[experiment, idx_scenario] = 0

returns_avg = np.mean(returns, axis=0) 
for idx_scenario, scenario in enumerate(possible_scenarios):      
    print("scenario=", scenario, "returns_avg=", returns_avg[idx_scenario])

"""print("=========================Reward function 2===============================")

returns = np.zeros((n_experiments, len(possible_scenarios))) # TUTTI I RITORNI CHE UN AGENTE PUO OTTENERE NEI DIVERSI SCENARI
for experiment in range(n_experiments):
    list_coins = [np.random.uniform(0,1) for i in range(n_agents)]
    coins_per_agent = np.array(list_coins)

    for idx_scenario, scenario in enumerate(possible_scenarios):
        if (scenario[0] == 'D'):

            if (scenario.count('C') >= threshold):
                returns[experiment, idx_scenario] = 1
            elif (scenario.count('C') < threshold):
                returns[experiment, idx_scenario] = 0
        elif (scenario[0] == 'C'):

            if (scenario.count('C') >= threshold):
                returns[experiment, idx_scenario] = 1-coins_per_agent[0]
            elif (scenario.count('C') < threshold ):
                returns[experiment, idx_scenario] = -coins_per_agent[0]

returns_avg = np.mean(returns, axis=0) 
for idx_scenario, scenario in enumerate(possible_scenarios):      
    print("scenario=", scenario, "returns_avg=", returns_avg[idx_scenario])"""