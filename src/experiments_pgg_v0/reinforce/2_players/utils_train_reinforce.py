import numpy as np
import itertools

def find_max_min1(multipliers, coins):
    n_agents = 2
    max_values = {}
    min_values = {}
    coins_per_agent = np.array([coins, coins, coins])

    possible_actions = ["C", "D"]
    possible_scenarios = [''.join(i) for i in itertools.product(possible_actions, repeat = n_agents)]
    scenarios_returns = {}
    for multiplier in multipliers:
        print("\nMultiplier=", multiplier)
        possible_actions = ["C", "D"]
        possible_scenarios = [''.join(i) for i in itertools.product(possible_actions, repeat = n_agents)]
        returns = np.zeros((len(possible_scenarios), n_agents)) # TUTTI I POSSIBILI RITORNI CHE UN AGENTE PUO OTTENERE, PER OGNI AGENTE

        for idx_scenario, scenario in enumerate(possible_scenarios):
            common_pot = np.sum([coins_per_agent[i] for i in range(n_agents) if scenario[i] == "C"])

            for ag_idx in range(n_agents):
                if (scenario[ag_idx] == 'C'):
                    returns[idx_scenario, ag_idx] = common_pot/n_agents*multiplier
                else: 
                    returns[idx_scenario, ag_idx] = common_pot/n_agents*multiplier + coins_per_agent[ag_idx]
            print("scenario=", scenario, "common_pot=", common_pot, "return=", returns[idx_scenario])
        max_values[multiplier] = np.amax(returns)
        min_values[multiplier] = np.amin(returns)
        #print("super min values=", min_values([min(min_values, key=min_values.get)]))
        #print("super max=", min_values([min(min_values, key=min_values.get)]))
        print(" max_values[", multiplier, "]=",  max_values[multiplier])
        print(" min_values[", multiplier, "]=",  min_values[multiplier])
        print("normalized=", (returns-min(min_values))/(max(max_values) - min(min_values)))

        scenarios_returns[multiplier] = returns
    print(max_values)
    print(min_values)
    min_min = min(min_values.items(), key=lambda x: x[1])
    max_max = max(max_values.items(), key=lambda x: x[1])
    print(scenarios_returns)

    return max_values


def find_max_min(multipliers, coins):
    n_agents = 2
    max_values = {}
    min_values = {}
    coins_per_agent = np.array([coins, coins, coins])

    possible_actions = ["C", "D"]
    possible_scenarios = [''.join(i) for i in itertools.product(possible_actions, repeat = n_agents)]

    for multiplier in multipliers:
        print("\nMultiplier=", multiplier)
        a = 0; b = 0; c = 0
        possible_actions = ["C", "D"]
        possible_scenarios = [''.join(i) for i in itertools.product(possible_actions, repeat = n_agents)]
        returns = np.zeros((len(possible_scenarios), n_agents)) # TUTTI I POSSIBILI RITORNI CHE UN AGENTE PUO OTTENERE, PER OGNI AGENTE

        scenarios_returns = {}
        for idx_scenario, scenario in enumerate(possible_scenarios):
            common_pot = np.sum([coins_per_agent[i] for i in range(n_agents) if scenario[i] == "C"])

            for ag_idx in range(n_agents):
                if (scenario[ag_idx] == 'C'):
                    returns[idx_scenario, ag_idx] = common_pot/n_agents*multiplier
                else: 
                    returns[idx_scenario, ag_idx] = common_pot/n_agents*multiplier + coins_per_agent[ag_idx]
            print("scenario=", scenario, "common_pot=", common_pot, "return=", returns[idx_scenario])

        max_values[multiplier] = np.amax(returns)
        min_values[multiplier] = np.amin(returns)
        print(" max_values[", multiplier, "]=",  max_values[multiplier])
        print(" min_values[", multiplier, "]=",  min_values[multiplier])
        print("normalized=", returns/max_values[multiplier]) #(returns-min_values[multiplier])/(max_values[multiplier] - min_values[multiplier]))
    return max_values



def eval(config, parallel_env, agents_dict, m, _print=False):
    observations = parallel_env.reset(m)

    if (_print == True):
        print("* Eval ===> Mult factor=", m)
        print("obs=", observations)
    #actions_agents = torch.zeros(config.n_agents)

    done = False
    while not done:

        actions = {agent: agents_dict[agent].select_action(observations[agent], True) for agent in parallel_env.agents}
        
        #for idx in range(config.n_agents):
        #    actions_agents[idx] = actions["agent_"+str(idx)]
        out = {agent: agents_dict[agent].get_distribution(observations[agent]) for agent in parallel_env.agents}

        _, rewards_eval, _, _ = parallel_env.step(actions)
        
        if (_print == True):
            print("actions=", actions)
            print("distributions", out)

        done = True

    return actions, out, rewards_eval #np.mean([actions["agent_"+str(idx)] for idx in range(config.n_agents)], dtype=object), out


#find_max_min1([0., 1.5, 3.], 4)