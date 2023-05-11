import torch
import itertools
import numpy as np

def eval(config, parallel_env, active_agents, active_agents_idxs, m, device, _print=False):
    observations = parallel_env.reset(m)

    if (_print == True):
        print("\n Eval ===> Mult factor=", m)
        print("obs=", observations)

    n_communicating_agents = config.communicating_agents.count(1)
    message = None
    done = False
    while not done:

        messages = {}
        actions = {}
        mex_distrib = {}
        act_distrib = {}
        active_agents["agent_"+str(active_agents_idxs[0])].digest_input((observations["agent_"+str(active_agents_idxs[0])], active_agents_idxs[1], active_agents["agent_"+str(active_agents_idxs[1])].reputation))
        active_agents["agent_"+str(active_agents_idxs[1])].digest_input((observations["agent_"+str(active_agents_idxs[1])], active_agents_idxs[0], active_agents["agent_"+str(active_agents_idxs[0])].reputation))

        # speaking
        for agent in parallel_env.active_agents:
            if (active_agents[agent].is_communicating):
                messages[agent] = active_agents[agent].select_message(_eval=True)
                mex_distrib[agent] = active_agents[agent].get_message_distribution()

        # listening
        for agent in parallel_env.active_agents:
            other = list(set(active_agents_idxs) - set([active_agents[agent].idx]))[0]
            if (active_agents[agent].is_listening and len(messages) != 0):
                active_agents[agent].get_message(messages["agent_"+str(other)])
            if (active_agents[agent].is_listening and n_communicating_agents != 0 and len(messages) == 0):
                message = torch.zeros(config.mex_size)
                active_agents[agent].get_message(message)

        # acting
        for agent in parallel_env.active_agents:
            actions[agent] = active_agents[agent].select_action(_eval=True)
            act_distrib[agent] = active_agents[agent].get_action_distribution()
                
        _, rewards_eval, _, _ = parallel_env.step(actions)

        if (_print == True):
            print("message=", message)
            print("actions=", actions)
            print("distrib=", act_distrib)
        observations, _, done, _ = parallel_env.step(actions)

    return actions, mex_distrib, act_distrib, rewards_eval 


def eval_old(config, parallel_env, agents, m, device, _print=False):
    observations = parallel_env.reset(m)

    if (_print == True):
        print("\n Eval ===> Mult factor=", m)
        print("obs=", observations)

    done = False
    while not done:

        if (config.random_baseline):
            messages = {agent: agents[agent].random_messages(observations[agent]) for agent in parallel_env.agents}
            mex_distrib = 0
        else:
            messages = {agent: agents[agent].select_message(observations[agent], True) for agent in parallel_env.agents}
            mex_distrib = {agent: agents[agent].get_message_distribution(observations[agent]) for agent in parallel_env.agents}
        message = torch.stack([v for _, v in messages.items()]).view(-1).to(device)
        actions = {agent: agents[agent].select_action(observations[agent], message, True) for agent in parallel_env.agents}
        acts_distrib = {agent: agents[agent].get_action_distribution(observations[agent], message) for agent in parallel_env.agents}
        
        _, rewards_eval, _, _ = parallel_env.step(actions)

        if (_print == True):
            print("message=", message)
            print("actions=", actions)
        observations, _, done, _ = parallel_env.step(actions)

    return actions, mex_distrib, acts_distrib, rewards_eval 

def find_max_min(config, coins):
    n_agents = config.n_agents
    multipliers = config.mult_fact
    max_values = {}
    min_values = {}
    coins_per_agent = np.array([coins for i in range(n_agents)])

    possible_actions = ["C", "D"]
    possible_scenarios = [''.join(i) for i in itertools.product(possible_actions, repeat = n_agents)]

    for multiplier in multipliers:
        #print("\nMultiplier=", multiplier)
        possible_actions = ["C", "D"]
        possible_scenarios = [''.join(i) for i in itertools.product(possible_actions, repeat = n_agents)]
        returns = np.zeros((len(possible_scenarios), n_agents)) # TUTTI I POSSIBILI RITORNI CHE UN AGENTE PUO OTTENERE, PER OGNI AGENTE

        #scenarios_returns = {}
        for idx_scenario, scenario in enumerate(possible_scenarios):
            common_pot = np.sum([coins_per_agent[i] for i in range(n_agents) if scenario[i] == "C"])

            for ag_idx in range(n_agents):
                if (scenario[ag_idx] == 'C'):
                    returns[idx_scenario, ag_idx] = common_pot/n_agents*multiplier
                else: 
                    returns[idx_scenario, ag_idx] = common_pot/n_agents*multiplier + coins_per_agent[ag_idx]
            #print("scenario=", scenario, "common_pot=", common_pot, "return=", returns[idx_scenario])

        max_values[multiplier] = np.amax(returns)
        min_values[multiplier] = np.amin(returns)
        #print(" max_values[", multiplier, "]=",  max_values[multiplier])
        #print(" min_values[", multiplier, "]=",  min_values[multiplier])
        #print("normalized=", returns/max_values[multiplier]) #(returns-min_values[multiplier])/(max_values[multiplier] - min_values[multiplier]))
    return max_values

def apply_norm(active_agents, active_agents_idxs, actions):
    #print("actions=", actions)
    for idx in active_agents_idxs:
        #print("agent=", idx)
        agent = active_agents["agent_"+str(idx)]
        change_reputation(agent, actions["agent_"+str(idx)])

def change_reputation(agent, action):
    #print("reputation before=", agent.reputation)
    if (action == 0 and agent.reputation >= 0.05):
        agent.reputation -= 0.01
    if (action == 1 and agent.reputation <= (1.-0.05)):
        agent.reputation += 0.01
    #print("reputation after=", agent.reputation)
