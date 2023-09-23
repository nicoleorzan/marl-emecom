import torch
import itertools
import numpy as np
import random

def utility(mf,actions):
    coins = 4
    common_pot = torch.sum(torch.Tensor([coins*a for a in actions]))

    return common_pot/2*mf+(coins-coins*actions[0])

def introspective_rewards(config, observations, active_agents, parallel_env, rewards, actions):
    new_rewards = {}
    for ag_idx, _ in active_agents.items():
        #print("agent=", ag_idx, actions[ag_idx])
        #print("obs mf=", observations[ag_idx])
        s = utility(observations[ag_idx],[actions[ag_idx],actions[ag_idx]])
        #print("s=",s)
        new_rewards[ag_idx] = config.alpha*rewards[ag_idx] + (1-config.alpha)*s
    return new_rewards

def pick_agents_idxs(config):

    active_agents_idxs = []
    if (config.non_dummy_idxs != []):
        first_agent_idx = random.sample(config.non_dummy_idxs, 1)[0] 
    else: 
        first_agent_idx = random.sample([i for i in range(config.n_agents)], 1)[0]
    second_agent_idx = random.sample( list(set(range(0, config.n_agents)) - set([first_agent_idx])) , 1)[0]
    active_agents_idxs = [first_agent_idx, second_agent_idx]

    return active_agents_idxs

def pick_agents_idxs_opponent_selection(config, agents):

    active_agents_idxs = []
    if (config.non_dummy_idxs != []):
        first_agent_idx = random.sample(config.non_dummy_idxs, 1)[0] 
    else: 
        first_agent_idx = random.sample([i for i in range(config.n_agents)], 1)[0]
    reputations = torch.stack([agents["agent_"+str(i)].reputation for i in range(config.n_agents) if i!= first_agent_idx], dim=1).long()[0]
    print(reputations)
    second_agent_idx = agents["agent_"+str(first_agent_idx)].select_opponent(reputations)
    active_agents_idxs = [first_agent_idx, second_agent_idx]
    print("active_agents_idxs=",active_agents_idxs)

    return active_agents_idxs

def eval(config, parallel_env, active_agents, active_agents_idxs, m, device, _print=False):
    #print("\nEVAL<<<=====================")
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

        if (config.get_index == True ):
            active_agents["agent_"+str(active_agents_idxs[0])].digest_input_with_idx((observations["agent_"+str(active_agents_idxs[0])], active_agents_idxs[1], active_agents["agent_"+str(active_agents_idxs[1])].reputation))
            active_agents["agent_"+str(active_agents_idxs[1])].digest_input_with_idx((observations["agent_"+str(active_agents_idxs[1])], active_agents_idxs[0], active_agents["agent_"+str(active_agents_idxs[0])].reputation))
        else:
            active_agents["agent_"+str(active_agents_idxs[0])].digest_input((observations["agent_"+str(active_agents_idxs[0])], active_agents["agent_"+str(active_agents_idxs[1])].reputation))
            active_agents["agent_"+str(active_agents_idxs[1])].digest_input((observations["agent_"+str(active_agents_idxs[1])], active_agents["agent_"+str(active_agents_idxs[0])].reputation))

        # speaking
        #print("\nspeaking")
        for agent in parallel_env.active_agents:
            if (active_agents[agent].is_communicating):
                messages[agent] = active_agents[agent].select_message(_eval=True)
                #print("messages["+str(agent)+"]=", messages[agent])
                if (active_agents[agent].is_dummy == False):
                    mex_distrib[agent] = active_agents[agent].get_message_distribution()

        # listening
        #print("\nlistening")
        for agent in parallel_env.active_agents:
            other = list(set(active_agents_idxs) - set([active_agents[agent].idx]))[0]
            if (active_agents[agent].is_listening and len(messages) != 0):
                active_agents[agent].get_message(messages["agent_"+str(other)])
            if (active_agents[agent].is_listening and n_communicating_agents != 0 and len(messages) == 0):
                message = torch.zeros(config.mex_size)
                active_agents[agent].get_message(message)

        # acting
        #print("\nacting")
        for agent in parallel_env.active_agents:
            actions[agent] = active_agents[agent].select_action(_eval=True)
            if (active_agents[agent].is_dummy == False):
                act_distrib[agent] = active_agents[agent].get_action_distribution()
        _, rewards_eval, _, _ = parallel_env.step(actions)

        if (_print == True):
            print("message=", message)
            print("actions=", actions)
            print("distrib=", act_distrib)
        observations, _, done, _ = parallel_env.step(actions)

    return actions, mex_distrib, act_distrib, rewards_eval 

def eval_anast(parallel_env, active_agents, active_agents_idxs, n_iterations, social_norm, gamma):

    [agent.reset() for _, agent in active_agents.items()]
    rewards = {}
    
    _ = parallel_env.reset()
    
    states = {}; next_states = {}
    for idx_agent, agent in active_agents.items():
        other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
        next_states[idx_agent] = torch.cat((other.reputation, agent.reputation, other.previous_action, agent.previous_action), 0)
        agent.state_act = next_states[idx_agent]

    done = False
    for i in range(n_iterations):
        if (i == 9): 
            done = True

        actions = {}; states = next_states
        for idx_agent, agent in active_agents.items():
            agent.state_act = states[idx_agent]
            #print("agent.state_act=", agent.state_act)
        
        # acting
        for agent in parallel_env.active_agents:
            #print("states=", states)
            #print("agent.state_act=", active_agents[agent].state_act)
            a, d = active_agents[agent].select_action(_eval=True)
            #print("a=", a,  "d=", d)
            actions[agent] = a

        _, rew, _, _ = parallel_env.step(actions)
        #print("rew=", rew)
        
        #social_norm.save_actions(actions, active_agents_idxs)
        #social_norm.rule09_binary(active_agents_idxs)
        for ag_idx in active_agents_idxs:       
            active_agents["agent_"+str(ag_idx)].old_reputation = active_agents["agent_"+str(ag_idx)].reputation
            if "agent_"+str(ag_idx) not in rewards.keys():
                rewards["agent_"+str(ag_idx)] = [rew["agent_"+str(ag_idx)]]
            else:
                rewards["agent_"+str(ag_idx)].append(rew["agent_"+str(ag_idx)])

        next_states = {}
        for idx_agent, agent in active_agents.items():
            other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
            next_states[idx_agent] = torch.cat((other.reputation, agent.reputation, other.previous_action, agent.previous_action), 0)
            agent.state_act = next_states[idx_agent]

        #print("next_states=", next_states)
        rew = {key: value+active_agents[key].reputation for key, value in rew.items()}
        #print("after rewards=", rew)

        if done:
            break

    R = {}
    for ag_idx, agent in active_agents.items():
        R[ag_idx] = 0
        for r in rewards[ag_idx][::-1]:
            R[ag_idx] = r + gamma * R[ag_idx]

    return R

def eval_new(parallel_env, active_agents, active_agents_idxs, n_iterations, social_norm, gamma):

    [agent.reset() for _, agent in active_agents.items()]
    rewards = {}
    
    observations = parallel_env.reset()
    
    states = {}; next_states = {}
    for idx_agent, agent in active_agents.items():
        other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
        next_states[idx_agent] = torch.cat((observations[idx_agent], other.reputation, agent.reputation, other.previous_action, agent.previous_action), 0)
        agent.state_act = next_states[idx_agent]

    done = False
    for i in range(n_iterations):
        if (i == 9): 
            done = True

        actions = {}; states = next_states
        
        # acting
        for agent in parallel_env.active_agents:
            actions[agent] = active_agents[agent].select_action()

        observations, rew, _, _ = parallel_env.step1(actions)
        
        social_norm.save_actions(actions, active_agents_idxs)
        social_norm.rule09_binary(active_agents_idxs)
        for ag_idx in active_agents_idxs:       
            active_agents["agent_"+str(ag_idx)].old_reputation = active_agents["agent_"+str(ag_idx)].reputation
            if "agent_"+str(ag_idx) not in rewards.keys():
                rewards["agent_"+str(ag_idx)] = [rew["agent_"+str(ag_idx)]]
            else:
                rewards["agent_"+str(ag_idx)].append(rew["agent_"+str(ag_idx)])

        next_states = {}
        for idx_agent, agent in active_agents.items():
            other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
            next_states[idx_agent] = torch.cat((observations[idx_agent],other.reputation, agent.reputation, other.previous_action, agent.previous_action), 0)
            agent.state_act = next_states[idx_agent]

        #print("next_states=", next_states)
        rew = {key: value+active_agents[key].reputation for key, value in rew.items()}
        #print("after rewards=", rew)

        if done:
            break

    R = {}
    for ag_idx, agent in active_agents.items():
        R[ag_idx] = 0
        for r in rewards[ag_idx][::-1]:
            R[ag_idx] = r + gamma * R[ag_idx]
    return R


def best_strategy_reward(config):
    my_strategy = [0, 1, 1, 1]
    opponent_strategy = [0, 1, 1, 1]
    all_returns_one_agent, returns, max_values = find_max_min(config, coins=config.coins_value, strategy=True)
    #print("all_returns_one_agent=",all_returns_one_agent)
    ret = []
    for idx in range(len(config.mult_fact)):
        print("\nmult=",config.mult_fact[idx])
        print("ret=",all_returns_one_agent[idx])
        #print("returns=", returns[idx])
        norm_returns = all_returns_one_agent[idx]/max_values[config.mult_fact[idx]]
        print("norm returns=",norm_returns)
        print("ret given strategy=",norm_returns[my_strategy[idx], opponent_strategy[idx]])
        ret.append(norm_returns[my_strategy[idx], opponent_strategy[idx]])
    avg_return = np.mean(ret)
    print("avg_return=",avg_return)

def find_max_min(config, coins, strategy=False):
    n_agents = 2 #config.n_agents
    multipliers = config.mult_fact
    max_values = {}
    min_values = {}
    coins_per_agent = np.array([coins for i in range(n_agents)])

    possible_actions = ["C", "D"]
    possible_scenarios = [''.join(i) for i in itertools.product(possible_actions, repeat = n_agents)]
    all_returns = []
    all_returns_one_agent = np.zeros((len(multipliers), 2, 2))

    i = 0
    for multiplier in multipliers:
        #print("\nMultiplier=", multiplier)
        possible_actions = ["C", "D"]
        possible_scenarios = [''.join(i) for i in itertools.product(possible_actions, repeat = n_agents)]
        returns = np.zeros((len(possible_scenarios), n_agents)) # TUTTI I POSSIBILI RITORNI CHE UN AGENTE PUO OTTENERE, PER OGNI AGENTE
        #print("possible_scen=", possible_scenarios)
        
        #scenarios_returns = {}
        for idx_scenario, scenario in enumerate(possible_scenarios):
            #print("scenario=", scenario, "scenario[0]=",scenario[0])
            y = 0
            if scenario[1] == 'C':
                y = 1
            common_pot = np.sum([coins_per_agent[i] for i in range(n_agents) if scenario[i] == "C"])

            for ag_idx in range(n_agents):
                if (scenario[ag_idx] == 'C'):
                    returns[idx_scenario, ag_idx] = common_pot/n_agents*multiplier
                    if (ag_idx == 0):
                        #print("x=", 1, "y=", y)
                        all_returns_one_agent[i, 1, y] = returns[idx_scenario, ag_idx]
                else: 
                    returns[idx_scenario, ag_idx] = common_pot/n_agents*multiplier + coins_per_agent[ag_idx]
                    if (ag_idx == 0):
                        #print("x=", 0, "y=", y)
                        all_returns_one_agent[i, 0, y] = returns[idx_scenario, ag_idx]
            #print("scenario=", scenario, "common_pot=", common_pot, "return=", returns[idx_scenario])

        max_values[multiplier] = np.amax(returns)
        min_values[multiplier] = np.amin(returns)
        print(" max_values[", multiplier, "]=",  max_values[multiplier])
        print(" min_values[", multiplier, "]=",  min_values[multiplier])
        #normalized = returns/max_values[multiplier]
        
        #print("normalized=", returns/max_values[multiplier]) #(returns-min_values[multiplier])/(max_values[multiplier] - min_values[multiplier]))
        all_returns.append(returns)
        i += 1

    if (strategy == True):
        return all_returns_one_agent, all_returns, max_values
    return max_values