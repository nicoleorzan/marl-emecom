import torch
import itertools
import numpy as np


class SocialNorm():

    def __init__(self, agents):
        self.agents = agents
        self.n_agents = len(self.agents)
        self.reset_saved_actions()

    def reset_saved_actions(self):
        self.saved_actions = {}

    def save_actions(self, act, active_agents_idxs):
        for ag_idx in active_agents_idxs:
            if (ag_idx not in self.saved_actions.keys()):
                self.saved_actions[ag_idx] = []
            else:
                self.saved_actions[ag_idx].append(act["agent_"+str(ag_idx)])

    def update_reputation(self, active_agents_idxs):
        for ag_idx in active_agents_idxs:
            #print("saved act=", self.saved_actions[ag_idx])
            #print("agent_"+str(ag_idx))
            #print(self.agents["agent_"+str(ag_idx)])
            #print("rep=",self.agents["agent_"+str(ag_idx)].reputation)
            self.agents["agent_"+str(ag_idx)].reputation = np.mean(self.saved_actions[ag_idx])
            #print("rep dopo",self.agents["agent_"+str(ag_idx)].reputation)
        self.reset_saved_actions()
        

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
        active_agents["agent_"+str(active_agents_idxs[0])].digest_input((observations["agent_"+str(active_agents_idxs[0])], active_agents["agent_"+str(active_agents_idxs[1])].reputation))
        active_agents["agent_"+str(active_agents_idxs[1])].digest_input((observations["agent_"+str(active_agents_idxs[1])], active_agents["agent_"+str(active_agents_idxs[0])].reputation))

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

def best_strategy_reward(config):
    my_strategy = [0, 1, 1, 1]
    opponent_strategy = [0, 1, 1, 1]
    all_returns_one_agent, returns, max_values = find_max_min(config, coins=4, strategy=True)
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
    n_agents = config.n_agents
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
        #print(" max_values[", multiplier, "]=",  max_values[multiplier])
        #print(" min_values[", multiplier, "]=",  min_values[multiplier])
        normalized = returns/max_values[multiplier]
        
        #print("normalized=", returns/max_values[multiplier]) #(returns-min_values[multiplier])/(max_values[multiplier] - min_values[multiplier]))
        all_returns.append(returns)
        i += 1

    if (strategy == True):
        return all_returns_one_agent, all_returns, max_values
    return max_values

def apply_norm(active_agents, active_agents_idxs, actions, f):
    reputation_rewards = {}
    addition = {}
    for idx in active_agents_idxs:
        agent = active_agents["agent_"+str(idx)]
        if (agent.is_dummy == False):
            #print("agent=", agent.idx)
            old_reputation = active_agents["agent_"+str(idx)].reputation
            #print("old rep=", old_reputation)
            other = list(set(active_agents_idxs) - set([idx]))[0]
            change_reputation(f, agent, actions["agent_"+str(idx)], active_agents["agent_"+str(other)].reputation, addition)
            #print("new one=", agent.reputation)
            reputation_rewards["agent_"+str(idx)] = active_agents["agent_"+str(idx)].reputation - old_reputation
    return reputation_rewards, addition
   
def apply_norm2(active_agents, active_agents_idxs, actions, f):
    reputation_rewards = {}
    addition = {}
    #for idx in active_agents_idxs:
    agent = active_agents["agent_0"]
    #print("agent=", agent.idx)
    old_reputation = active_agents["agent_0"].reputation
    #print("old rep=", old_reputation)
    print("actions=", actions)
    avg_val = np.mean(actions)
    print("avg=", avg_val)
    other = list(set(active_agents_idxs) - set([0]))[0]
    agent.reputation = avg_val
    #change_reputation(f, agent, actions["agent_0"], active_agents["agent_"+str(other)].reputation, addition)
    #print("new one=", agent.reputation)
    reputation_rewards["agent_0"] = active_agents["agent_0"].reputation - old_reputation
    return reputation_rewards, addition
   
def change_reputation(f, agent, action, opponent_reputation, addition):
    if ( f > 1 and f < 2. ):
        if (action == 1):
            if (opponent_reputation == 1):
                agent.reputation = min(agent.reputation + 0.1, 1.)
            elif (opponent_reputation == 0):
                agent.reputation = max(agent.reputation - 0.4, 0.)
        
        elif (action == 0):
            if (opponent_reputation == 1):
                agent.reputation = max(agent.reputation - 0.4, 0.)
            elif (opponent_reputation == 0):
                agent.reputation = min(agent.reputation + 0.1, 1.)

def binary_change_reputation(f, agent, action, opponent_reputation, addition):
    if ( f > 1. ):
        if (action == 1):
            if (opponent_reputation == 1):
                agent.reputation = 1.
            elif (opponent_reputation == 0):
                agent.reputation = 0.
        
        elif (action == 0):
            if (opponent_reputation == 1):
                agent.reputation = 0.
            elif (opponent_reputation == 0):
                agent.reputation = 1.

def change_reputation_f_aware(f, agent, action, addition):
    # if the game is purely competitive (f<1), I do not encourage anyone to defect or cooperate. 
    # Reputation therefore reamins unchanged for every action agents take
    # if the game is cooperative or mixed motive (f>1), I want a metric that encourages cooperation
    #print("agent=", agent.idx)
    #print("reputation before=", agent.reputation)
    if (f > 1):
        if (action == 0):
            agent.reputation = max(agent.reputation-0.5, 0.)
        if (action == 1):
            agent.reputation = min(agent.reputation+0.2, 1.)

        if (agent.reputation == 1.):
            addition["agent_"+str(agent.idx)] = 0
        else: 
            addition["agent_"+str(agent.idx)] = 0.2
            
    else: 
        addition["agent_"+str(agent.idx)] = 0
    #print("reputation after=", agent.reputation)