import torch
import numpy as np
import torch.nn.functional as F

def partner_selection(config, parallel_env, agents):

    reputations = torch.from_numpy(np.fromiter((agents[agent].reputation for agent in parallel_env.agents), float)).float()
    agent_num = np.random.randint(0,len(parallel_env.agents)-1)
    agent_name = "agent_"+str(agent_num)
    partner_num = agents[agent_name].select_partner(reputations)
    partner_name = "agent_"+str(partner_num)
    agents[partner_name].partner_id = F.one_hot(torch.Tensor([agent_num]).to(torch.int64), num_classes=config.n_agents)[0]
    # print("agents[partner_name].partner_id=",  agents[partner_name].partner_id)
    playing_agents = {agent_name: agents[agent_name], partner_name: agents[partner_name]}
    
    return playing_agents


def eval_ps(config, parallel_env, agents, m, device, _print=False):
    observations = parallel_env.reset(m)

    if (_print == True):
        print("\n Eval ===> Mult factor=", m)
        print("obs=", observations)

    message = None
    done = False
    while not done:

        messages = {}; actions = {}
        mex_distrib = {}; act_distrib = {}
        [agents[agent].set_state(state=observations[agent], _eval=True) for agent in parallel_env.agents]

        playing_agents = partner_selection(config, parallel_env, agents)

        # acting
        for ag_idx, ag in playing_agents.items():
            actions[ag_idx] = ag.select_action(_eval=True)
            act_distrib[ag_idx] = ag.get_action_distribution()
        #print("actions=", actions)
        # speaking
        #for agent in parallel_env.agents:
        #    if (agents[agent].is_communicating):
        #        messages[agent] = agents[agent].select_message(_eval=True)
        #        mex_distrib[agent] = agents[agent].get_message_distribution()
        # listening
        #if (config.communicating_agents.count(1) != 0):
        #    message = torch.stack([v for _, v in messages.items()]).view(-1).to(device)
        #    [agents[agent].get_message(message) for agent in parallel_env.agents if (agents[agent].is_listening)]

        # acting
        #for agent in parallel_env.agents:
        #    actions[agent] = agents[agent].select_action(_eval=True)
        #    act_distrib[agent] = agents[agent].get_action_distribution()
                
        _, rewards_eval, done, _ = parallel_env.step(actions)

        if (_print == True):
            print("message=", message)
            print("actions=", actions)
            print("distrib=", act_distrib)

    return actions, mex_distrib, act_distrib, rewards_eval 

def eval(config, parallel_env, agents, m, device, _print=False):
    observations = parallel_env.reset(m)

    if (_print == True):
        print("\n Eval ===> Mult factor=", m)
        print("obs=", observations)

    message = None
    done = False
    while not done:

        messages = {}; actions = {}
        mex_distrib = {}; act_distrib = {}
        [agents[agent].set_state(state=observations[agent], _eval=True) for agent in parallel_env.agents]

        # speaking
        for agent in parallel_env.agents:
            if (agents[agent].is_communicating):
                messages[agent] = agents[agent].select_message(_eval=True)
                mex_distrib[agent] = agents[agent].get_message_distribution()
        # listening
        if (config.communicating_agents.count(1) != 0):
            message = torch.stack([v for _, v in messages.items()]).view(-1).to(device)
            [agents[agent].get_message(message) for agent in parallel_env.agents if (agents[agent].is_listening)]

        # acting
        for agent in parallel_env.agents:
            actions[agent] = agents[agent].select_action(_eval=True)
            act_distrib[agent] = agents[agent].get_action_distribution()
                
        _, rewards_eval, _, _ = parallel_env.step(actions)

        if (_print == True):
            print("message=", message)
            print("actions=", actions)
            print("distrib=", act_distrib)
        observations, _, done, _ = parallel_env.step(actions)

    return actions, mex_distrib, act_distrib, rewards_eval 