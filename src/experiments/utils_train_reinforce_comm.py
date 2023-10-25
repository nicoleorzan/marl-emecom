import torch

def eval1(config, parallel_env, agents, m, device, _print=False):
    observations = parallel_env.reset(m)

    if (_print == True):
        print("\n Eval ===> Mult factor=", m)
        print("obs=", observations)

    message = None
    done = False
    while not done:

        messages = {}
        actions = {}
        mex_distrib = {}
        act_distrib = {}
        [agents[agent].set_observation(observations[agent], _eval=True) for agent in parallel_env.agents]

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
            #print("actions=", actions)
            act_distrib[agent] = agents[agent].get_action_distribution()
                
        _, rewards_eval, _, _ = parallel_env.step(actions)

        if (_print == True):
            print("message=", message)
            print("actions=", actions)
            print("distrib=", act_distrib)
        observations, _, done, _ = parallel_env.step(actions)

    return actions, mex_distrib, act_distrib, rewards_eval 


def eval(config, parallel_env, agents, m, device, _print=False):
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