import numpy as np
import torch
import pandas as pd 
import wandb

def eval(config, parallel_env, agents_dict, m, device, _print=False):
    observations = parallel_env.reset(m)

    if (_print == True):
        print("\n Eval ===> Mult factor=", m)
        print("obs=", observations)

    done = False
    while not done:

        if (config.random_baseline):
            messages = {agent: agents_dict[agent].random_messages(observations[agent]) for agent in parallel_env.agents}
            mex_distrib = 0
        else:
            messages = {agent: agents_dict[agent].select_message(observations[agent], True) for agent in parallel_env.agents}
            mex_distrib = {agent: agents_dict[agent].get_message_distribution(observations[agent]) for agent in parallel_env.agents}
        message = torch.stack([v for _, v in messages.items()]).view(-1).to(device)
        actions = {agent: agents_dict[agent].select_action(observations[agent], message, True) for agent in parallel_env.agents}
        acts_distrib = {agent: agents_dict[agent].get_action_distribution(observations[agent], message) for agent in parallel_env.agents}
        
        _, rewards_eval, _, _ = parallel_env.step(actions)

        if (_print == True):
            print("message=", message)
            print("actions=", actions)
        observations, _, done, _ = parallel_env.step(actions)

    return actions, mex_distrib, acts_distrib, rewards_eval 