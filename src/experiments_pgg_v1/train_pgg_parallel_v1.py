from src.environments import pgg_parallel_v1
from src.algos.PPO import PPO
from src.nets.ActorCritic import ActorCriticDiscrete
import numpy as np
import torch
import wandb
import json
import pandas as pd
import os
from src.analysis.utils import plot_train_returns, cooperativity_plot, plots_experiments

hyperparameter_defaults = dict(
    n_experiments = 3,
    threshold = 2,
    episodes_per_experiment = 1000,
    update_timestep = 40,        # update policy every n timesteps
    n_agents = 3,
    uncertainties = [0., 0., 0.],
    num_game_iterations = 1,
    obs_dim = 1,                 # we observe coins we have
    action_space = 2,
    K_epochs = 40,               # update policy for K epochs
    eps_clip = 0.2,              # clip parameter for PPO
    gamma = 0.99,                # discount factor
    c1 = 0.5,
    c2 = -0.01,
    lr_actor = 0.001,            # learning rate for actor network
    lr_critic = 0.001,           # learning rate for critic network
    comm = False,
    plots = False,
    save_models = False,
    save_data = True,
    save_interval = 50,
    print_freq = 100
)


wandb.init(project="pgg_v1_parallel", entity="nicoleorzan", config=hyperparameter_defaults, mode="offline")
config = wandb.config

if (any(config.uncertainties) != 0.):
    unc = "w_uncert"
else: 
    unc = "wOUT_uncert"

folder = str(config.n_agents)+"agents/"+str(config.num_game_iterations)+"iters_"+unc+"/parallel/"

path = "data/pgg_v1/"+folder
if not os.path.exists(path):
    os.makedirs(path)
    print("New directory is created!")

print("path=", path)

with open(path+'params.json', 'w') as fp:
    json.dump(hyperparameter_defaults, fp)

def train(config):

    n_agents = config.n_agents

    parallel_env = pgg_parallel_v1.parallel_env(n_agents=config.n_agents, threshold=config.threshold, \
        num_iterations=config.num_game_iterations, uncertainties=config.uncertainties)

    all_returns = np.zeros((n_agents, config.n_experiments, int(config.episodes_per_experiment/config.save_interval)))
    all_coop = np.zeros((n_agents, config.n_experiments, int(config.episodes_per_experiment/config.save_interval)))

    if (config.save_data == True):
        df = pd.DataFrame(columns=['experiment', 'episode', 'ret_ag0', 'ret_ag1', 'ret_ag2', 'coop_ag0', 'coop_ag1', 'coop_ag2'])

    for experiment in range(config.n_experiments):

        #print("\nExperiment ", experiment)

        agents_dict = {}
        agent_to_idx = {}
        for idx in range(config.n_agents):
            model = ActorCriticDiscrete(config.obs_dim, config.action_space)
            optimizer = torch.optim.Adam([{'params': model.actor.parameters(), 'lr': config.lr_actor},
                    {'params': model.critic.parameters(), 'lr': config.lr_critic} ])

            agents_dict['agent_'+str(idx)] = PPO(model, optimizer, config.lr_actor, config.lr_critic,  \
            config.gamma, config.K_epochs, config.eps_clip, config.c1, config.c2)
            agent_to_idx['agent_'+str(idx)] = idx

        #### TRAINING LOOP
        for ep_in in range(config.episodes_per_experiment):
            #print("\nEpisode=", ep_in)

            observations = parallel_env.reset()
            i_internal_loop = 0
                
            for ag_idx, agent in agents_dict.items():
                agent.tmp_return = 0
                agent.tmp_actions = []

            done = False
            while not done:

                actions = {agent: agents_dict[agent].select_action(observations[agent]) for agent in parallel_env.agents}
                observations, rewards, done, _ = parallel_env.step(actions)

                for ag_idx, agent in agents_dict.items():
                    
                    agent.buffer.rewards.append(rewards[ag_idx])
                    agent.buffer.is_terminals.append(done)
                    agent.tmp_return += rewards[ag_idx]
                    if (actions[ag_idx] is not None):
                        agent.tmp_actions.append(actions[ag_idx])
                    if done:
                        agent.train_returns.append(agent.tmp_return)
                        agent.coop.append(np.mean(agent.tmp_actions))

                # break; if the episode is over
                if done:
                    break

                i_internal_loop += 1

            if (ep_in+1) % config.print_freq == 0:
                print("Experiment : {} \t Episode : {} \t Iters: {} ".format(experiment, \
                    ep_in, config.num_game_iterations))
                print("Episodic Reward:")
                for ag_idx, agent in agents_dict.items():
                    print("Agent=", ag_idx, "action=", actions[ag_idx], "rew=", rewards[ag_idx])

            # update PPO agents
            if ep_in != 0 and ep_in % config.update_timestep == 0:
                for ag_idx, agent in agents_dict.items():
                    agent.update()

            if (config.n_experiments == 1 and ep_in%10 == 0):
                for ag_idx, agent in agents_dict.items():
                    wandb.log({ag_idx+"_return": agent.tmp_return}, step=ep_in)
                    wandb.log({ag_idx+"_coop_level": np.mean(agent.tmp_actions)}, step=ep_in)
                wandb.log({"episode": ep_in}, step=ep_in)

            if (config.save_data == True and ep_in%config.save_interval == 0):
                df = df.append({'experiment': experiment, 'episode': ep_in, \
                    "ret_ag0": agents_dict["agent_0"].tmp_return, \
                    "ret_ag1": agents_dict["agent_1"].tmp_return, \
                    "ret_ag2": agents_dict["agent_2"].tmp_return, \
                    "coop_ag0": np.mean(agents_dict["agent_0"].tmp_actions), \
                    "coop_ag1": np.mean(agents_dict["agent_1"].tmp_actions), \
                    "coop_ag2": np.mean(agents_dict["agent_2"].tmp_actions)}, ignore_index=True)

        for ag_idx in range(n_agents):
            all_returns[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].train_returns[0::config.save_interval]
            all_coop[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].coop[0::config.save_interval]

        if (config.plots == True):
            ### PLOT TRAIN RETURNS
            plot_train_returns(config, agents_dict, path, "train_returns_pgg")

            # COOPERATIVITY PERCENTAGE PLOT
            cooperativity_plot(config, agents_dict, path, "train_cooperativeness")

    if (config.save_data == True):
        df.to_csv(path+'data_no_comm.csv')
    
    # save models
    print("Saving models...")
    if (config.save_models == True):
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy.state_dict(), path+"model_"+str(ag_idx))

    #mean calculations
    #if (config.n_experiments > 1):
    print("PATH=", path)
    plots_experiments(config, all_returns, all_coop, path, "")


if __name__ == "__main__":
    train(config)
