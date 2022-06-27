from src.environments import pgg_v1
from algos.PPO import PPO
from nets.ActorCritic import ActorCriticDiscrete
import numpy as np
import torch
import wandb
import json
import pandas as pd
import os
from analysis.utils import plot_train_returns, cooperativity_plot, plots_experiments

hyperparameter_defaults = dict(
    n_experiments = 1,
    threshold = 2,
    episodes_per_experiment = 1000,
    eval_eps = 100,
    update_timestep = 40, # update policy every n timesteps
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
    save_interval = 1,
    print_freq = 100
)

wandb.init(project="pgg_v1", entity="nicoleorzan", config=hyperparameter_defaults, mode="offline")
config = wandb.config

if (any(config.uncertainties) != 0.):
    unc = "w_uncert"
else: 
    unc = "wOUT_uncert"

folder = str(config.n_agents)+"agents/"+str(config.num_game_iterations)+"iters_"+unc+"/"

path = "data/pgg_v1/"+folder
if not os.path.exists(path):
    os.makedirs(path)
    print("New directory is created!")

print("path=", path)

with open(path+'params.json', 'w') as fp:
    json.dump(hyperparameter_defaults, fp)


def evaluate_episode(agents_dict, agent_to_idx):
    env = pgg_v1.env(n_agents=config.n_agents, threshold = config.threshold, \
        num_iterations=config.num_game_iterations, uncertainties=config.uncertainties)
    env.reset()
    i = 0
    ag_rets = np.zeros(len(agents_dict))

    for id_agent in env.agent_iter():
        idx = agent_to_idx[id_agent]
        acting_agent = agents_dict[id_agent]
        obs, reward, done, _ = env.last()
        act = acting_agent.select_action(obs) if not done else None
        env.step(act)
        ag_rets[idx] += reward
        i += 1

        if (done and idx == config.n_agents-1):  
            break

    env.close()
    return ag_rets


def train(config):

    n_agents = config.n_agents

    env = pgg_v1.env(n_agents=config.n_agents, threshold = config.threshold, \
        num_iterations=config.num_game_iterations, uncertainties=config.uncertainties)

    all_returns = np.zeros((n_agents, config.n_experiments, config.episodes_per_experiment))
    all_coop = np.zeros((n_agents, config.n_experiments, config.episodes_per_experiment))

    if (config.save_data == True):
        df = pd.DataFrame(columns=['experiment', 'episode', 'ret_ag0', 'ret_ag1', 'ret_ag2', 'coop_ag0', 'coop_ag1', 'coop_ag2'])

    for experiment in range(config.n_experiments):

        print("Experiment ", experiment)

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
            #print("ep_in", ep_in)

            env.reset()
            i_internal_loop = 0
                
            for ag_idx, agent in agents_dict.items():
                agent.tmp_return = 0
                agent.tmp_actions = []

            for id_agent in env.agent_iter():
                #print("agent=", id_agent)
                idx = agent_to_idx[id_agent]
                acting_agent = agents_dict[id_agent]
                
                obs, rew, done, _ = env.last()
                #print(obs, rew, done)
                #print(obs.shape)
                act = acting_agent.select_action(obs) if not done else None
                #print("act=", act)
                env.step(act)

                if (i_internal_loop > config.n_agents-1):
                    acting_agent.buffer.rewards.append(rew)
                    acting_agent.buffer.is_terminals.append(done)
                    acting_agent.tmp_return += rew
                    #print("acting_agent.buffer.rewards=",acting_agent.buffer.rewards)
                if (act is not None):
                    acting_agent.tmp_actions.append(act)

                # break; if the episode is over
                if (done):
                    acting_agent.train_returns.append(acting_agent.tmp_return)
                    acting_agent.coop.append(np.mean(acting_agent.tmp_actions))
                    #print("acting_agent.tmp_return",acting_agent.tmp_return)
                    if (idx == config.n_agents-1):
                        break

                i_internal_loop += 1

            if (ep_in+1) % config.print_freq == 0:
                print("Experiment : {} \t Episode : {} \t Iters: {} ".format(experiment, \
                    ep_in, config.num_game_iterations))
                print("Episodic Reward:")
                for ag_idx, agent in agents_dict.items():
                    print("Agent=", ag_idx, "rew=", agent.buffer.rewards[-1])

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

        if (ep_in%config.save_interval == 0):
            for ag_idx in range(n_agents):
                all_returns[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].train_returns
                all_coop[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].coop

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
    plots_experiments(config, all_returns, all_coop, path, "")


if __name__ == "__main__":
    train(config)
