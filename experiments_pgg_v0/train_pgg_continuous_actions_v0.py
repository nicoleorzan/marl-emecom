from src.environments import pgg_v0
from algos.PPO import PPO
from nets.ActorCritic import ActorCriticContinuous
import numpy as np
import wandb
import json
import os
import torch
from analysis.utils import plot_train_returns, cooperativity_plot, plots_experiments


hyperparameter_defaults = dict(
    n_experiments = 10,
    episodes_per_experiment = 3000,
    eval_eps = 100,
    update_timestep = 40, # update policy every n timesteps
    n_agents = 3,
    uncertainties = [0., 0., 0.],
    coins_per_agent = 4,
    mult_fact = [1.3,1.3], # list giivn min and max value of mult factor
    num_game_iterations = 1,
    obs_dim = 2,                 # we observe coins we have, and multiplier factor with uncertainty
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

wandb.init(project="pgg_cont_actions_v0", entity="nicoleorzan", config=hyperparameter_defaults, mode="offline")
config = wandb.config

if (any(config.uncertainties) != 0.):
    unc = "w_uncert"
else: 
    unc = "wOUT_uncert"

if (config.mult_fact[0] != config.mult_fact[1]):
    folder = str(config.n_agents)+"agents/"+"variating_m_"+str(config.num_game_iterations)+"iters_"+unc+"/normal/"
else: 
    folder = str(config.n_agents)+"agents/"+str(config.mult_fact[0])+"mult_"+str(config.num_game_iterations)+"iters_"+unc+"/normal/"

path = "data/pgg_v0/"+folder
if not os.path.exists(path):
    os.makedirs(path)
    print("New directory is created!")

print("path=", path)

with open(path+'params.json', 'w') as fp:
    json.dump(hyperparameter_defaults, fp)


def evaluate_episode(agents_dict, agent_to_idx):
    env = pgg_v0.env(n_agents=config.n_agents, coins_per_agent=config.coins_per_agent, \
        num_iterations=config.num_game_iterations, mult_fact=config.mult_fact, \
        uncertainties=config.uncertainties, fraction=True)
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

    env = pgg_v0.env(n_agents=config.n_agents, coins_per_agent=config.coins_per_agent, \
        num_iterations=config.num_game_iterations, mult_fact=config.mult_fact, \
        uncertainties=config.uncertainties, fraction=True)

    all_returns = np.zeros((n_agents, config.n_experiments, config.episodes_per_experiment))
    all_coop = np.zeros((n_agents, config.n_experiments, config.episodes_per_experiment))

    for experiment in range(config.n_experiments):

        print("Experiment ", experiment)

        agents_dict = {}
        agent_to_idx = {}
        for idx in range(config.n_agents):
            model = ActorCriticContinuous(config.obs_dim, config.action_space)
            optimizer = torch.optim.Adam([{'params': model.policy.parameters(), 'lr': config.lr_actor},
                    {'params': model.critic.parameters(), 'lr': config.lr_critic},
                    {'params': model.mean.parameters(), 'lr': config.lr_actor},
                    {'params': model.var.parameters(), 'lr': config.lr_actor}, ])

            agents_dict['agent_'+str(idx)] = PPO(model, optimizer, config.lr_actor, config.lr_critic,  \
            config.gamma, config.K_epochs, config.eps_clip, config.c1, config.c2)
            agent_to_idx['agent_'+str(idx)] = idx

        #### TRAINING LOOP
        for ep_in in range(config.episodes_per_experiment):

            env.reset()
            i_internal_loop = 0
                
            for ag_idx, agent in agents_dict.items():
                agent.tmp_return = 0
                agent.tmp_actions = []

            for id_agent in env.agent_iter():
                idx = agent_to_idx[id_agent]
                acting_agent = agents_dict[id_agent]
                
                obs, rew, done, _ = env.last()
                act = acting_agent.select_action(obs) if not done else None
                env.step(act)

                if (i_internal_loop > config.n_agents-1):
                    acting_agent.buffer.rewards.append(rew)
                    acting_agent.buffer.is_terminals.append(done)
                    acting_agent.tmp_return += rew
                if (act is not None):
                    acting_agent.tmp_actions.append(act)

                # break; if the episode is over
                if (done):
                    acting_agent.train_returns.append(acting_agent.tmp_return)
                    acting_agent.coop.append(np.mean(acting_agent.tmp_actions))
                    if (idx == config.n_agents-1):
                        break

                i_internal_loop += 1

            if (ep_in+1) % config.print_freq == 0:
                print("Experiment : {} \t Episode : {} \t Mult factor : {} \t Iters: {} ".format(experiment, \
                    ep_in, env.env.current_multiplier, config.num_game_iterations))
                print("Episodic Reward:")
                for ag_idx, agent in agents_dict.items():
                    print("Agent=", ag_idx, "rew=", agent.buffer.rewards[-1])

            # update PPOnormal agents
            if ep_in != 0 and ep_in % config.update_timestep == 0:
                for ag_idx, agent in agents_dict.items():
                    agent.update()

            if (config.n_experiments == 1 and ep_in%10 == 0):
                for ag_idx, agent in agents_dict.items():
                    wandb.log({ag_idx+"_return": agent.tmp_return}, step=ep_in)
                    wandb.log({ag_idx+"_coop_level": np.mean(agent.tmp_actions)}, step=ep_in)
                wandb.log({"episode": ep_in}, step=ep_in)

        if (ep_in%config.save_interval == 0):
            for ag_idx in range(n_agents):
                all_returns[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].train_returns
                all_coop[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].coop


        if (config.plots == True):
            ### PLOT TRAIN RETURNS
            plot_train_returns(config, agents_dict, path, "train_returns_pgg")

            # COOPERATIVITY PERCENTAGE PLOT
            cooperativity_plot(config, agents_dict, path, "train_cooperativeness")

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
    print("Stuff saved in", folder)
