from src.environments import pgg_v0
from PPOcomm import PPOcomm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
import seaborn as sns
import os
from utils import plot_hist_returns, plot_train_returns, cooperativity_plot, evaluation, plot_avg_on_experiments

hyperparameter_defaults = dict(
    n_experiments = 50,
    episodes_per_experiment = 600,
    eval_eps = 1000,
    update_timestep = 40, # update policy every n timesteps
    n_agents = 3,
    uncertainties = [0., 0., 0.],
    coins_per_agent = 4,
    mult_fact = [0, 0.1, 0.2, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    num_game_iterations = 5,
    comm = True,
    action_space = 2,
    obs_dim = 2,                 # we observe coins we have,  and multiplier factor with uncertainty
    mex_space = 2,               # vocabulary
    K_epochs = 40,               # update policy for K epochs
    eps_clip = 0.2,              # clip parameter for PPO
    gamma = 0.99,                # discount factor
    c1 = 0.5,
    c2 = -0.01,
    lr_actor = 0.001,       # learning rate for actor network
    lr_critic = 0.001,       # learning rate for critic network,
    plots = False
)

wandb.init(project="pgg_comm", entity="nicoleorzan", config=hyperparameter_defaults, mode="offline")
config = wandb.config

assert (config.n_agents == len(config.uncertainties))

if (any(config.uncertainties != 0.)):
    unc = "w_uncert"
else: 
    unc = "wOUT_uncert"

if hasattr(config.mult_fact, '__len__'):
    folder = str(config.n_agents)+"agents/"+"variating_m_"+str(config.num_game_iterations)+"iters_"+unc+"/comm/"
else: 
    folder = str(config.n_agents)+"agents/"+str(config.mult_fact)+"mult_"+str(config.num_game_iterations)+"iters_"+unc+"/comm/"

path = "images/pgg/"+folder
if not os.path.exists(path):
    os.makedirs(path)
    print("New directory is created!")

print_freq = 500     # print avg reward in the interval (in num timesteps)

def evaluate_episode(agents_dict, agent_to_idx):
    env = pgg_v0.env(n_agents=config.n_agents, coins_per_agent=config.coins_per_agent, num_iterations=config.num_game_iterations, \
        mult_fact=config.mult_fact, uncertainties=config.uncertainties)
    env.reset()
    i = 0
    ag_rets = np.zeros(len(agents_dict))

    mex_in = torch.zeros((config.mex_space*config.n_agents), dtype=torch.int64)
    mex_out_aggreg = torch.zeros((config.mex_space*config.n_agents)).long()

    for id_agent in env.agent_iter():
        idx = agent_to_idx[id_agent]
        acting_agent = agents_dict[id_agent]
        obs, reward, done, _ = env.last()
        obs = torch.FloatTensor(obs)
        if (i > config.n_agents-1):
            mex_in = mex_out_aggreg
        state = torch.cat((obs, mex_in), dim=0)

        if not done:
            act, mex_out = acting_agent.select_action(state)
            mex_out = F.one_hot(mex_out, num_classes=config.mex_space)[0]
            mex_out_aggreg[idx*config.mex_space:idx*config.mex_space+config.mex_space] = mex_out

        else:
            act = None
            mex_out = None

        env.step(act)
        ag_rets[idx] += reward

        i += 1
    env.close()
    return ag_rets



def train(config):

    env = pgg_v0.env(n_agents=config.n_agents, coins_per_agent=config.coins_per_agent, num_iterations=config.num_game_iterations, \
        mult_fact=config.mult_fact, uncertainties=config.uncertainties)

    n_agents = config.n_agents
    mex_space = config.mex_space

    all_returns = np.zeros((n_agents, config.n_experiments, config.episodes_per_experiment))
    all_cooperativeness = np.zeros((n_agents, config.n_experiments, config.episodes_per_experiment))

    for experiment in range(config.n_experiments):
        print("Experiment=", experiment)

        agents_dict = {}
        un_agents_dict = {}
        agent_to_idx = {}
        for idx in range(config.n_agents):
            agents_dict['agent_'+str(idx)] = PPOcomm(config.obs_dim, config.action_space, config.mex_space, config.n_agents, config.lr_actor, config.lr_critic,  \
            config.gamma, config.K_epochs, config.eps_clip, config.c1, config.c2)
            un_agents_dict['agent_'+str(idx)] = PPOcomm(config.obs_dim, config.action_space, config.mex_space, config.n_agents, config.lr_actor, config.lr_critic,  \
            config.gamma, config.K_epochs, config.eps_clip, config.c1, config.c2)
            agent_to_idx['agent_'+str(idx)] = idx

        if (config.plots == True):
            print("\nEVALUATION BEFORE LEARNING")
            rews_before = evaluation(un_agents_dict, config.eval_eps, agent_to_idx)

        #### TRAINING LOOP
        time_step = 0
        i_episode = 0

        print_running_reward = np.zeros(config.n_agents)
        print_running_episodes = np.zeros(config.n_agents)

        for _ in range(config.episodes_per_experiment):
            env.reset()

            current_ep_reward = np.zeros(n_agents)
            i_internal_loop = 0
            for ag_idx in range(n_agents):
                agents_dict['agent_'+str(ag_idx)].tmp_return = 0
                agents_dict['agent_'+str(ag_idx)].tmp_actions = []

            mex_in = torch.zeros((mex_space*n_agents), dtype=torch.int64)
            mex_out_aggreg = torch.zeros((mex_space*n_agents)).long()
            #print("mex out aggr=", mex_out_aggreg)

            for id_agent in env.agent_iter():
                idx = agent_to_idx[id_agent]
                acting_agent = agents_dict[id_agent]
                
                obs, rew, done, _ = env.last()
                obs = torch.FloatTensor(obs)

                if (i_internal_loop > n_agents-1):
                    mex_in = mex_out_aggreg
                #print("mex_in =", mex_in)
                state = torch.cat((obs, mex_in), dim=0)
                
                if not done:
                    act, mex_out = acting_agent.select_action(state)
                    mex_out = F.one_hot(mex_out, num_classes=mex_space)[0]
                    mex_out_aggreg[idx*mex_space:idx*mex_space+mex_space] = mex_out
                else:
                    act = None
                    mex_out = None
                #print("act=", act, "mex_out=", mex_out)
                env.step(act)

                if (i_internal_loop > n_agents-1):
                    acting_agent.buffer.rewards.append(rew)
                    acting_agent.buffer.is_terminals.append(done)
                    acting_agent.tmp_return += rew
                    if (act is not None):
                        acting_agent.tmp_actions.append(act)

                time_step += 1

                if rew != None:
                    current_ep_reward[idx] += rew

                # break; if the episode is over
                if (done):  
                    acting_agent.train_returns.append(acting_agent.tmp_return)
                    acting_agent.cooperativeness.append(np.mean(acting_agent.tmp_actions))
                    if (idx == n_agents-1):
                        break

                i_internal_loop += 1
            
            if time_step % print_freq == 0:
                print_avg_reward = np.zeros(n_agents)
                for k in range(n_agents):
                    print_avg_reward[k] = print_running_reward[k] / print_running_episodes[k]
                    print_avg_reward[k] = round(print_avg_reward[k], 2)

                print("Episode : {} \t\t Timestep : {} \t\t ".format(i_episode, time_step))
                print("Average and Episodic Reward:")
                for i_print in range(n_agents):
                    print("Average rew agent",str(i_print),"=", print_avg_reward[i_print], "episodic reward=", agents_dict['agent_'+str(i_print)].buffer.rewards[-1])

                for i in range(n_agents):
                    print_running_reward[i] = 0
                    print_running_episodes[i] = 0

            # update PPO agent
            if time_step % config.update_timestep == 0:
                for ag_idx in range(n_agents):
                    agents_dict['agent_'+str(ag_idx)].update()
            
            for i in range(n_agents):
                print_running_reward[i] += current_ep_reward[i]
                print_running_episodes[i] += 1

            if (i_episode%10 == 0):
                for ag_idx in range(config.n_agents):
                    wandb.log({"agent"+str(ag_idx)+"_return": agents_dict['agent_'+str(ag_idx)].tmp_return}, step=i_episode)
                    wandb.log({"agent"+str(ag_idx)+"_coop_level": np.mean(agents_dict['agent_'+str(ag_idx)].tmp_actions)}, step=i_episode)
                wandb.log({"episode": i_episode}, step=i_episode)

            i_episode += 1


        for ag_idx in range(n_agents):
            all_returns[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].train_returns
            all_cooperativeness[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].cooperativeness

        # PLOTS
        if (config.plots == True):
            ### PLOT TRAIN RETURNS
            plot_train_returns(config, agents_dict, path, "train_returns_pgg_comm")

            # COOPERATIVITY PERCENTAGE PLOT
            cooperativity_plot(config, agents_dict, path, "train_cooperativeness_comm")

            ### EVALUATION
            print("\n\nEVALUATION AFTER LEARNING")

            rews_after = evaluation(agents_dict, config.eval_eps, agent_to_idx)
            print("average rews before", np.average(rews_before[0]), np.average(rews_before[1]))
            print("average rews ater", np.average(rews_after[0]), np.average(rews_after[1]))

            #plot_hist_returns(rews_before, rews_after)

            # Print policy
            pox_coins = np.linspace(0, int(max(rews_after[0])), int(max(rews_after[0])))
            heat = np.zeros((n_agents, len(pox_coins), len(config.mult_fact)))
            mex_in = torch.zeros((config.mex_space*config.n_agents), dtype=torch.int64)
            for ag in range(config.n_agents):
                for ii in range(len(pox_coins)):
                    for jj in range(len(config.mult_fact)):
                        obs = np.array((pox_coins[ii], n_agents, config.mult_fact[jj]))
                        obs = torch.FloatTensor(obs)
                        state = torch.cat((obs, mex_in), dim=0)
                        act, _ = agents_dict['agent_'+str(ag_idx)].select_action(state)
                        heat[ag, ii,jj] = act
                
            fig, ax = plt.subplots(1, n_agents, figsize=(n_agents*4, 4))
            for ag in range(config.n_agents):
                sns.heatmap(heat[ag], ax=ax[ag])
            print("Saving heatmap..")
            plt.savefig(path+"heatmap_comm.png")



    plot_avg_on_experiments(config, all_returns, all_cooperativeness, path, "comm")


if __name__ == "__main__":
    train(config)