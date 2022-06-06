from src.environments import pgg_v0
from PPOcomm import PPOcomm
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import seaborn as sns
import os
from utils import plot_hist_returns, plot_train_returns, cooperativity_plot, evaluation, plot_avg_on_experiments

hyperparameter_defaults = dict(
    n_experiments = 10,
    episodes_per_experiment = 100,
    eval_eps = 1000,
    update_timestep = 40, # update policy every n timesteps
    n_agents = 3,
    uncertainties = [0., 0., 0.],
    coins_per_agent = 4,
    mult_fact = 4, #[0, 0.1, 0.2, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    num_game_iterations = 1,
    comm = True,
    action_space = 2,
    obs_dim = 2,                 # we observe coins we have,  and multiplier factor with uncertainty
    mex_space = 2,               # vocabulary
    K_epochs = 40,               # update policy for K epochs
    eps_clip = 0.2,              # clip parameter for PPO
    gamma = 0.99,                # discount factor
    c1 = 0.5,
    c2 = -0.01,
    lr_actor = 0.001,        # learning rate for actor network
    lr_critic = 0.001,       # learning rate for critic network,
    plots = False,
    save_models = False
)

mode = "offline"

if (hyperparameter_defaults['n_experiments'] == 1):
    mode = None
wandb.init(project="pgg_comm", entity="nicoleorzan", config=hyperparameter_defaults, mode=mode)
config = wandb.config

assert (config.n_agents == len(config.uncertainties))


if (any(config.uncertainties) != 0.):
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

print_freq = 400     # print avg reward in the interval (in num timesteps)

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
    all_coop = np.zeros((n_agents, config.n_experiments, config.episodes_per_experiment))

    for experiment in range(config.n_experiments):

        print("Experiment=", experiment)

        agents_dict = {}
        agent_to_idx = {}
        for idx in range(config.n_agents):
            agents_dict['agent_'+str(idx)] = PPOcomm(config.obs_dim, config.action_space, config.mex_space, config.n_agents, config.lr_actor, config.lr_critic,  \
            config.gamma, config.K_epochs, config.eps_clip, config.c1, config.c2)
            agent_to_idx['agent_'+str(idx)] = idx

        #### TRAINING LOOP
        for ep_in in range(config.episodes_per_experiment):

            env.reset()
            i_internal_loop = 0
            for ag_idx, agent in agents_dict.items():
                agent.tmp_return = 0
                agent.tmp_actions = []

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

                # break; if the episode is over
                if (done):  
                    acting_agent.train_returns.append(acting_agent.tmp_return)
                    acting_agent.cooperativeness.append(np.mean(acting_agent.tmp_actions))
                    if (idx == n_agents-1):
                        break

                i_internal_loop += 1
            
            if (ep_in+1) % print_freq == 0:
                print("Experiment : {} \t Episode : {} \t Mult factor : {} ".format(experiment, ep_in, env.env.env.current_multiplier))
                print("Episodic Reward:")
                for ag_idx, agent in agents_dict.items():
                    print("Agent=", ag_idx, "rew=", agent.buffer.rewards[-1])

            # update PPO agent
            if ep_in != 0 and ep_in % config.update_timestep == 0:
                for ag_idx, agent in agents_dict.items():
                    agent.update()

            if (config.n_experiments == 1 and ep_in%10 == 0):
                for ag_idx, agent in agents_dict.items():#range(config.n_agents):
                    wandb.log({"agent"+str(ag_idx)+"_return": agent.tmp_return}, step=ep_in)
                    wandb.log({"agent"+str(ag_idx)+"_coop_level": np.mean(agent.tmp_actions)}, step=ep_in)
                agents_coop = np.sum([np.mean(agent.tmp_actions) for _, agent in agents_dict.items()])
                wandb.log({"agents_coop_level": agents_coop}, step=ep_in)
                agents_coop = np.sum([np.mean(agent.tmp_return) for _, agent in agents_dict.items()])
                wandb.log({"agents_summed_returns": agents_coop}, step=ep_in)
                wandb.log({"episode": ep_in}, step=ep_in)


        for ag_idx in range(n_agents):
            all_returns[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].train_returns
            all_coop[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].cooperativeness

        # PLOTS
        if (config.plots == True):
            ### PLOT TRAIN RETURNS
            plot_train_returns(config, agents_dict, path, "train_returns_pgg_comm")

            # COOPERATIVITY PERCENTAGE PLOT
            cooperativity_plot(config, agents_dict, path, "train_coop_comm")

            ### EVALUATION
            print("\n\nEVALUATION AFTER LEARNING")

            rews_after = evaluation(agents_dict, config.eval_eps, agent_to_idx)
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


    if (config.n_experiments > 1):
        plot_avg_on_experiments(config, all_returns, all_coop, path, "comm")

    # save models
    if (config.save_models == True):
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy.state_dict(), "model_"+str(ag_idx))


if __name__ == "__main__":
    train(config)