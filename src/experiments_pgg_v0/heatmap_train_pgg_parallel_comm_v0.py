from src.environments import pgg_parallel_v0_bis
from src.algos.PPOcomm import PPOcomm
import numpy as np
import torch
import wandb
import json
import pandas as pd
import sys
import os
import src.analysis.utils as U
import matplotlib.pyplot as plt

hyperparameter_defaults = dict(
    n_experiments = 1,
    episodes_per_experiment = 100,
    update_timestep = 100,        # update policy every n timesteps
    n_agents = 3,
    unc = 1.0, #[0.1, 0.2, 0.5, 1.],
    coins_mean = 4,
    mult_factors = [0., 0.1, 0.2, 0.5, 0.7, 1., 2., 3., 6., 8., 10.],
    num_game_iterations = 1,
    obs_size = 2,                 # we observe coins we have, and multiplier factor with uncertainty
    action_size = 2,
    hidden_size = 50,
    K_epochs = 40,               # update policy for K epochs
    eps_clip = 0.2,              # clip parameter for PPO
    gamma = 0.99,                # discount factor
    c1 = -0.3,#-1,
    c2 = 0.1,# 0.01,
    c3 = -0.5,#-1,
    c4 = 0.001, #.01, # governa entropy comm
    lr_actor = 0.1,            # learning rate for actor network
    lr_critic = 0.001,           # learning rate for critic network
    lr_actor_comm = 0.05,            # learning rate for actor network
    lr_critic_comm = 0.001,           # learning rate for critic network
    decayRate = 0.9999,
    fraction = False,
    comm = True,
    plots = False,
    save_models = False,
    save_data = True,
    save_interval = 1,
    print_freq = 500,
    mex_size = 7,
    random_baseline = False,
    recurrent = False,
    wandb_mode ="offline"
)


wandb.init(project="pgg_v0_parallel_comm", entity="nicoleorzan", config=hyperparameter_defaults, mode=hyperparameter_defaults["wandb_mode"])
config = wandb.config

folder = str(config.n_agents)+"agents/heatmap/variating_m/comm/"

path = "data/pgg_v0/"+folder
if not os.path.exists(path):
    os.makedirs(path)
    print("New directory is created!")

print("path=", path)

with open(path+'params.json', 'w') as fp:
    json.dump(hyperparameter_defaults, fp)

def train(config):

    mut01 = []; mut10 = []; mut12 = []; mut21 = []; mut20 = []; mut02 = []
    mut0_avg = []; mut1_avg = []; mut2_avg = []
    sc0 = []; sc1 = []; sc2 = []
    h0 = []; h1 = []; h2  = []
    mult_factors = []

    parallel_env = pgg_parallel_v0_bis.parallel_env(n_agents=config.n_agents, num_iterations=config.num_game_iterations)

    if (config.save_data == True):
        df = pd.DataFrame(columns=['experiment', 'episode', 'mult_factor', 'uncertainty'] + \
            ["ret_ag"+str(i) for i in range(config.n_agents)] + \
            ["coop_ag"+str(i) for i in range(config.n_agents)])

    for experiment in range(len(config.mult_factors)):

        print("Mult_factor=", config.mult_factors[experiment])
        for times in range(config.n_experiments):
            #print("\nExperiment ", experiment)

            agents_dict = {}
            for idx in range(config.n_agents):
                agents_dict['agent_'+str(idx)] = PPOcomm(config)

            #### TRAINING LOOP
            avg_coop_time = []
            for ep_in in range(config.episodes_per_experiment):
                #print("\nEpisode=", ep_in)

                observations = parallel_env.reset('uniform', config.unc, config.mult_factors[experiment])
                    
                [agent.reset() for _, agent in agents_dict.items()]

                done = False
                while not done:

                    if (config.random_baseline):
                        messages = {agent: agents_dict[agent].random_messages(observations[agent]) for agent in parallel_env.agents}
                    else:
                        messages = {agent: agents_dict[agent].select_message(observations[agent]) for agent in parallel_env.agents}
                    message = torch.stack([v for _, v in messages.items()]).view(-1)
                    actions = {agent: agents_dict[agent].select_action(observations[agent], message) for agent in parallel_env.agents}
                    observations, rewards, done, _ = parallel_env.step(actions)

                    for ag_idx, agent in agents_dict.items():
                        
                        agent.buffer.rewards.append(rewards[ag_idx])
                        agent.buffer.is_terminals.append(done)
                        agent.tmp_return += rewards[ag_idx]
                        if (actions[ag_idx] is not None):
                            agent.tmp_actions.append(actions[ag_idx])
                            #agent.tmp_messages.append(messages[ag_idx])
                        if done:
                            agent.train_returns.append(agent.tmp_return)
                            agent.coop.append(np.mean(agent.tmp_actions))

                    # mut 01 is how much the messages of agent 1 influenced the actions of agent 0 in the last buffer (group of episodes on which I want to learn)
                    mut01.append(U.calc_mutinfo(agents_dict['agent_0'].buffer.actions, agents_dict['agent_1'].buffer.messages, config.action_size, config.mex_size))
                    mut10.append(U.calc_mutinfo(agents_dict['agent_1'].buffer.actions, agents_dict['agent_0'].buffer.messages, config.action_size, config.mex_size))
                    mut12.append(U.calc_mutinfo(agents_dict['agent_1'].buffer.actions, agents_dict['agent_2'].buffer.messages, config.action_size, config.mex_size))
                    mut21.append(U.calc_mutinfo(agents_dict['agent_2'].buffer.actions, agents_dict['agent_1'].buffer.messages, config.action_size, config.mex_size))
                    mut20.append(U.calc_mutinfo(agents_dict['agent_2'].buffer.actions, agents_dict['agent_0'].buffer.messages, config.action_size, config.mex_size))
                    mut02.append(U.calc_mutinfo(agents_dict['agent_0'].buffer.actions, agents_dict['agent_2'].buffer.messages, config.action_size, config.mex_size))
                    #print("sc0=", U.calc_mutinfo(agents_dict['agent_0'].buffer.actions, agents_dict['agent_0'].buffer.messages, config.action_size, config.mex_size), "mut=", np.mean([mut01[-1], mut02[-1]]))
                    sc0.append(U.calc_mutinfo(agents_dict['agent_0'].buffer.actions, agents_dict['agent_0'].buffer.messages, config.action_size, config.mex_size))
                    sc1.append(U.calc_mutinfo(agents_dict['agent_1'].buffer.actions, agents_dict['agent_1'].buffer.messages, config.action_size, config.mex_size))
                    sc2.append(U.calc_mutinfo(agents_dict['agent_2'].buffer.actions, agents_dict['agent_2'].buffer.messages, config.action_size, config.mex_size))
                    
                    # voglio salvare dati relativi a quanto gli aenti SONO INFLUENZATI
                    agents_dict['agent_0'].buffer.mut_info.append(np.mean([mut01[-1], mut02[-1]]))
                    #print("sc0=", sc0[-1])
                    #print("agents_dict['agent_0'].buffer.mut_info=", agents_dict['agent_0'].buffer.mut_info[-1])
                    agents_dict['agent_1'].buffer.mut_info.append(np.mean([mut10[-1], mut12[-1]]))
                    agents_dict['agent_2'].buffer.mut_info.append(np.mean([mut21[-1], mut20[-1]]))
                    mut0_avg.append(np.mean([mut01[-1], mut02[-1]]))
                    mut1_avg.append(np.mean([mut10[-1], mut12[-1]]))
                    mut2_avg.append(np.mean([mut21[-1], mut20[-1]]))

                    # break; if the episode is over
                    if done:
                        break

                if (ep_in+1) % config.print_freq == 0:
                    print("Experiment : {} \t Episode : {} \t Mult factor : {} \t Iters: {} ".format(experiment, \
                        ep_in, parallel_env.current_multiplier, config.num_game_iterations))
                    print("Episodic Reward:")
                    for ag_idx, agent in agents_dict.items():
                        print("Agent=", ag_idx, "action=", actions[ag_idx], "rew=", rewards[ag_idx])

                if (ep_in%config.save_interval == 0):
                    #sc0.append(U.calc_mutinfo(agents_dict['agent_0'].buffer.actions, agents_dict['agent_0'].buffer.messages, config.action_size, config.mex_size))
                    #sc1.append(U.calc_mutinfo(agents_dict['agent_1'].buffer.actions, agents_dict['agent_1'].buffer.messages, config.action_size, config.mex_size))
                    #sc2.append(U.calc_mutinfo(agents_dict['agent_2'].buffer.actions, agents_dict['agent_2'].buffer.messages, config.action_size, config.mex_size))
                    h0.append(U.calc_entropy(agents_dict['agent_0'].buffer.messages, config.mex_size))
                    h1.append(U.calc_entropy(agents_dict['agent_1'].buffer.messages, config.mex_size))
                    h2.append(U.calc_entropy(agents_dict['agent_2'].buffer.messages, config.mex_size))
                # update PPO agents     
                if ep_in != 0 and ep_in % config.update_timestep == 0:
                    for ag_idx, agent in agents_dict.items():
                        agent.update()

                if (ep_in%config.save_interval == 0):

                    avg_coop_time.append(np.mean([np.mean(agent.tmp_actions) for _, agent in agents_dict.items()]))
                    if (config.wandb_mode == "online"):
                        for ag_idx, agent in agents_dict.items():
                            wandb.log({ag_idx+"_return": agent.tmp_return}, step=ep_in)
                            wandb.log({ag_idx+"_coop_level": np.mean(agent.tmp_actions)}, step=ep_in)
                        wandb.log({"episode": ep_in}, step=ep_in)
                        wandb.log({"avg_return": np.mean([agent.tmp_return for _, agent in agents_dict.items()])}, step=ep_in)
                        wandb.log({"avg_coop": avg_coop_time[-1]}, step=ep_in)
                        wandb.log({"avg_coop_time": np.mean(avg_coop_time[-10:])}, step=ep_in)

                    if (config.save_data == True):
                        df_ret = {"ret_ag"+str(i): agents_dict["agent_"+str(i)].tmp_return for i in range(config.n_agents)}
                        df_coop = {"coop_ag"+str(i): np.mean(agents_dict["agent_"+str(i)].tmp_actions) for i in range(config.n_agents)}
                        df_avg_coop = {"avg_coop": avg_coop_time[-1]}
                        df_avg_coop_time = {"avg_coop_time": np.mean(avg_coop_time[-10:])}
                        df_dict = {**{'experiment': experiment, 'episode': ep_in, \
                            'mult_factor': parallel_env.current_multiplier, 'uncertainty': config.unc}, \
                            **df_ret, **df_coop, **df_avg_coop, **df_avg_coop_time}
                        df = pd.concat([df, pd.DataFrame.from_records([df_dict])])

        if (config.plots == True):
            ### PLOT TRAIN RETURNS
            U.plot_train_returns(config, agents_dict, path, "train_returns_pgg")

            # COOPERATIVITY PERCENTAGE PLOT
            U.cooperativity_plot(config, agents_dict, path, "train_cooperativeness")

            mutinfos = [mut0_avg, mut1_avg, mut2_avg]
            #print("mut0_avg=", mut0_avg)
            U.plot_info(config, mutinfos, path, "mutinfo_or_instantaneous_coordination")

            SCs = [sc0, sc1, sc2]
            U.plot_info(config, SCs, path, "speaker_consistency")

            Hs = [h0, h1, h2]
            U.plot_info(config, Hs, path, "entropy")

            plt.figure(0)
            n, bins, patches = plt.hist(mult_factors)
            plt.savefig(path+"hist.png")

    if (config.save_data == True):
        df.to_csv(path+'heatmap_data_unc'+str(config.unc)+'.csv')
    
    # save models
    if (config.save_models == True):
        print("Saving models...")
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy_comm.state_dict(), path+"policy_comm_"+str(ag_idx))
            torch.save(ag.policy_act.state_dict(), path+"policy_act_"+str(ag_idx))

if __name__ == "__main__":

    print('cmd entry:', sys.argv)
    unc1 = sys.argv[1]
    print("unc1=", unc1)
    train(config)