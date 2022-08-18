from src.environments import pgg_parallel_v1
from src.algos.PPOcomm_recurrent import PPOcomm_recurrent
import numpy as np
import torch.nn.functional as F
import torch
import wandb
import json
import pandas as pd
import os
import src.analysis.utils as U

hyperparameter_defaults = dict(
    n_experiments = 1,
    threshold = 2,
    episodes_per_experiment = 1000,
    update_timestep = 15, #23,        # update policy every n timesteps
    n_agents = 3,
    uncertainties = [0., 0., 0.],# uncertainty on the observation of your own coins
    num_game_iterations = 1,
    communication_loops = 1, #2
    obs_size = 1,                # we observe coins we have (+ actions of all other agents)
    hidden_size = 53, #23,
    num_rnn_layers = 1,
    action_size = 2,
    K_epochs = 64, #40,               # update policy for K epochs
    eps_clip = 0.11, #0.2,              # clip parameter for PPO
    gamma = 0.99,                # discount factor
    c1 = 0.4, #0.5,
    c2 = 0.004, #-0.01,
    lr = 0.004, #0.002, #0.001,     con 0.002 andava         # learning rate
    decayRate = 0.999,
    comm = False,
    plots = True,
    save_models = True,
    save_data = True,
    save_interval = 10,
    print_freq = 10,
    recurrent = True,
    mex_size = 4, #2,
    c3 = 0.04, #0.8,
    c4 = 0.009, #-0.003,
    random_baseline = False
)




wandb.init(project="pgg_v1_memory_comm", entity="nicoleorzan", config=hyperparameter_defaults)#, mode="offline")
config = wandb.config

folder = str(config.n_agents)+"agents/"+str(config.num_game_iterations)+"iters_"+str(config.uncertainties)+"uncertainties/"

path = "data/pgg_v1/"+folder
if not os.path.exists(path):
    os.makedirs(path)
    print("New directory is created!")

print("path=", path)

with open(path+'params.json', 'w') as fp:
    json.dump(hyperparameter_defaults, fp)

def train(config):

    mut01 = []; mut12 = []; mut20 = []
    sc0 = []; sc1 = []; sc2 = []
    h0 = []; h1 = []; h2  =[]

    parallel_env = pgg_parallel_v1.parallel_env(n_agents=config.n_agents, threshold=config.threshold, \
        num_iterations=config.num_game_iterations, uncertainties=config.uncertainties)

    if (config.save_data == True):
        df = pd.DataFrame(columns=['experiment', 'episode'] + \
            ["ret_ag"+str(i) for i in range(config.n_agents)] + \
            ["coop_ag"+str(i) for i in range(config.n_agents)])

    for experiment in range(config.n_experiments):
        #print("\nExperiment ", experiment)

        agents_dict = {}
        for idx in range(config.n_agents):
            agents_dict['agent_'+str(idx)] = PPOcomm_recurrent(config)

        #### TRAINING LOOP
        for ep_in in range(config.episodes_per_experiment):
            #print("\nEpisode=", ep_in)

            observations = parallel_env.reset()
                
            [agent.reset() for _, agent in agents_dict.items()]

            done = False
            messages_cat = torch.zeros((config.n_agents*config.mex_size), dtype=torch.int64)
            while not done:

                for _ in range(config.communication_loops):

                    messages = {agent: agents_dict[agent].select_message(torch.cat((torch.Tensor(observations[agent]), messages_cat))) for agent in parallel_env.agents}
                    #print("messages=", messages)
                    messages_cat = torch.stack([F.one_hot(torch.Tensor([v.item()]).long(), num_classes=config.mex_size)[0] for _, v in messages.items()]).view(-1)
                    
                actions = {agent: agents_dict[agent].select_action(torch.cat((torch.Tensor(observations[agent]), messages_cat))) for agent in parallel_env.agents}
                #print("actions=", actions)
                observations, rewards, done, _ = parallel_env.step(actions)

                for ag_idx, agent in agents_dict.items():
                    
                    agent.buffer.rewards.append(rewards[ag_idx])
                    agent.buffer.is_terminals.append(done)
                    agent.tmp_return += rewards[ag_idx]
                    if (actions[ag_idx] is not None):
                        agent.tmp_actions.append(actions[ag_idx])
                        agent.train_actions.append(actions[ag_idx])
                    if done:
                        agent.train_returns.append(agent.tmp_return)
                        agent.coop.append(np.mean(agent.tmp_actions))

                # break; if the episode is over
                if done:
                    break

            if (ep_in+1) % config.print_freq == 0:
                print("Experiment : {} \t Episode : {} \t Iters: {} ".format(experiment, \
                    ep_in, config.num_game_iterations))
                print("Episodic Reward:")
                for ag_idx, agent in agents_dict.items():
                    print("Agent=", ag_idx, "action=", actions[ag_idx], "rew=", rewards[ag_idx])

            # update PPO agents
            if ep_in != 0 and ep_in % config.update_timestep == 0:
                mut01.append(U.calc_mutinfo(agents_dict['agent_0'].buffer.actions, agents_dict['agent_1'].buffer.messages, config.action_size, config.mex_size))
                mut12.append(U.calc_mutinfo(agents_dict['agent_1'].buffer.actions, agents_dict['agent_2'].buffer.messages, config.action_size, config.mex_size))
                mut20.append(U.calc_mutinfo(agents_dict['agent_2'].buffer.actions, agents_dict['agent_0'].buffer.messages, config.action_size, config.mex_size))
                sc0.append(U.calc_mutinfo(agents_dict['agent_0'].buffer.actions, agents_dict['agent_0'].buffer.messages, config.action_size, config.mex_size))
                sc1.append(U.calc_mutinfo(agents_dict['agent_1'].buffer.actions, agents_dict['agent_1'].buffer.messages, config.action_size, config.mex_size))
                sc2.append(U.calc_mutinfo(agents_dict['agent_2'].buffer.actions, agents_dict['agent_2'].buffer.messages, config.action_size, config.mex_size))
                h0.append(U.calc_entropy(agents_dict['agent_0'].buffer.messages, config.mex_size))
                h1.append(U.calc_entropy(agents_dict['agent_1'].buffer.messages, config.mex_size))
                h2.append(U.calc_entropy(agents_dict['agent_2'].buffer.messages, config.mex_size))
                for ag_idx, agent in agents_dict.items():
                    agent.update()

            if (config.n_experiments == 1 and ep_in%10 == 0):
                for ag_idx, agent in agents_dict.items():
                    wandb.log({ag_idx+"_return": agent.tmp_return}, step=ep_in)
                    wandb.log({ag_idx+"_coop_level": np.mean(agent.tmp_actions)}, step=ep_in)
                wandb.log({"episode": ep_in}, step=ep_in)
                wandb.log({"avg_return": np.mean([agent.tmp_return for _, agent in agents_dict.items()])}, step=ep_in)
                wandb.log({"avg_coop": np.mean([np.mean(agent.tmp_actions) for _, agent in agents_dict.items()])}, step=ep_in)

            if (config.save_data == True and ep_in%config.save_interval == 0):
                df_ret = {"ret_ag"+str(i): agents_dict["agent_"+str(i)].tmp_return for i in range(config.n_agents)}
                df_coop = {"coop_ag"+str(i): np.mean(agents_dict["agent_"+str(i)].tmp_actions) for i in range(config.n_agents)}
                df_dict = {**{'experiment': experiment, 'episode': ep_in}, **df_ret, **df_coop}
                df = pd.concat([df, pd.DataFrame.from_records([df_dict])])

        if (config.plots == True):
            ### PLOT TRAIN RETURNS
            U.plot_train_returns(config, agents_dict, path, "train_returns_pgg")

            # COOPERATIVITY PERCENTAGE PLOT
            U.cooperativity_plot(config, agents_dict, path, "train_cooperativeness")

            mutinfos = [mut01, mut12, mut20]
            U.plot_info(config, mutinfos, path, "instantaneous coordination")

            SCs = [sc0, sc1, sc2]
            U.plot_info(config, SCs, path, "speaker_consistency")

            Hs = [h0, h1, h2]
            U.plot_info(config, Hs, path, "entropy")

    if (config.save_data == True):
        df.to_csv(path+'data_comm_memory.csv')
    
    # save models
    print("Saving models...")
    if (config.save_models == True):
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy.state_dict(), path+"model_comm_memory_"+str(ag_idx))


if __name__ == "__main__":
    train(config)
