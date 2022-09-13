from src.environments import pgg_parallel_v1
from src.algos.PPOcomm import PPOcomm
import numpy as np
import torch
import wandb
import json
import pandas as pd
import os
import src.analysis.utils as U


hyperparameter_defaults = dict(
    n_experiments = 50,
    threshold = 2,
    episodes_per_experiment = 5000,
    update_timestep = 20,        # update policy every n timesteps
    n_agents = 3,
    uncertainties = [0., 0., 0.],# uncertainty on the observation of your own coins
    num_game_iterations = 1,
    obs_size = 1,                # we observe coins we have
    hidden_size = 53,          
    action_size = 2,
    K_epochs = 34,               # update policy for K epochs5
    eps_clip = 0.28,              # clip parameter for PPO
    gamma = 0.99,                # discount factor
    c1 = 0.35,
    c2 = -0.001,
    lr_actor = 0.003, # 0.001,            # learning rate for actor network
    lr_critic = 0.003, # 0.001,           # learning rate for critic network
    decayRate = 0.980,
    comm = True,
    plots = False,
    save_models = False,
    save_data = True,
    save_interval = 20,
    print_freq = 1000,
    mex_size = 5,
    c3 = 0.39,
    c4 = -0.0004,
    random_baseline = False
)

wandb.init(project="pgg_v1_parallel_comm", entity="nicoleorzan", config=hyperparameter_defaults)#, mode="offline")
config = wandb.config

folder = str(config.n_agents)+"agents/"+str(config.num_game_iterations)+"iters_"+str(config.uncertainties)+"uncertainties"+"/comm/"

path = "data/pgg_v1/"+folder

if (hyperparameter_defaults['random_baseline'] == True):
    path = path+"random_baseline/"

if not os.path.exists(path):
    os.makedirs(path)
    print("New directory is created!")

print("path=", path)

with open(path+'params.json', 'w') as fp:
    json.dump(hyperparameter_defaults, fp)

def train(config):

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
            agents_dict['agent_'+str(idx)] = PPOcomm(config)

        #### TRAINING LOOP
        avg_coop_time = []
        for ep_in in range(config.episodes_per_experiment):
            #print("\nEpisode=", ep_in)

            observations = parallel_env.reset()
                
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
                for ag_idx, agent in agents_dict.items():
                    agent.update()

            if (config.n_experiments == 1 and ep_in%10 == 0):
                avg_coop_time.append(np.mean([np.mean(agent.tmp_actions) for _, agent in agents_dict.items()]))
                for ag_idx, agent in agents_dict.items():
                    wandb.log({ag_idx+"_return": agent.tmp_return}, step=ep_in)
                    wandb.log({ag_idx+"_coop_level": np.mean(agent.tmp_actions)}, step=ep_in)
                wandb.log({"episode": ep_in}, step=ep_in)
                wandb.log({"avg_return": np.mean([agent.tmp_return for _, agent in agents_dict.items()])}, step=ep_in)
                wandb.log({"avg_coop": avg_coop_time[-1]}, step=ep_in)
                wandb.log({"avg_coop_time": np.mean(avg_coop_time[-10:])}, step=ep_in)

            if (config.save_data == True and ep_in%config.save_interval == 0):
                df_ret = {"ret_ag"+str(i): agents_dict["agent_"+str(i)].tmp_return for i in range(config.n_agents)}
                df_coop = {"coop_ag"+str(i): np.mean(agents_dict["agent_"+str(i)].tmp_actions) for i in range(config.n_agents)}
                df_avg_coop = {"avg_coop": np.mean([np.mean(agent.tmp_actions) for _, agent in agents_dict.items()])}
                df_dict = {**{'experiment': experiment, 'episode': ep_in}, **df_ret, **df_coop, **df_avg_coop}
                df = pd.concat([df, pd.DataFrame.from_records([df_dict])])

        if (config.plots == True):
            ### PLOT TRAIN RETURNS
            U.plot_train_returns(config, agents_dict, path, "train_returns_pgg")

            # COOPERATIVITY PERCENTAGE PLOT
            U.cooperativity_plot(config, agents_dict, path, "train_cooperativeness")

    if (config.save_data == True):
        df.to_csv(path+'data_comm_single.csv')
    
    # save models
    print("Saving models...")
    if (config.save_models == True):
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy_comm.state_dict(), path+"policy_comm_"+str(ag_idx))
            torch.save(ag.policy_act.state_dict(), path+"policy_act_"+str(ag_idx))


if __name__ == "__main__":
    train(config)
