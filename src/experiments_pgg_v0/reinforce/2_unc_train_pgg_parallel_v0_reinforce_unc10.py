import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from src.environments import pgg_parallel_v0
from src.algos.Reinforce import Reinforce
from src.nets.ActorCritic import ActorCritic
import numpy as np
import torch
import wandb
import json
import pandas as pd
import src.analysis.utils as U
import time
from utils_train_reinforce import eval

np.seterr(all='raise')

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

hyperparameter_defaults = dict(
    n_experiments = 1,
    episodes_per_experiment = 60000,
    update_timestep = 128,       # update policy every n timesteps: same as batch side in this case
    n_agents = 3,
    uncertainties = [0., 10., 10.],
    mult_fact = [0.,1.,2.,3.,4.,5.],        # list givin min and max value of mult factor
    num_game_iterations = 1,
    obs_size = 2,                # we observe coins we have, and multiplier factor with uncertainty
    hidden_size = 32, # power of two!
    action_size = 2,
    lr_actor = 0.01,              # learning rate for actor network
    lr_critic = 0.01,           # learning rate for critic network
    decayRate = 0.99,
    fraction = False,
    comm = False,
    plots = True,
    save_models = True,
    save_data = True,
    recurrent = False,
    random_baseline = False,
    wandb_mode = "online",
    normalize_nn_inputs = True
)


wandb.init(project="2_unc_reinforce_pgg_v0_unc10", entity="nicoleorzan", config=hyperparameter_defaults, mode=hyperparameter_defaults["wandb_mode"])
config = wandb.config

if (config.mult_fact[0] != config.mult_fact[1]):
    folder = str(config.n_agents)+"agents/"+"variating_m_"+str(config.num_game_iterations)+"iters_"+str(config.uncertainties)+"uncertainties/REINFORCE/"
else: 
    folder = str(config.n_agents)+"agents/"+str(config.mult_fact[0])+"mult_"+str(config.num_game_iterations)+"iters_"+str(config.uncertainties)+"uncertainties/REINFORCE/"
if (config.normalize_nn_inputs == True):
    folder = folder + "normalized/"

path = "data/pgg_v0/"+folder
if not os.path.exists(path):
    os.makedirs(path)
    print("New directory is created!")

print("path=", path)

with open(path+'params.json', 'w') as fp:
    json.dump(hyperparameter_defaults, fp)

def train(config):

    torch.autograd.set_detect_anomaly(True)

    parallel_env = pgg_parallel_v0.parallel_env(config)
    m_min = min(config.mult_fact)
    m_max = max(config.mult_fact)
    
    if (config.save_data == True):
        df = pd.DataFrame(columns=['experiment', 'episode'] + \
            ["ret_ag"+str(i)+"_train" for i in range(config.n_agents)] + \
            ["coop_ag"+str(i)+"_train" for i in range(config.n_agents)] + \
            ["avg_coop_train", "avg_coop_time_train", "coop_m"+str(m_min), "coop_m"+str(m_max), "performance_metric"])

    for experiment in range(config.n_experiments):
        #print("\nExperiment ", experiment)

        agents_dict = {}
        for idx in range(config.n_agents):
            model = ActorCritic(config, config.obs_size, config.action_size)
            model.to(device)
            optimizer = torch.optim.Adam([
             {'params': model.actor.parameters(), 'lr': config.lr_actor},
             {'params': model.critic.parameters(), 'lr': config.lr_critic} 
             ])
            agents_dict['agent_'+str(idx)] = Reinforce(model, optimizer, config)
            #wandb.watch(agents_dict['agent_'+str(idx)].policy, log = 'all', log_freq = 1)

        #### TRAINING LOOP
        avg_coop_time = []

        update_idx = 0
        for ep_in in range(0,config.episodes_per_experiment):
            #print("\nEpisode=", ep_in)

            # free variables that change in each episode
            [agent.reset_episode() for _, agent in agents_dict.items()]

            observations = parallel_env.reset()

            done = False
            while not done:

                train_mult_factor = parallel_env.current_multiplier

                #print(observations)
                obs_old = observations
              
                actions = {agent: agents_dict[agent].select_action(observations[agent]) for agent in parallel_env.agents}
                
                observations, rewards, done, _ = parallel_env.step(actions)
                #print("rews=", rewards)
                for ag_idx, agent in agents_dict.items():
                    
                    agent.rewards.append(rewards[ag_idx])
                    agent.return_episode += rewards[ag_idx]
                    if (actions[ag_idx] is not None):
                        agent.tmp_actions.append(actions[ag_idx])
                    if done:
                        agent.train_returns.append(agent.return_episode)
                        agent.coop.append(np.mean(agent.tmp_actions))

                # break; if the episode is over
                if done:
                    break

            # update agents with REINFORCE
            if ep_in != 0 and ep_in % config.update_timestep == 0:
                for ag_idx, agent in agents_dict.items():
                    agent.update()

                print("\nExperiment : {} \t Episode : {} \t Mult factor : {} \t Iters: {} ".format(experiment, \
                    ep_in, parallel_env.current_multiplier, config.num_game_iterations))

                coop_min, distrib_min = eval(config, parallel_env, agents_dict, m_min)
                coop_max, distrib_max = eval(config, parallel_env, agents_dict, m_max)
                coops_eval = {}
                for m in config.mult_fact:
                    _, distrib = eval(config, parallel_env, agents_dict, m)
                    coops_eval[m] = distrib
                print("eval coop with m="+str(m_min)+":", coop_min)
                print("eval coop with m="+str(m_max)+":", coop_max)
                performance_metric = coop_max+(1.-coop_min)
                print("Episodic Reward:")
                coins = parallel_env.get_coins()
                for ag_idx, agent in agents_dict.items():
                    print("Agent=", ag_idx, "coins=", str.format('{0:.3f}', coins[ag_idx]), "obs=", obs_old[ag_idx], "action=", actions[ag_idx], "rew=", rewards[ag_idx])

                avg_coop_time.append(np.mean([agent.tmp_actions_old for _, agent in agents_dict.items()]))

                if (config.wandb_mode == "online"):
                    for ag_idx, agent in agents_dict.items():
                        wandb.log({ag_idx+"_return_train": agent.return_episode_old.numpy(),
                        ag_idx+"prob_coop_m_0": coops_eval[0.][ag_idx][1], # action 1 is cooperative
                        ag_idx+"prob_coop_m_1": coops_eval[1.][ag_idx][1],
                        ag_idx+"prob_coop_m_2": coops_eval[2.][ag_idx][1],
                        ag_idx+"prob_coop_m_3": coops_eval[3.][ag_idx][1],
                        ag_idx+"prob_coop_m_5": coops_eval[5.][ag_idx][1],
                        ag_idx+"_coop_level_train": np.mean(agent.tmp_actions_old)}, step=update_idx)
                    wandb.log({"train_mult_factor": train_mult_factor,
                        "update_idx": update_idx,
                        "episode": ep_in,
                        "avg_return_train": np.mean([agent.return_episode_old.numpy() for _, agent in agents_dict.items()]),
                        "avg_coop_train": avg_coop_time[-1],
                        "avg_coop_time_train": np.mean(avg_coop_time[-10:]),
                        # insert some evaluation for m_min and m_max
                        "mult_"+str(m_min)+"_coop": coop_min,
                        "mult_"+str(m_max)+"_coop": coop_max, 
                        "performance_mult_("+str(m_min)+","+str(m_max)+")": performance_metric}, step=update_idx)
                    
                if (config.save_data == True):
                    df_ret = {"ret_ag"+str(i)+"_train": agents_dict["agent_"+str(i)].return_episode_old.numpy()[0] for i in range(config.n_agents)}
                    df_coop = {"coop_ag"+str(i)+"_train": np.mean(agents_dict["agent_"+str(i)].tmp_actions_old) for i in range(config.n_agents)}
                    df_avg_coop = {"avg_coop_train": avg_coop_time[-1]}
                    df_avg_coop_time = {"avg_coop_time_train": np.mean(avg_coop_time[-10:])}

                    df_performance = {"coop_m"+str(m_min): coop_min, "coop_m"+str(m_max): coop_max, "performance_metric": performance_metric}
                    df_dict = {**{'experiment': experiment, 'episode': ep_in}, **df_ret, **df_coop, **df_avg_coop, **df_avg_coop_time, **df_performance}
                    df = pd.concat([df, pd.DataFrame.from_records([df_dict])])

                update_idx += 1

        if (config.plots == True):
            ### PLOT TRAIN RETURNS
            U.plot_train_returns(config, agents_dict, path, "train_returns_pgg")

            # COOPERATIVITY PERCENTAGE PLOT
            U.cooperativity_plot(config, agents_dict, path, "train_cooperativeness")

            U.plot_losses(config, agents_dict, path, "losses")

    if (config.save_data == True):
        if (config.random_baseline == True):
            df.to_csv(path+'2_unc_data_simple_RND.csv')
        else: 
            df.to_csv(path+'2_unc_data_simple_unc10'+time.strftime("%Y%m%d-%H%M%S")+'.csv')

    # save models
    print("Saving models...")
    if (config.save_models == True):
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy.state_dict(), path+"model_"+str(ag_idx))


if __name__ == "__main__":
    train(config)