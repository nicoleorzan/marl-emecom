import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from src.environments import pgg_parallel_v0
from src.algos.ReinforceComm import ReinforceComm
import numpy as np
import torch
import wandb
import json
import pandas as pd
import src.analysis.utils as U
import time
from utils_train_reinforce_comm import eval

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
    episodes_per_experiment = 80000,
    update_timestep = 128,        # update policy every n timesteps
    n_agents = 3,
    uncertainties = [0., 0., 0.],
    mult_fact = [0.,3.,5.],        # list givin min and max value of mult factor
    num_game_iterations = 1,
    obs_size = 2,                # we observe coins we have, and multiplier factor with uncertainty
    action_size = 2,
    hidden_size = 128,
    lr_actor = 0.001,             # learning rate for actor network
    lr_critic = 0.0005,           # learning rate for critic network
    lr_actor_comm = 0.01,        # learning rate for actor network
    lr_critic_comm = 0.05,      # learning rate for critic network
    decayRate = 0.99,
    fraction = True,
    comm = True,
    plots = True,
    save_models = True,
    save_data = True,
    mex_size = 2,
    random_baseline = False,
    recurrent = False,
    wandb_mode ="online",
    normalize_nn_inputs = True,
    new_loss = True,
    sign_lambda = [0.01, 0.01, 0.01],
    list_lambda = [0.1, 0.1, 0.1]
)


wandb.init(project="reinforce_pgg_v0_comm", entity="nicoleorzan", config=hyperparameter_defaults, mode=hyperparameter_defaults["wandb_mode"])#, sync_tensorboard=True)
config = wandb.config

if (config.mult_fact[0] != config.mult_fact[1]):
    folder = str(config.n_agents)+"agents/"+"variating_m_"+str(config.num_game_iterations)+"iters_"+str(config.uncertainties)+"uncertainties"+"/comm/REINFORCE/"
else: 
    folder = str(config.n_agents)+"agents/"+str(config.mult_fact[0])+"mult_"+str(config.num_game_iterations)+"iters_"+str(config.uncertainties)+"uncertainties"+"/comm/REINFORCE/"
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

    mut01 = []; mut10 = []; mut12 = []; mut21 = []; mut20 = []; mut02 = []
    sc0 = []; sc1 = []; sc2 = []
    h0 = []; h1 = []; h2  = []
    mult_factors = []

    parallel_env = pgg_parallel_v0.parallel_env(config)
    m_min = min(config.mult_fact)
    m_max = max(config.mult_fact)

    if (config.save_data == True):
        df = pd.DataFrame(columns=['experiment', 'episode'] + \
            ["ret_ag"+str(i)+"_train" for i in range(config.n_agents)] + \
            ["coop_ag"+str(i)+"_train" for i in range(config.n_agents)] + \
            ["avg_coop_train", "avg_coop_time_train", "coop_m"+str(m_min), "coop_m"+str(m_max), "performance_metric"] + \
            ["mutinfo_signaling_ag"+str(i) for i in range(config.n_agents)] + \
            ["mutinfo_listening_ag"+str(i) for i in range(config.n_agents)])

    update_idx = 0
    for experiment in range(config.n_experiments):
        #print("\nExperiment ", experiment)

        agents_dict = {}
        for idx in range(config.n_agents):
            agents_dict['agent_'+str(idx)] = ReinforceComm(config, config.sign_lambda[idx], config.list_lambda[idx])
            #wandb.watch(agents_dict['agent_'+str(idx)].policy_act, log = 'all', log_freq = 1)

        #### TRAINING LOOP
        avg_coop_time = []
        for ep_in in range(config.episodes_per_experiment):
            #print("\nEpisode=", ep_in)

            observations = parallel_env.reset()
            #print("obs=", observations)
            mult_factors.append(parallel_env.current_multiplier)
                
            [agent.reset_episode() for _, agent in agents_dict.items()]

            done = False
            while not done:

                train_mult_factor = parallel_env.current_multiplier

                if (config.random_baseline):
                    messages = {agent: agents_dict[agent].random_messages(observations[agent]) for agent in parallel_env.agents}
                else:
                    messages = {agent: agents_dict[agent].select_message(observations[agent]) for agent in parallel_env.agents}
                message = torch.stack([v for _, v in messages.items()]).view(-1).to(device)
                #print("mex=", message)
                actions = {agent: agents_dict[agent].select_action(observations[agent], message) for agent in parallel_env.agents}
                observations, rewards, done, _ = parallel_env.step(actions)

                for ag_idx, agent in agents_dict.items():
                    
                    agent.rewards.append(rewards[ag_idx])
                    agent.return_episode += rewards[ag_idx]
                    if (actions[ag_idx] is not None):
                        agent.tmp_actions.append(actions[ag_idx])
                    if done:
                        agent.train_returns.append(agent.return_episode)
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
                
                # voglio salvare dati relativi a quanto gli agenti INFLUNEZANO
                agents_dict['agent_0'].mutinfo_signaling.append(np.mean([mut10[-1], mut20[-1]]))
                agents_dict['agent_1'].mutinfo_signaling.append(np.mean([mut01[-1], mut21[-1]]))
                agents_dict['agent_2'].mutinfo_signaling.append(np.mean([mut02[-1], mut12[-1]]))

                # voglio salvare dati relativi a quanto gli agenti SONO INFLUENZATI
                agents_dict['agent_0'].mutinfo_listening.append(np.mean([mut01[-1], mut02[-1]]))
                agents_dict['agent_1'].mutinfo_listening.append(np.mean([mut10[-1], mut12[-1]]))
                agents_dict['agent_2'].mutinfo_listening.append(np.mean([mut20[-1], mut21[-1]]))

                agents_dict['agent_0'].sc.append(sc0)
                agents_dict['agent_1'].sc.append(sc1)
                agents_dict['agent_2'].sc.append(sc2)

                # break; if the episode is over
                if done:
                    break

            if (ep_in != 0 and ep_in%config.update_timestep == 0):
                h0.append(U.calc_entropy(agents_dict['agent_0'].buffer.messages, config.mex_size))
                h1.append(U.calc_entropy(agents_dict['agent_1'].buffer.messages, config.mex_size))
                h2.append(U.calc_entropy(agents_dict['agent_2'].buffer.messages, config.mex_size))
                # update PPO agents     
                for ag_idx, agent in agents_dict.items():
                    agent.update()

                print("\nExperiment : {} \t Episode : {} \t Mult factor : {} \t Iters: {} ".format(experiment, \
                ep_in, parallel_env.current_multiplier, config.num_game_iterations))
                
                coop_min, distrib_min = eval(config, parallel_env, agents_dict, m_min, device)
                coop_max, distrib_max = eval(config, parallel_env, agents_dict, m_max, device)
                
                coops_eval = {}
                for m in config.mult_fact:
                    _, distrib = eval(config, parallel_env, agents_dict, m, device)
                    coops_eval[m] = distrib

                print("coop with m="+str(m_min)+":", coop_min)
                print("coop with m="+str(m_max)+":", coop_max)
                performance_metric = coop_max+(1.-coop_min)
                print("Episodic Reward:")
                for ag_idx, agent in agents_dict.items():
                    print("Agent=", ag_idx, "coins=", str.format('{0:.3f}', parallel_env.coins[ag_idx]),\
                        "obs=", agent.buffer.states_a[-1], "action=", actions[ag_idx], "rew=", rewards[ag_idx])
                        #"mutinfo=", agent.mutinfo[-1], "comm entropy=",  str.format('{0:.3f}', agent.comm_entropy[-1].detach().item()))

                avg_coop_time.append(np.mean([agent.tmp_actions_old for _, agent in agents_dict.items()]))
                
                if (config.wandb_mode == "online"):
                    for ag_idx, agent in agents_dict.items():
                        wandb.log({ag_idx+"_return_train": agent.return_episode_old.numpy(),
                            ag_idx+"prob_coop_m_0": coops_eval[0.][ag_idx][1], # action 1 is cooperative
                            #ag_idx+"prob_coop_m_1": coops_eval[1.][ag_idx][1],
                            #ag_idx+"prob_coop_m_2": coops_eval[2.][ag_idx][1],
                            ag_idx+"prob_coop_m_3": coops_eval[3.][ag_idx][1],
                            ag_idx+"prob_coop_m_5": coops_eval[5.][ag_idx][1],
                            ag_idx+"_coop_level_train": np.mean(agent.tmp_actions_old),
                            ag_idx+"_loss": agent.saved_losses[-1],
                            ag_idx+"_loss_comm": agent.saved_losses_comm[-1],
                            ag_idx+"mutinfo_signaling": agent.mutinfo_signaling_old[-1],
                            ag_idx+"mutinfo_listening": agent.mutinfo_listening_old[-1],
                            ag_idx+"messages_prob_distrib_m"+str(m_min): distrib_min[ag_idx],
                            ag_idx+"messages_prob_distrib_m"+str(m_max): distrib_max[ag_idx],
                            ag_idx+"mex_entropy": U.calc_entropy(agents_dict[ag_idx].buffer.messages, config.mex_size)}, step=update_idx)
                    wandb.log({"train_mult_factor": train_mult_factor,
                        "update_idx": update_idx,
                        "episode": ep_in,
                        "avg_return_train": np.mean([agent.return_episode_old.numpy() for _, agent in agents_dict.items()]),
                        "avg_coop_train": avg_coop_time[-1],
                        "avg_coop_time_train": np.mean(avg_coop_time[-10:]),

                        "avg_loss": np.mean([agent.saved_losses[-1] for _, agent in agents_dict.items()]),
                        "avg_loss_comm": np.mean([agent.saved_losses_comm[-1] for _, agent in agents_dict.items()]),
                        "sum_avg_losses": np.mean([agent.saved_losses_comm[-1] for _, agent in agents_dict.items()]) + np.mean([agent.saved_losses[-1] for _, agent in agents_dict.items()]),

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
                   
                    df_signaling = {"mutinfo_signaling_ag"+str(i): agents_dict["agent_"+str(i)].mutinfo_signaling_old[-1] for i in range(config.n_agents)}
                    df_listening = {"mutinfo_listening_ag"+str(i): agents_dict["agent_"+str(i)].mutinfo_listening_old[-1] for i in range(config.n_agents)}
                    df_dict = {**{'experiment': experiment, 'episode': ep_in}, **df_ret, **df_coop, \
                        **df_avg_coop, **df_avg_coop_time, **df_performance, \
                        **df_signaling, **df_listening}
                    df = pd.concat([df, pd.DataFrame.from_records([df_dict])])

                update_idx += 1
                
        if (config.plots == True):
            print("\nPlotsssssss")
            ### PLOT TRAIN RETURNS
            U.plot_train_returns(config, agents_dict, path, "train_returns_pgg")

            # COOPERATIVITY PERCENTAGE PLOT
            U.cooperativity_plot(config, agents_dict, path, "train_cooperativeness")

            mutinfos = [agents_dict["agent_"+str(i)].mutinfo_signaling for i in range(config.n_agents)]
            #print("mut0_avg=", mut0_avg)
            U.plot_info(config, mutinfos, path, "mutinfo_or_instantaneous_coordination")

            SCs = [sc0, sc1, sc2]
            U.plot_info(config, SCs, path, "speaker_consistency")

            Hs = [h0, h1, h2]
            U.plot_info(config, Hs, path, "entropy")

            U.plot_losses(config, agents_dict, path, "losses")

            U.plot_losses(config, agents_dict, path, "losses_comm", True)

    if (config.save_data == True):
        print("\n\n\n\n===========>Saving data")
        print(df.head(3))
        df.to_csv(path+'data_comm'+time.strftime("%Y%m%d-%H%M%S")+'.csv')
    
    # save models
    if (config.save_models == True):
        print("Saving models...")
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy_comm.state_dict(), path+"policy_comm_"+str(ag_idx))
            torch.save(ag.policy_act.state_dict(), path+"policy_act_"+str(ag_idx))

if __name__ == "__main__":
    train(config)