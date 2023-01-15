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
from utils_train_reinforce import find_max_min

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
    episodes_per_experiment = 160000,
    update_timestep = 128,        # update policy every n timesteps
    n_agents = 2,
    uncertainties = [0., 0.5],
    mult_fact =  [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5],          # list givin m$
    num_game_iterations = 1,
    obs_size = 2,                # we observe coins we have, and mu$
    action_size = 2,
    hidden_size = 64,
    lr_actor = 0.05,             # learning rate for actor network
    lr_critic = 0.005,           # learning rate for critic network
    lr_actor_comm = 0.005,        # learning rate for actor network
    lr_critic_comm = 0.005,      # learning rate for critic network
    decayRate = 0.999,
    fraction = True,
    comm = True,
    plots = False,
    save_models = False,
    mex_size = 3,
    random_baseline = False,
    recurrent = False,
    wandb_mode ="online",
    normalize_nn_inputs = True,
    new_loss = True,
    sign_lambda = 0.05,
    list_lambda = 0.05,
    gmm_ = False,
    new = True
)



wandb.init(project="new_2_agents_reinforce_pgg_v0_comm_1_unc", entity="nicoleorzan", config=hyperparameter_defaults, mode=hyperparameter_defaults["wandb_mode"])#, sync_tensorboard=True)
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

    parallel_env = pgg_parallel_v0.parallel_env(config)
    m_min = min(config.mult_fact)
    m_max = max(config.mult_fact)

    max_values = find_max_min(config.mult_fact, 4)

    update_idx = 0
    for experiment in range(config.n_experiments):

        agents_dict = {}
        for idx in range(config.n_agents):
            if (config.gmm_ == True and config.uncertainties[idx] != 0.):
                agents_dict['agent_'+str(idx)] = ReinforceComm(config, idx, True)
            else:
                agents_dict['agent_'+str(idx)] = ReinforceComm(config, idx, False)
            #wandb.watch(agents_dict['agent_'+str(idx)].policy_act, log = 'all', log_freq = 1)

        #### TRAINING LOOP
        for ep_in in range(config.episodes_per_experiment):

            observations = parallel_env.reset()
            old_obs = observations
                
            [agent.reset_episode() for _, agent in agents_dict.items()]

            done = False
            while not done:
                mf = parallel_env.current_multiplier

                if (config.random_baseline):
                    messages = {agent: agents_dict[agent].random_messages(observations[agent]) for agent in parallel_env.agents}
                else:
                    messages = {agent: agents_dict[agent].select_message(observations[agent]) for agent in parallel_env.agents}
                message = torch.stack([v for _, v in messages.items()]).view(-1).to(device)
                actions = {agent: agents_dict[agent].select_action(observations[agent], message) for agent in parallel_env.agents}
                observations, rewards, done, _ = parallel_env.step(actions)

                rewards_norm = {key: value/max_values[float(parallel_env.current_multiplier[0])] for key, value in rewards.items()}

                for ag_idx, agent in agents_dict.items():
                    
                    agent.rewards.append(rewards[ag_idx])
                    agent.return_episode_norm += rewards_norm[ag_idx]
                    if (actions[ag_idx] is not None):
                        agent.tmp_actions.append(actions[ag_idx])
                    if done:
                        agent.train_returns_norm.append(agent.return_episode_norm)
                        agent.coop.append(np.mean(agent.tmp_actions))
  
                # voglio salvare dati relativi a quanto gli agenti INFLUNEZANO
                #agents_dict['agent_0'].mutinfo_signaling.append(mut10[-1])
                #agents_dict['agent_1'].mutinfo_signaling.append(mut01[-1])

                # voglio salvare dati relativi a quanto gli agenti SONO INFLUENZATI
                agents_dict['agent_0'].mutinfo_listening.append(U.calc_mutinfo(agents_dict['agent_0'].buffer.actions, agents_dict['agent_1'].buffer.messages, config.action_size, config.mex_size))
                agents_dict['agent_1'].mutinfo_listening.append(U.calc_mutinfo(agents_dict['agent_1'].buffer.actions, agents_dict['agent_0'].buffer.messages, config.action_size, config.mex_size))

                agents_dict['agent_0'].sc.append(U.calc_mutinfo(agents_dict['agent_0'].buffer.actions, agents_dict['agent_0'].buffer.messages, config.action_size, config.mex_size))
                agents_dict['agent_1'].sc.append(U.calc_mutinfo(agents_dict['agent_1'].buffer.actions, agents_dict['agent_1'].buffer.messages, config.action_size, config.mex_size))

                # break; if the episode is over
                if done:
                    break

            if (ep_in != 0 and ep_in%config.update_timestep == 0):
                # update agents     
                for ag_idx, agent in agents_dict.items():
                    agent.update()

                print("\nExperiment : {} \t Episode : {} \t Mult factor : {} \t Update: {} ".format(experiment, \
                ep_in, parallel_env.current_multiplier, update_idx))
                
                #print("====>EVALUATION")
                coops_distrib = {}
                coops_eval = {}
                mex_distrib_given_m = {}
                rewards_eval_norm_m = {}

                for m in config.mult_fact:
                    coop_val, mex_distrib, act_distrib, rewards_eval = eval(config, parallel_env, agents_dict, m, device, False)
                    coops_eval[m] = coop_val
                    coops_distrib[m] = act_distrib
                    mex_distrib_given_m[m] = mex_distrib # distrib dei messaggei per ogni agente, calcolata con dato input
                    rewards_eval_norm_m[m] = {key: value/max_values[m] for key, value in rewards_eval.items()}

                coop_max = coops_eval[m_max]
                coop_min = coops_eval[m_min]
     
                if (config.wandb_mode == "online"):
                    for ag_idx, agent in agents_dict.items():

                        df_prob_coop = {ag_idx+"prob_coop_m_"+str(i): coops_distrib[i][ag_idx][1] for i in config.mult_fact} # action 1 is cooperative
                        df_mex = {ag_idx+"messages_prob_distrib_m_"+str(i): mex_distrib_given_m[i][ag_idx] for i in config.mult_fact}
                        df_ret = {ag_idx+"rewards_eval_norm_m"+str(i): rewards_eval_norm_m[i][ag_idx] for i in config.mult_fact}
                        agent_dict = {**{
                            ag_idx+"_return_train_norm": agent.return_episode_old_norm.numpy(),
                            ag_idx+"gmm_means": agent.means,
                            ag_idx+"gmm_probabilities": agent.probs,
                            ag_idx+"_coop_level_train": np.mean(agent.tmp_actions_old),
                            ag_idx+"mutinfo_listening": agent.mutinfo_listening_old[-1],
                            ag_idx+"sc": agent.sc_old[-1],
                            ag_idx+"avg_mex_entropy": agent.entropy,
                            'episode': ep_in}, 
                            **df_prob_coop, **df_ret}
                        
                        wandb.log(agent_dict, step=update_idx, commit=False)

                    wandb.log({
                        "update_idx": update_idx,
                        "mf": mf,
                        "avg_loss": np.mean([agent.saved_losses[-1] for _, agent in agents_dict.items()]),
                        "avg_loss_comm": np.mean([agent.saved_losses_comm[-1] for _, agent in agents_dict.items()]),
                        "mult_"+str(m_min)+"_coop": coop_min,
                        "mult_"+str(m_max)+"_coop": coop_max},
                        step=update_idx, 
                        commit=True)

                update_idx += 1

    # save models
    if (config.save_models == True):
        print("Saving models...")
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy_comm.state_dict(), path+"policy_comm_"+str(ag_idx))
            torch.save(ag.policy_act.state_dict(), path+"policy_act_"+str(ag_idx))

if __name__ == "__main__":
    train(config)
