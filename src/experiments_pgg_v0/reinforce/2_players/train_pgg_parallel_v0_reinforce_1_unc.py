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
import time
from utils_train_reinforce import eval, find_max_min

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
    episodes_per_experiment = 40000,
    update_timestep = 64,       # update policy every n timesteps: same as batch side in this case
    n_agents = 2,
    uncertainties = [0., 0.5],
    mult_fact =  [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5],       # list givin min and max value of mult factor
    num_game_iterations = 1,
    obs_size = 2,                # we observe coins we have, and multiplier factor with uncertainty
    hidden_size = 8, # power of two!
    action_size = 2,
    lr_actor = 0.01, #0.005,              # learning rate for actor network
    lr_critic = 0.1, #0.005,           # learning rate for critic network
    decayRate = 0.995,
    fraction = False,
    comm = False,
    save_models = False,
    recurrent = False,
    random_baseline = False,
    wandb_mode = "online",
    normalize_nn_inputs = True,
    gmm_ = False,
    new = True
)


wandb.init(project="new_2_agents_reinforce_pgg_v0_1_unc", entity="nicoleorzan", config=hyperparameter_defaults, mode=hyperparameter_defaults["wandb_mode"])
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
    max_values = find_max_min(config.mult_fact, 4)

    for experiment in range(config.n_experiments):
        #print("\nExperiment ", experiment)

        agents_dict = {}
        for idx in range(config.n_agents):
            if (config.gmm_ == True and config.uncertainties[idx] != 0.):
                print("agente", idx, "modella l'uncertainty")
                model = ActorCritic(config, len(config.mult_fact), config.action_size, True)
            else:
                print("agente", idx, "NON modella l'uncertainty")
                model = ActorCritic(config, config.obs_size, config.action_size, False)
            model.to(device)
            optimizer = torch.optim.Adam([
            {'params': model.actor.parameters(), 'lr': config.lr_actor},
            {'params': model.critic.parameters(), 'lr': config.lr_critic} 
            ])
            agents_dict['agent_'+str(idx)] = Reinforce(model, optimizer, config, idx)
            #wandb.watch(agents_dict['agent_'+str(idx)].policy, log = 'all', log_freq = 1)

        #### TRAINING LOOP
        update_idx = 0
        for ep_in in range(0,config.episodes_per_experiment):

            # free variables that change in each episode
            [agent.reset_episode() for _, agent in agents_dict.items()]

            observations = parallel_env.reset()

            done = False
            while not done:

                mf = parallel_env.current_multiplier
                #print("mf=", mf)

                obs_old = observations
                #print("obs=", obs_old)
              
                actions = {agent: agents_dict[agent].select_action(observations[agent]) for agent in parallel_env.agents}
                
                observations, rewards, done, _ = parallel_env.step(actions)
                rewards_norm = {key: value/max_values[float(parallel_env.current_multiplier[0])] for key, value in rewards.items()}
                
                for ag_idx, agent in agents_dict.items():
                    
                    agent.rewards.append(rewards[ag_idx])
                    agent.return_episode_norm =+ rewards_norm[ag_idx]
                    if (actions[ag_idx] is not None):
                        agent.tmp_actions.append(actions[ag_idx])
                    if done:
                        agent.train_returns_norm.append(agent.return_episode_norm)
                        agent.coop.append(np.mean(agent.tmp_actions))

                # break if the episode is over
                if done:
                    break

            # update agents with REINFORCE
            if ep_in != 0 and ep_in % config.update_timestep == 0:
                for ag_idx, agent in agents_dict.items():
                    agent.update()

                print("\nExperiment: {} \t Episode : {} \t Mult factor : {} \t Update: {} ".format(experiment, \
                    ep_in, parallel_env.current_multiplier, update_idx))
                
                coops_eval = {}
                rewards_eval_norm_m = {}

                for m in config.mult_fact:
                    _, distrib, rewards_eval = eval(config, parallel_env, agents_dict, m, max_values)
                    coops_eval[m] = distrib
                    rewards_eval_norm_m[m] = {key: value/max_values[m] for key, value in rewards_eval.items()}

                coop_max = coops_eval[m_max]
                coop_min = coops_eval[m_min]

                if (config.wandb_mode == "online"):
                    for ag_idx, agent in agents_dict.items():

                        df_prob_coop = {ag_idx+"prob_coop_m_"+str(i): coops_eval[i][ag_idx][1] for i in config.mult_fact} # action 1 is cooperative
                        df_ret = {ag_idx+"rewards_eval_norm_m"+str(i): rewards_eval_norm_m[i][ag_idx] for i in config.mult_fact}
                        agent_dict = {**{
                            ag_idx+"_return_train_norm": agent.return_episode_old_norm.numpy(),
                            ag_idx+"gmm_means": agent.means,
                            ag_idx+"gmm_probabilities": agent.probs,
                            ag_idx+"_coop_level_train": np.mean(agent.tmp_actions_old),
                            'episode': ep_in}, 
                            **df_prob_coop, **df_ret}
                        wandb.log(agent_dict, step=update_idx, commit=False)

                    wandb.log({
                        "update_idx": update_idx,
                        "current_multiplier=": mf,
                        "mult_"+str(m_min)+"_coop": coop_min,
                        "mult_"+str(m_max)+"_coop": coop_max},
                        step=update_idx, 
                        commit=True)

                update_idx += 1

    print("Saving models...")
    if (config.save_models == True):
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy.state_dict(), path+"model_"+str(ag_idx))


if __name__ == "__main__":
    train(config)