import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from src.environments import pgg_parallel_v0
from src.algos.Reinforce import Reinforce
from src.nets.ActorCritic import ActorCritic
import numpy as np
import torch
import wandb
from utils_train_reinforce import eval, find_max_min

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def setup_training(params, repo_name):

    unc = [0. for i in range(params.n_agents)]
    for i in range(params.n_uncertain, 0, -1):
        unc[-i] = params.uncertainty
        
    print("uncertianties=", unc)
    hyperparameter_defaults = dict(
        n_experiments = 1,
        episodes_per_experiment = params.episodes_per_experiment,
        update_timestep = params.update_timestep,         # update policy every n timesteps: same as batch side in this case
        n_agents = params.n_agents,
        uncertainties = unc,
        mult_fact = params.mult_fact,
        num_game_iterations = 1,
        obs_size = 2,                 # we observe coins we have, and multiplier factor with uncertainty
        hidden_size = params.hidden_size,              # power of two!
        n_hidden = params.n_hidden,
        action_size = 2,
        lr_actor = params.lr_actor,              # learning rate for actor network
        lr_critic = 0.01,              # learning rate for critic network
        decayRate = 0.995,
        comm = False,
        save_models = False,
        random_baseline = False,
        wandb_mode = "online",
        gmm_ = params.gmm_
    )

    wandb.init(project=repo_name, entity="nicoleorzan", config=hyperparameter_defaults, mode=hyperparameter_defaults["wandb_mode"])
    config = wandb.config
    print("config=", config)

    return config


def train(config):

    torch.autograd.set_detect_anomaly(True)

    parallel_env = pgg_parallel_v0.parallel_env(config)
    max_values = find_max_min(config.mult_fact, 4)

    for experiment in range(config.n_experiments):
        #print("\nExperiment ", experiment)

        agents_dict = {}
        for idx in range(config.n_agents):
            if (config.gmm_ == True and config.uncertainties[idx] != 0.):
                model = ActorCritic(params=config, input_size=len(config.mult_fact), 
                output_size=config.action_size, n_hidden=config.n_hidden, gmm=True)
            else:
                model = ActorCritic(params=config, input_size=config.obs_size, 
                output_size=config.action_size, n_hidden=config.n_hidden, gmm=False)
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

                _ = observations
              
                actions = {agent: agents_dict[agent].select_action(observations[agent]) for agent in parallel_env.agents}
                
                observations, rewards, done, _ = parallel_env.step(actions)

                rewards_norm = {key: value/max_values[float(parallel_env.current_multiplier[0])] for key, value in rewards.items()}
                
                for ag_idx, agent in agents_dict.items():
                    
                    agent.rewards.append(rewards[ag_idx])
                    agent.return_episode_norm =+ rewards_norm[ag_idx]
                    agent.return_episode =+ rewards[ag_idx]
                    if (actions[ag_idx] is not None):
                        agent.tmp_actions.append(actions[ag_idx])
                    if done:
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

                rewards_eval_norm_m = {}
                actions_eval_m = {}

                for m in config.mult_fact:
                    actions, _, rewards_eval = eval(config, parallel_env, agents_dict, m, max_values)
                    actions_eval_m[m] = actions
                    rewards_eval_norm_m[m] = {key: value/max_values[m] for key, value in rewards_eval.items()}

                if (config.wandb_mode == "online"):
                    for ag_idx, agent in agents_dict.items():
                        df_actions = {ag_idx+"actions_eval_m"+str(i): actions_eval_m[i][ag_idx] for i in config.mult_fact}
                        df_ret = {ag_idx+"rewards_eval_norm_m"+str(i): rewards_eval_norm_m[i][ag_idx] for i in config.mult_fact}
                        agent_dict = {**{
                            ag_idx+"_return_train_norm": agent.return_episode_old_norm.numpy(),
                            ag_idx+"_return_train_"+str(mf[0]): agent.return_episode_old.numpy(),
                            ag_idx+"gmm_means": agent.means,
                            ag_idx+"gmm_probabilities": agent.probs,
                            ag_idx+"_coop_level_train": np.mean(agent.tmp_actions_old),
                            'episode': ep_in}, 
                            **df_actions, **df_ret}
                        wandb.log(agent_dict, step=update_idx, commit=False)

                    wandb.log({
                        "update_idx": update_idx,
                        "current_multiplier": mf[0],
                        "avg_return_train": np.mean([agent.return_episode_old_norm.numpy() for _, agent in agents_dict.items()])},
                        step=update_idx, 
                        commit=True)

                update_idx += 1

    if (config.save_models == True):
        print("Saving models...")
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy.state_dict(), "model_"+str(ag_idx))


def training_function(params, repo_name):
    print("wandb: saving data in ", repo_name)
    config = setup_training(params, repo_name)
    train(config)