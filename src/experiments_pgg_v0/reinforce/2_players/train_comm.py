import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from src.environments import pgg_parallel_v0
from src.algos.ReinforceComm import ReinforceComm
import numpy as np
import torch
import wandb
import src.analysis.utils as U
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


def setup_training(params, repo_name):

    unc = [0. for i in range(params.n_agents)]
    for i in range(params.n_uncertain, 0, -1):
        unc[-i] = params.uncertainty

    hyperparameter_defaults = dict(
        n_experiments = 1,
        episodes_per_experiment = params.episodes_per_experiment,
        update_timestep = params.update_timestep,         # update policy every n timesteps
        n_agents = params.n_agents,
        uncertainties = unc,
        mult_fact = params.mult_fact,        # list givin min and max value of mult factor
        num_game_iterations = 1,
        obs_size = 2,                        # we observe coins we have, and multiplier factor with uncertainty
        action_size = 2,
        hidden_size_comm = params.hidden_size_comm,
        hidden_size_act = params.hidden_size_act,
        n_hidden_comm = params.n_hidden_comm,
        n_hidden_act = params.n_hidden_act,
        lr_actor = params.lr_act,             # learning rate for actor network
        lr_critic = 0.01,
        lr_actor_comm = params.lr_comm,       # learning rate for actor network
        lr_critic_comm = 0.05,
        decayRate = params.decay_rate,
        comm = True,
        save_models = False,
        mex_size = params.mex_size,
        random_baseline = params.random_baseline,
        wandb_mode ="online",
        sign_lambda = params.sign_lambda,
        list_lambda = params.list_lambda,
        gmm_ = params.gmm_
    )

    wandb.init(project=repo_name, entity="nicoleorzan", config=hyperparameter_defaults, mode=hyperparameter_defaults["wandb_mode"])#, sync_tensorboard=True)
    config = wandb.config
    print("config=", config)

    return config


def train(config):

    parallel_env = pgg_parallel_v0.parallel_env(config)
    max_values = find_max_min(config.mult_fact, 4)

    update_idx = 0
    for experiment in range(config.n_experiments):

        agents_dict = {}
        for idx in range(config.n_agents):
            if (config.gmm_ == True and config.uncertainties[idx] != 0.):
                print("agent models uncertainty with GMM")
                agents_dict['agent_'+str(idx)] = ReinforceComm(config, idx, True)
            else:
                print("agent DOES NOT model uncertainty with GMM")
                agents_dict['agent_'+str(idx)] = ReinforceComm(config, idx, False)

        #### TRAINING LOOP
        avg_returns_train_list = []
        for ep_in in range(config.episodes_per_experiment):

            observations = parallel_env.reset()
            
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
                    agent.return_episode =+ rewards[ag_idx]
                    #if (actions[ag_idx] is not None):
                    #    agent.tmp_actions.append(actions[ag_idx])
                    #if done:
                    #    agent.train_returns_norm.append(agent.return_episode_norm)
                    #    agent.coop.append(np.mean(agent.tmp_actions))

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
                
                mex_distrib_given_m = {}
                rewards_eval_norm_m = {}
                actions_eval_m = {}

                for m in config.mult_fact:
                    act_eval, mex_distrib, _, rewards_eval = eval(config, parallel_env, agents_dict, m, device, False)
                    mex_distrib_given_m[m] = mex_distrib # distrib dei messaggei per ogni agente, calcolata con dato input
                    rewards_eval_norm_m[m] = {key: value/max_values[m] for key, value in rewards_eval.items()}
                    actions_eval_m[m] = act_eval

                if (config.wandb_mode == "online"):
                    for ag_idx, agent in agents_dict.items():
                        df_actions = {ag_idx+"actions_eval_m_"+str(i): actions_eval_m[i][ag_idx] for i in config.mult_fact}
                        df_mex = {ag_idx+"messages_prob_distrib_m_"+str(i): mex_distrib_given_m[i][ag_idx] for i in config.mult_fact}
                        df_ret = {ag_idx+"rewards_eval_norm_m"+str(i): rewards_eval_norm_m[i][ag_idx] for i in config.mult_fact}
                        agent_dict = {**{
                            ag_idx+"_return_train_norm": agent.return_episode_old_norm.numpy(),
                            #ag_idx+"_return_train_"+str(mf[0]): agent.return_episode_old.numpy(),
                            ag_idx+"gmm_means": agent.means,
                            ag_idx+"gmm_probabilities": agent.probs,
                            ag_idx+"mutinfo_listening": agent.mutinfo_listening_old[-1],
                            ag_idx+"sc": agent.sc_old[-1],
                            ag_idx+"_avg_mex_entropy": torch.mean(agent.entropy),
                            'episode': ep_in}, 
                            **df_actions, **df_ret, **df_mex}
                        
                        wandb.log(agent_dict, step=update_idx, commit=False)
                    avg_returns_train_list.append([agent.return_episode_old_norm.numpy() for _, agent in agents_dict.items()])
                    wandb.log({
                        "update_idx": update_idx,
                        "current_multiplier": mf,
                        "avg_return_train": avg_returns_train_list[-1],
                        "avg_return_train_time": np.mean(avg_returns_train_list[-10:]),
                        "avg_loss": np.mean([agent.saved_losses[-1] for _, agent in agents_dict.items()]),
                        "avg_loss_comm": np.mean([agent.saved_losses_comm[-1] for _, agent in agents_dict.items()]),
                        },
                        step=update_idx, 
                        commit=True)

                update_idx += 1
    
    if (config.save_models == True):
        print("Saving models...")
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy_comm.state_dict(), "policy_comm_"+str(ag_idx))
            torch.save(ag.policy_act.state_dict(), "policy_act_"+str(ag_idx))

def training_function(params, repo_name):
    print("wandb: saving data in ", repo_name)
    config = setup_training(params, repo_name)
    train(config)