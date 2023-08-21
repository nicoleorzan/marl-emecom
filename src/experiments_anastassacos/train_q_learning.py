from src.environments import prisoner_dilemma
from src.algos.anast.Q_learning_anast import Q_learning_agent
import numpy as np
import optuna
import random
from optuna.trial import TrialState
import torch
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
from src.algos.normativeagent import NormativeAgent
from src.utils.social_norm import SocialNorm
from src.utils.utils import  pick_agents_idxs
from src.experiments_anastassacos.params import setup_training_hyperparams

torch.autograd.set_detect_anomaly(True)


def define_agents(config):
    agents = {}
    for idx in range(config.n_agents):
        if (config.is_dummy[idx] == 0):
            agents['agent_'+str(idx)] = Q_learning_agent(config, idx) 
        else: 
            agents['agent_'+str(idx)] = NormativeAgent(config, idx)
    return agents

def interaction_loop(parallel_env, active_agents, active_agents_idxs, n_iterations, social_norm, gamma, _eval=False):
    # By default this is a training loop

    _ = parallel_env.reset()
    rewards_dict = {}
    actions_dict = {}
        
    states = {}; next_states = {}
    for idx_agent, agent in active_agents.items():
        other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
        next_states[idx_agent] = torch.Tensor([other.reputation])

    done = False
    for i in range(n_iterations):

        # state
        actions = {}; states = next_states
        for idx_agent, agent in active_agents.items():
            agent.state_act = states[idx_agent]
        
        # action
        for agent in parallel_env.active_agents:
            a, d = active_agents[agent].select_action(_eval)
            actions[agent] = a

        # reward
        _, rewards, done, _ = parallel_env.step(actions)

        if (_eval==True):
            for ag_idx in active_agents_idxs:       
                if "agent_"+str(ag_idx) not in rewards_dict.keys():
                    rewards_dict["agent_"+str(ag_idx)] = [rewards["agent_"+str(ag_idx)]]
                    actions_dict["agent_"+str(ag_idx)] = [actions["agent_"+str(ag_idx)]]
                else:
                    rewards_dict["agent_"+str(ag_idx)].append(rewards["agent_"+str(ag_idx)])
                    actions_dict["agent_"+str(ag_idx)].append(actions["agent_"+str(ag_idx)])

        social_norm.save_actions(actions, active_agents_idxs)
        social_norm.rule09_binary(active_agents, active_agents_idxs)

        # next state
        next_states = {}
        for idx_agent, agent in active_agents.items():
            other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
            next_states[idx_agent] = torch.Tensor([other.reputation])

        if (_eval == False):
            # save iteration            
            for ag_idx, agent in active_agents.items():
                agent.append_to_replay(states[idx_agent], actions[idx_agent], rewards[idx_agent], next_states[idx_agent])
                agent.return_episode =+ rewards[ag_idx]

        if done:
            if (_eval == True):
                R = {}; avg_coop = {}
                for ag_idx, agent in active_agents.items():
                    R[ag_idx] = 0
                    for r in rewards_dict[ag_idx][::-1]:
                        R[ag_idx] = r + gamma * R[ag_idx]
                    avg_coop[ag_idx] = torch.mean(torch.stack(actions_dict[ag_idx]))
            break

    if (_eval == True):
        return R, avg_coop

def objective(args, repo_name, trial=None):

    all_params = setup_training_hyperparams(args, trial)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)
    config = wandb.config
    print("config=", config)

    # define env
    parallel_env = prisoner_dilemma.parallel_env(config)

    # define agents
    agents = define_agents(config)

    # define social norm
    social_norm = SocialNorm(config, agents)
    
    #### TRAINING LOOP
    avg_returns_train = []; avg_rep_list = []
    for epoch in range(config.n_episodes):
        print("\n==========>Epoch=", epoch)

        # pick a pair of agents
        active_agents_idxs = pick_agents_idxs(config)
        active_agents = {"agent_"+str(key): agents["agent_"+str(key)] for key, _ in zip(active_agents_idxs, agents)}

        [agent.reset() for _, agent in active_agents.items()]

        parallel_env.set_active_agents(active_agents_idxs)
        
        interaction_loop(parallel_env, active_agents, active_agents_idxs, config.num_game_iterations, social_norm, config.gamma, _eval=False)

        # update agents
        for ag_idx, agent in active_agents.items():
            agent.update()

        # evalutaion step
        returns_eval, avg_coop = interaction_loop(parallel_env, active_agents, active_agents_idxs, config.num_game_iterations, social_norm, config.gamma, _eval=True)
        print("returns_eval=", returns_eval)
        print("avg_coop=", avg_coop)

        avg_rep = np.mean([agent.reputation[0] for _, agent in agents.items() if (agent.is_dummy == False)])
        measure = avg_rep
        avg_rep_list.append(avg_rep)

        if (config.optuna_):
            trial.report(measure, epoch)
            
            if trial.should_prune():
                print("is time to pruneee")
                wandb.finish()
                raise optuna.exceptions.TrialPruned()

        if (config.wandb_mode == "online" and float(epoch)%10. == 0.):
            for ag_idx, agent in active_agents.items():
                if (agent.is_dummy == False):
                    df_avg_coop = {ag_idx+"actions_eval": avg_coop[ag_idx]}
                    df_ret = {ag_idx+"returns_eval": returns_eval[ag_idx]}
                    df_Q = {ag_idx+"Q[0,0]": agent.Q[0,0], ag_idx+"Q[0,1]": agent.Q[0,1], ag_idx+"Q[1,0]": agent.Q[1,0], ag_idx+"Q[1,1]": agent.Q[1,1]}
                    df_agent = {**{
                        ag_idx+"_reputation": agent.reputation,
                        'epoch': epoch}, 
                        **df_avg_coop, **df_ret, **df_Q
                        }
                else:
                    df_avg_coop = {ag_idx+"dummy_actions_eval": avg_coop[ag_idx]}
                    df_ret = {ag_idx+"dummy_returns_eval": returns_eval[ag_idx]}
                    df_agent = {**{
                        ag_idx+"dummy_reputation": agent.reputation,
                        'epoch': epoch}, 
                        **df_avg_coop, **df_ret
                        }
                
                if ('df_agent' in locals() ):
                    wandb.log(df_agent, step=epoch, commit=False)
                    
            wandb.log({
                "epoch": epoch,
                "avg_rep": avg_rep,
                "avg_rew_time": measure,
                "mean_Q00": torch.mean(torch.stack([agent.Q[0,0] for _, agent in agents.items() if agent.is_dummy == False])),
                "mean_Q01": torch.mean(torch.stack([agent.Q[0,1] for _, agent in agents.items() if agent.is_dummy == False])),
                "mean_Q10": torch.mean(torch.stack([agent.Q[1,0] for _, agent in agents.items() if agent.is_dummy == False])),
                "mean_Q11": torch.mean(torch.stack([agent.Q[1,1] for _, agent in agents.items() if agent.is_dummy == False]))
                },
                step=epoch, commit=True)

        if (epoch%10 == 0):
            print("Epoch : {} \t Measure: {} ".format(epoch, measure))
    
    wandb.finish()
    return measure


def train_q_learning(args):

    unc_string = "no_unc_"
    if (args.uncertainties.count(0.) != args.n_agents):
        unc_string = "unc_"

    repo_name = "ANAST_"+ str(args.n_agents) + "agents_" + \
        unc_string + args.algorithm + "_dummy_population_" + str(args.proportion_dummy_agents)
    
    if (args.addition != ""):
        repo_name += "_"+ str(args.addition)
    print("repo_name=", repo_name)

    # If optuna, then optimize or get best params. Else use given params
    if (args.optuna_ == 0):
        objective(args, repo_name)
    else:
        func = lambda trial: objective(args, repo_name, trial)

        # sql not optimized for paralel sync
        #storage = optuna.storages.RDBStorage(url="sqlite:///"+repo_name+"-db") 
        journal_name = repo_name + "_binary_"+str(args.binary_reputation)

        storage = JournalStorage(JournalFileStorage("optuna-journal"+journal_name+".log"))

        study = optuna.create_study(
            study_name=repo_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
            n_startup_trials=0, n_warmup_steps=40, interval_steps=3
            )
        )

        if (args.optimize):
            study.optimize(func, n_trials=100, timeout=1000)

        else:
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            
            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")

            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            
            print("Running with best params:")
            objective(args, repo_name+"_BEST", study.best_trial)