from src.environments import pgg
from src.algos.anast.Reinforce_anast import Reinforce
import numpy as np
import optuna
from optuna.trial import TrialState
import torch
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
from src.algos.anast.normativeagent_anast import NormativeAgent
from src.utils.social_norm import SocialNorm
from src.utils.utils import pick_agents_idxs
from src.experiments_my_game.params import setup_training_hyperparams

torch.autograd.set_detect_anomaly(True)


def define_agents(config):
    agents = {}
    for idx in range(config.n_agents):
        if (config.is_dummy[idx] == 0):
            agents['agent_'+str(idx)] = Reinforce(config, idx) 
        else: 
            agents['agent_'+str(idx)] = NormativeAgent(config, idx)
    return agents

def interaction_loop(parallel_env, active_agents, active_agents_idxs, n_iterations, social_norm, _eval=False):
    # By default this is a training loop

    _ = parallel_env.reset()
    rewards_dict = {}
    actions_dict = {}
        
    states = {}; next_states = {}
    for idx_agent, agent in active_agents.items():
        other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
        next_states[idx_agent] = torch.Tensor([other.reputation])

    done = False
    for _ in range(n_iterations):

        # state
        actions = {}; states = next_states; logprobs = {}
        for idx_agent, agent in active_agents.items():
            agent.state_act = states[idx_agent]
        
        # action
        for agent in parallel_env.active_agents:
            a, logp = active_agents[agent].select_action(_eval)
            actions[agent] = a
            logprobs[agent] = logp

        # reward
        _, rewards, done, _ = parallel_env.step(actions)

        #if (_eval == True):
        #    print("actions=", actions)
        #    print("rewards=", rewards)

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
                if (agent.is_dummy == False):
                    agent.append_to_replay(states[ag_idx], actions[ag_idx], rewards[ag_idx], next_states[ag_idx], logprobs[ag_idx], done)
                    agent.return_episode =+ rewards[ag_idx]

        if done:
            if (_eval == True):
                avg_reward = {}; avg_coop = {}
                for ag_idx, agent in active_agents.items():
                    #print("actions_dict[ag_idx])",torch.stack(actions_dict[ag_idx]).float())
                    avg_coop[ag_idx] = torch.mean(torch.stack(actions_dict[ag_idx]).float())
                    avg_reward[ag_idx] = torch.mean(torch.stack(rewards_dict[ag_idx]))
            break

    if (_eval == True):
        return avg_reward, avg_coop

def objective(args, repo_name, trial=None):

    all_params = setup_training_hyperparams(args, trial)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)
    config = wandb.config
    print("config=", config)

    # define env
    parallel_env = pgg.parallel_env(config)

    # define agents
    agents = define_agents(config)

    # define social norm
    social_norm = SocialNorm(config, agents)
    
    #### TRAINING LOOP
    avg_rep_list = []
    weighted_average_coop_list = []

    for epoch in range(config.n_episodes):
        print("\n==========>Epoch=", epoch)

        # pick a pair of agents
        active_agents_idxs = pick_agents_idxs(config)
        active_agents = {"agent_"+str(key): agents["agent_"+str(key)] for key, _ in zip(active_agents_idxs, agents)}

        [agent.reset() for _, agent in active_agents.items()]

        parallel_env.set_active_agents(active_agents_idxs)

        # TRAIN
        print("TRAIN")        
        interaction_loop(parallel_env, active_agents, active_agents_idxs, config.num_game_iterations, social_norm, _eval=False)

        # update agents
        losses = {}
        for ag_idx, agent in active_agents.items():
            losses[ag_idx] = agent.update()

        # evaluation step
        avg_rew, avg_coop = interaction_loop(parallel_env, active_agents, active_agents_idxs, config.num_game_iterations, social_norm, _eval=True)
        avg_coop_tot = torch.mean(torch.stack([cop_val for _, cop_val in avg_coop.items()]))
        print("avg_rew_normalized_per_b=", {ag_idx:avg_i/config.b_value for ag_idx, avg_i in avg_rew.items()})
        print("avg_coop_tot=", avg_coop_tot)
        #print("HERE=[losses[ag_idx] for ag_idx, agent in active_agents.items() if (agent.is_dummy == False)]", [losses[ag_idx] for ag_idx, agent in active_agents.items() if (agent.is_dummy == False)])
        avg_loss = torch.mean(torch.stack([losses[ag_idx] for ag_idx, agent in active_agents.items() if (agent.is_dummy == False)]))
        print("avg loss=", avg_loss)

        avg_rep = np.mean([agent.reputation[0] for _, agent in agents.items() if (agent.is_dummy == False)])
        weighted_average_coop = torch.mean(torch.stack([avg_i/config.b_value for _, avg_i in avg_rew.items()]))
        weighted_average_coop_list.append(weighted_average_coop)
        weighted_average_coop_time = torch.mean(torch.stack(weighted_average_coop_list[-10:]))
        measure = avg_rep

        print("weighted_average_coop", weighted_average_coop)
        print("weighted_average_coop_time", weighted_average_coop_time)

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
                    df_avg_coop = {ag_idx+"avg_coop": avg_coop[ag_idx]}
                    df_avg_rew = {ag_idx+"avg_rew": avg_rew[ag_idx]}
                    df_loss = {ag_idx+"loss": losses[ag_idx]}
                    df_agent = {**{
                        ag_idx+"_reputation": agent.reputation,
                        'epoch': epoch}, 
                        **df_avg_coop, **df_avg_rew, **df_loss
                        }
                else:
                    df_avg_coop = {ag_idx+"dummy_avg_coop": avg_coop[ag_idx]}
                    df_avg_rew = {ag_idx+"dummy_avg_rew": avg_rew[ag_idx]}
                    df_agent = {**{
                        ag_idx+"dummy_reputation": agent.reputation,
                        'epoch': epoch}, 
                        **df_avg_coop, **df_avg_rew
                        }
                
                if ('df_agent' in locals() ):
                    wandb.log(df_agent, step=epoch, commit=False)
            dff = {
                "epoch": epoch,
                "avg_rep": avg_rep,
                "avg_loss": avg_loss,
                "avg_rew_time": measure,
                "avg_coop_from_agents": avg_coop_tot,
                "weighted_average_coop": torch.mean(torch.stack([avg_i/config.b_value for _, avg_i in avg_rew.items()])), # only on the agents that played, of course
                "weighted_average_coop_time": weighted_average_coop_time # only on the agents that played, of course
                }
            wandb.log(dff,
                step=epoch, commit=True)

        if (epoch%10 == 0):
            print("Epoch : {} \t Measure: {} ".format(epoch, measure))
    
    wandb.finish()
    return measure


def train_reinforce(args):

    unc_string = "no_unc_"
    if (args.uncertainties.count(0.) != args.n_agents):
        unc_string = "unc_"

    repo_name = "PGG_"+ str(args.n_agents) + "agents_" + \
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