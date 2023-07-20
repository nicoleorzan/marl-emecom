from src.environments import pgg_parallel_v0
from src.algos.Reinforce_anast import Reinforce
import numpy as np
import optuna
import random
from optuna.trial import TrialState
import torch
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
from src.experiments_pgg_v0.utils_train_reinforce import eval_anast
from src.algos.normativeagent import NormativeAgent
from social_norm import SocialNorm
from params import setup_training_hyperparams

torch.autograd.set_detect_anomaly(True)


def define_agents(config, is_dummy):
    agents = {}
    for idx in range(config.n_agents):
        if (is_dummy[idx] == 0):
            if (config.algorithm == "reinforce"):
                agents['agent_'+str(idx)] = Reinforce(config, idx)
        else: 
            agents['agent_'+str(idx)] = NormativeAgent(config, idx)
    return agents

def objective(args, repo_name, trial=None):

    all_params = setup_training_hyperparams(args, trial)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)
    config = wandb.config
    print("config=", config)

    parallel_env = pgg_parallel_v0.parallel_env(config)

    n_uncertain = config.n_agents - config.uncertainties.count(0.)
    print("n_uncertain=", n_uncertain)

    n_dummy = int(args.proportion_dummy_agents*config.n_agents)
    is_dummy = list(reversed([1 if i<n_dummy else 0 for i in range(config.n_agents) ]))
    print("is_dummy=", is_dummy)
    non_dummy_idxs = [i for i,val in enumerate(is_dummy) if val==0]

    agents = define_agents(config, is_dummy)
    #print("\nAGENTS=",agents)
    
    #### TRAINING LOOP
    avg_returns_train = []; avg_rep_list = []

    social_norm = SocialNorm(config, agents)

    for epoch in range(config.n_epochs):
        print("\n==========>Epoch=", epoch)

        #pick a pair of agents
        active_agents_idxs = []
        first_agent_idx = random.sample(non_dummy_idxs, 1)[0]        
        second_agent_idx = random.sample( list(set(range(0, config.n_agents)) - set([first_agent_idx])) , 1)[0]
        active_agents_idxs = [first_agent_idx, second_agent_idx]
        active_agents = {"agent_"+str(key): agents["agent_"+str(key)] for key, _ in zip(active_agents_idxs, agents)}
        [agent.reset() for _, agent in active_agents.items()]

        parallel_env.set_active_agents(active_agents_idxs)
        
        _ = parallel_env.reset()
        
        states = {}; next_states = {}
        for idx_agent, agent in active_agents.items():
            other = agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
            next_states[idx_agent] = torch.cat((other.reputation, agent.reputation, other.previous_action, agent.previous_action), 0)

        done = False
        for i in range(config.num_game_iterations):
            #print("iter=", i)

            if (i == config.num_game_iterations-1): 
                done = True

            actions = {}; states = next_states
            for idx_agent, agent in active_agents.items():
                agent.state_act = states[idx_agent]
            #print("states=", states)
            
            # acting
            for idx_agent, agent in active_agents.items():
                other = agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
                actions[idx_agent] = agents[idx_agent].select_action()

            _, rewards, _, _ = parallel_env.step1(actions)
            print("actions=", actions)
            #print("rewards from the game=", rewards)

            social_norm.save_actions(actions, active_agents_idxs)
            social_norm.rule09_binary(active_agents_idxs)

            for ag_idx in active_agents_idxs:       
                agents["agent_"+str(ag_idx)].old_reputation = agents["agent_"+str(ag_idx)].reputation

            if (config.reputation_in_reward):
                rewards = {key: value+agents[key].reputation for key, value in rewards.items()}
            #print("after rewards=", rewards)

            next_states = {}
            for idx_agent, agent in active_agents.items():
                other = agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
                next_states[idx_agent] = torch.cat((other.reputation, agent.reputation, other.previous_action, agent.previous_action), 0)
            #print("next_states=", next_states)

            #rewards_norm = {key: value/(parallel_env.mv+1.) for key, value in rewards.items()}
            
            for ag_idx, agent in active_agents.items():
                #print("buffer")
                
                agent.previous_action = actions[ag_idx].reshape(1)
                agent.buffer.states.append(states[ag_idx])
                agent.buffer.rewards.append(rewards[ag_idx])
                agent.buffer.next_states.append(next_states[ag_idx])
                agent.buffer.is_terminals.append(done)
                agent.return_episode =+ rewards[ag_idx]
                agent.previous_action = actions[ag_idx].reshape(1)

            # break; if the episode is over
            if done:
                break

        # update agents
        #print("UPDATE")
        for ag_idx, agent in active_agents.items():
            agent.update()

        # =================== EVALUATION =================== CHANGE!!!
        #print("EVALUATION")

        returns_eval = eval_anast(parallel_env, active_agents, active_agents_idxs, config.num_game_iterations, social_norm, 0.99)
        #print("R eval=", returns_eval)
        #print(np.mean([val for k, val in returns_eval.items()]))
        #rewards_eval = {key: value+agents[key].reputation for key, value in rewards_eval.items()}

        #avg_return = np.mean([agent.return_episode_old.numpy().item() for _, agent in active_agents.items()])
        #avg_returns_train.append(avg_return)

        measure = np.mean(avg_returns_train[-10:])

        avg_rep = np.mean([agent.reputation[0] for _, agent in agents.items() if (agent.is_dummy == False)])
        avg_rep_list.append(avg_rep)
        #print("avg rep in time=", np.mean(avg_rep_list[-10:]))
        if (config.optuna_):
            measure = avg_rep
            trial.report(measure, epoch)
            
            if trial.should_prune():
                print("is time to pruneee")
                wandb.finish()
                raise optuna.exceptions.TrialPruned()
        

        if (config.wandb_mode == "online" and float(epoch)%20. == 0.):
            for ag_idx, agent in active_agents.items():
                if (agent.is_dummy == False):
                    #df_actions = {ag_idx+"actions_eval": actions_eval[ag_idx]}
                    #df_rew = {ag_idx+"rewards_eval": rewards_eval[ag_idx]}
                    df_agent = {**{
                        ag_idx+"_reputation": agent.reputation,
                        'epoch': epoch}, 
                        #**df_actions, **df_rew
                        }
                else:
                    #df_actions = {ag_idx+"N_actions_eval": actions_eval[ag_idx]}
                    #df_rew = {ag_idx+"N_rewards_eval": rewards_eval[ag_idx]}
                    df_agent = {**{
                        ag_idx+"N_reputation": agent.reputation,
                        'epoch': epoch}, 
                        #**df_actions, **df_rew
                        }
                
                if ('df_agent' in locals() ):
                    wandb.log(df_agent, step=epoch, commit=False)

            wandb.log({
                "epoch": epoch,
                "avg_rep": avg_rep,
                "avg_rew_time": measure,
                },
                step=epoch, commit=True)

        if (epoch%10 == 0):
            print("Epoch : {} \t Measure: {} ".format(epoch, measure))
    
    wandb.finish()
    return measure


def training_function(args):

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