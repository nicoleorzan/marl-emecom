from src.environments import pgg_parallel_v0
from src.algos.Reinforce import Reinforce
from src.algos.DQN import DQN
from src.algos.PPO import PPO
import numpy as np
import optuna
import random
from optuna.trial import TrialState
import torch
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
from src.experiments_pgg_v0.utils_train_reinforce import eval_anast, find_max_min
from src.algos.normativeagent import NormativeAgent
from social_norm import SocialNorm
from params import setup_training_hyperparams


torch.autograd.set_detect_anomaly(True)

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def define_agents(config, is_dummy):
    agents = {}
    for idx in range(config.n_agents):
        if (is_dummy[idx] == 0):
            if (config.algorithm == "reinforce"):
                agents['agent_'+str(idx)] = Reinforce(config, idx)
            elif (config.algorithm == "PPO"):
                agents['agent_'+str(idx)] = PPO(config, idx)
            elif (config.algorithm == "dqn"):
                agents['agent_'+str(idx)] = DQN(config, idx)
        else: 
            agents['agent_'+str(idx)] = NormativeAgent(config, idx)
    return agents

def objective(args, repo_name, trial=None):

    all_params = setup_training_hyperparams(args, trial)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)#, sync_tensorboard=True)
    config = wandb.config
    print("config=", config)
    print("binary_reputation=", config.binary_reputation)

    parallel_env = pgg_parallel_v0.parallel_env(config)
    max_values = find_max_min(config, config.coins_value)

    n_communicating_agents = config.communicating_agents.count(1)
    n_uncertain = config.n_agents - config.uncertainties.count(0.)
    print("n_uncertain=", n_uncertain)

    #is_dummy = np.random.binomial(1, p=config.proportion_dummy_agents, size=config.n_agents)
    n_dummy = int(args.proportion_dummy_agents*config.n_agents)
    print("n_dummy=", n_dummy)
    is_dummy = list(reversed([1 if i<n_dummy else 0 for i in range(config.n_agents) ]))
    # print("config.uncertainties=", config.uncertainties)
    print("is_dummy=", is_dummy)
    non_dummy_idxs = [i for i,val in enumerate(is_dummy) if val==0]

    agents = define_agents(config, is_dummy)
    print("\nAGENTS=",agents)
    
    #### TRAINING LOOP
    avg_norm_returns_train_list = []; avg_rew_time = []

    social_norm = SocialNorm(config, agents)

    for epoch in range(config.n_epochs):
        #print("\n=========================")
        print("\n==========>Epoch=", epoch)
        for ag_idx, agent in agents.items():
            print("Reputation agent ", ag_idx, agent.reputation)

        #pick a pair of agents
        #print("OPPONENT SELECTION")
        active_agents_idxs = []
        while ((any(active_agents_idxs) == True) == False):
            if (config.opponent_selection == True):
                first_agent_idx = random.sample(range(config.n_agents), 1)[0]
                reputations = torch.Tensor([agent.reputation for ag_idx, agent in agents.items()])
                second_agent_idx = int(agents["agent_"+str(first_agent_idx)].select_opponent(reputations))
                while (second_agent_idx == first_agent_idx):
                    second_agent_idx = int(agents["agent_"+str(first_agent_idx)].select_opponent(reputations))
                active_agents_idxs = [first_agent_idx, second_agent_idx]
            else:
                active_agents_idxs = random.sample(range(config.n_agents), 2)
                print("active_agents_idxs=",active_agents_idxs)
                while ( any([i in non_dummy_idxs for i in active_agents_idxs]) == False):
                    active_agents_idxs = random.sample(range(config.n_agents), 2)
                    print("active_agents_idxs=",active_agents_idxs)

            #print("active_agents_idxs=",active_agents_idxs)
            active_agents = {"agent_"+str(key): agents["agent_"+str(key)] for key, value in zip(active_agents_idxs, agents)}
            #print("ACTIVE AGENTS=", active_agents)

        parallel_env.set_active_agents(active_agents_idxs)

        [agent.reset_batch() for _, agent in active_agents.items()]
        act_batch = {}; rew_batch = {}; rew_norm_batch = {}
        
        for batch_idx in range(config.batch_size):

            observations = parallel_env.reset()
            
            [agent.reset_episode() for _, agent in active_agents.items()]

            done = False

            while not done:
                actions = {}
                active_agents["agent_"+str(active_agents_idxs[0])].digest_input_anast((active_agents["agent_"+str(active_agents_idxs[1])].reputation, active_agents["agent_"+str(active_agents_idxs[1])].previous_action))
                active_agents["agent_"+str(active_agents_idxs[1])].digest_input_anast((active_agents["agent_"+str(active_agents_idxs[0])].reputation, active_agents["agent_"+str(active_agents_idxs[0])].previous_action))

                # acting
                #print("\nacting")
                for agent in parallel_env.active_agents:
                    actions[agent] = agents[agent].select_action() # m value is given only to compute metrics
                #print("\nactions=", actions)
                observations, rewards, done, _ = parallel_env.step1(actions)
                #print("rewards=", rewards)

                social_norm.save_actions(actions, active_agents_idxs)

                rewards_norm = {key: value/parallel_env.mv for key, value in rewards.items()}
                #print("rewards_norm=", rewards_norm)
                
                for ag_idx, agent in active_agents.items():
                    
                    agent.buffer.rewards.append(rewards[ag_idx])
                    agent.buffer.rewards_norm.append(rewards_norm[ag_idx].item())
                    agent.buffer.next_states_a.append(observations[ag_idx])
                    agent.buffer.is_terminals.append(done)
                    agent.return_episode_norm += rewards_norm[ag_idx]
                    agent.return_episode =+ rewards[ag_idx]

                for k in actions:
                    if (k not in act_batch):
                        act_batch[k] = [actions[k].item()]
                    else:
                        act_batch[k].append(actions[k].item())

                # break; if the episode is over
                #print("actions=", actions)
                for ag_idx, agent in active_agents.items():
                    #print("ag_idx=", ag_idx)
                    #print("agent=", agent)
                    #print("agent.previous_action=", agent.previous_action)
                    #print("actions[agent]=",actions[ag_idx])
                    agent.previous_action = actions[ag_idx].reshape(1)
                    #print("new:agent.previous_action=", agent.previous_action)


                if done:
                    break

        #print("act_batch=", act_batch)
        #print("Avg act agent0=", np.mean(act_batch["agent_0"]))
        #print("Avg act agent1=", np.mean(act_batch["agent_1"]))
        #print("last rewards=",rewards)

        #print("update reputation")
        if (config.binary_reputation == True):
            social_norm.rule09_binary(active_agents_idxs)
        else:    
            social_norm.rule09(active_agents_idxs)

        for ag_idx in active_agents_idxs:       
            agents["agent_"+str(ag_idx)].old_reputation = agents["agent_"+str(ag_idx)].reputation

        #if agents["agent_"+str(config.n_agents-1)+"].reputation == 0.0:
        #    print("\n\n\n\n\n\n\n\n\n\n\nERROR!!!!!!!!!!!!!!!")

        # update agents     
        for ag_idx, agent in active_agents.items():
            agent.update()

        # =================== EVALUATION ===================
        # computing evaluation against the behavior of all the dummy agents
        #print("EVALUATION")
        rewards_eval = {}; rewards_eval_norm = {}; actions_eval = {};  act_distrib = {}

        act_eval, act_distrib, rewards_eval = eval_anast(parallel_env, active_agents, active_agents_idxs)
        rewards_eval = rewards_eval
        print("rewards_eval=", rewards_eval)
        print("max_values=",max_values)
        rewards_eval_norm = {key: value/max_values[1.5] for key, value in rewards_eval.items()}
        actions_eval = act_eval
        act_distrib = act_distrib
        print("act_distrib_m:", act_distrib)
        print("actions=", actions_eval)
        print("rewards=", rewards_eval)
        print("rewards_norm=", rewards_eval_norm)

        avg_norm_return = np.mean([agent.return_episode_old_norm.numpy().item() for _, agent in active_agents.items()])
        avg_norm_returns_train_list.append(avg_norm_return)

        rew_values = [rewards_eval[ag_idx][0] for ag_idx, _ in active_agents.items()]
        print(rew_values)
        avg_rew_time.append(np.mean(rew_values))
        #print(avg_rew_time)
        measure = np.mean(avg_rew_time[-10:])

        avg_rep = np.mean([agent.reputation for ag_idx, agent in agents.items() if (agent.is_dummy == False)])
        #print("avg_rep=", avg_rep)
        if (config.optuna_):
            measure = avg_rep
            print("measure=", measure)
            trial.report(measure, epoch)
            
            if trial.should_prune():
                print("is time to pruneee")
                wandb.finish()
                raise optuna.exceptions.TrialPruned()

        if (config.wandb_mode == "online" and float(epoch)%20. == 0.):
            for ag_idx, agent in active_agents.items():
                if (agent.is_dummy == False):
                    df_actions = {ag_idx+"actions_eval": actions_eval[ag_idx]}
                    df_rew = {ag_idx+"rewards_eval": rewards_eval[ag_idx]}
                    df_rew_norm = {ag_idx+"rewards_eval_norm": rewards_eval_norm[ag_idx]}
                    df_agent = {**{
                        ag_idx+"_return_train_norm": agent.return_episode_old_norm,
                        ag_idx+"_reputation": agent.reputation,
                        ag_idx+"_return_train_": agent.return_episode_old,
                        'epoch': epoch}, 
                        **df_actions, **df_rew, **df_rew_norm}
                else:
                    df_actions = {ag_idx+"N_actions_eval": actions_eval[ag_idx]}
                    df_rew = {ag_idx+"N_rewards_eval": rewards_eval[ag_idx]}
                    df_rew_norm = {ag_idx+"N_rewards_eval_norm": rewards_eval_norm[ag_idx]}
                    df_agent = {**{
                        ag_idx+"N_return_train_norm": agent.return_episode_old_norm,
                        ag_idx+"N_reputation": agent.reputation,
                        ag_idx+"N_return_train_": agent.return_episode_old,
                        'epoch': epoch}, 
                        **df_actions, **df_rew, **df_rew_norm}
                
                if ('df_agent' in locals() ):
                    wandb.log(df_agent, step=epoch, commit=False)

            wandb.log({
                "epoch": epoch,
                "avg_rep": avg_rep,
                "avg_return_train": avg_norm_return,
                "avg_return_train_time": np.mean(avg_norm_returns_train_list[-10:]),
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