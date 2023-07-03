from src.environments import pgg_parallel_v0
from src.algos.Reinforce import Reinforce
from src.algos.DQN import DQN
from src.algos.PPO import PPO
import numpy as np
import optuna
import random
from optuna.trial import TrialState
import torch
#from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
import src.analysis.utils as U
from src.experiments_pgg_v0.utils_train_reinforce import eval, find_max_min, apply_norm
from src.algos.normativeagent import NormativeAgent
from social_norm import SocialNorm


torch.autograd.set_detect_anomaly(True)

EPOCHS = 400 # total episodes
# batch size are the number of episodes in which 2 agents interact with each other alone
OBS_SIZE = 3 # input: multiplication factor (with noise), opponent reputation.
# the opponent index is embedded in the agent class
ACTION_SIZE = 2
#WANDB_MODE = "online"
RANDOM_BASELINE = False

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def setup_training_hyperparams(trial, args):

    if (args.binary_reputation == True):
        o_r_t = 1.
        c_t = 1.
    else: 
        o_r_t = trial.suggest_categorical("other_reputation_threshold", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        c_t = trial.suggest_categorical("cooperation_threshold", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    game_params = dict(
        n_agents = args.n_agents,
        algorithm = args.algorithm,
        wandb_mode = args.wandb_mode,
        num_game_iterations = 1,
        n_epochs = EPOCHS,
        obs_size = OBS_SIZE,
        action_size = ACTION_SIZE,
        n_gmm_components = args.mult_fact, #trial.suggest_categorical("n_gmm_components", [3, len(args.mult_fact)]),
        decayRate = 0.999,
        mult_fact = args.mult_fact,
        uncertainties = args.uncertainties,
        gmm_ = args.gmm_,
        random_baseline = RANDOM_BASELINE,
        communicating_agents = args.communicating_agents,
        listening_agents = args.listening_agents,
        batch_size = 128,
        lr_actor = trial.suggest_float("lr_actor", 1e-4, 1e-1, log=True),
        lr_critic = trial.suggest_float("lr_critic", 1e-4, 1e-1, log=True),
        lr_opponent = trial.suggest_float("lr_opponent", 1e-3, 1e-1, log=True),
        n_hidden_act = 2,
        hidden_size_act = trial.suggest_categorical("hidden_size_act", [8, 16, 32, 64]),
        embedding_dim = 1,
        binary_reputation = args.binary_reputation,
        get_index = False,
        get_opponent_is_uncertain = False,
        opponent_selection = args.opponent_selection,
        other_reputation_threshold = o_r_t, #trial.suggest_categorical("other_reputation_threshold", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        cooperation_threshold = c_t, #trial.suggest_categorical("cooperation_threshold", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        b_value = args.b_value
    )

    if (args.algorithm == "reinforce"):
        algo_params = dict()
    elif (args.algorithm == "PPO"):
        algo_params = dict(
            K_epochs = 40, #trial.suggest_int("K_epochs", 30, 80),
            eps_clip = 0.2, #trial.suggest_float("eps_clip", 0.1, 0.4),
            gamma = trial.suggest_float("gamma", 0.99, 0.999, log=True),
            c1 = trial.suggest_float("c1", 0.01, 0.5, log=True),
            c2 = trial.suggest_float("c2", 0.0001, 0.1, log=True),
            c3 = trial.suggest_float("c3", 0.01, 0.5, log=True),
            c4 = trial.suggest_float("c4", 0.0001, 0.1, log=True),
            tau = trial.suggest_float("tau", 0.001, 0.5)
        )
    elif (args.algorithm == "dqn"):
        algo_params = dict(
            memory_size = trial.suggest_int("memory_size", 100, 1000)
        )

    print("args.communicating_agents.count(1.)=",args.communicating_agents.count(1.))
    if (args.communicating_agents.count(1.) != 0):
        comm_params = dict(
            n_hidden_comm = 2,
            hidden_size_comm = trial.suggest_categorical("hidden_size_comm", [8, 16, 32, 64]),
            lr_actor_comm = trial.suggest_float("lr_actor_comm", 1e-4, 1e-1, log=True),
            lr_critic_comm = trial.suggest_float("lr_critic_comm", 1e-4, 1e-1, log=True),
            mex_size = 5,
            sign_lambda = trial.suggest_float("sign_lambda", -5.0, 5.0),
            list_lambda = trial.suggest_float("list_lambda", -5.0, 5.0)
        )
    else: 
        comm_params = dict()

    all_params = {**game_params, **algo_params, **comm_params}
    print("all_params=", all_params)

    return all_params

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

def objective(trial, args, repo_name):

    all_params = setup_training_hyperparams(trial, args)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)#, sync_tensorboard=True)
    config = wandb.config
    print("config=", config)
    print("binary_reputation=", config.binary_reputation)

    parallel_env = pgg_parallel_v0.parallel_env(config)
    max_values = find_max_min(config, 4)

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
        #print("\n")
        #print("OPPONENT SELECTION")
        active_agents_idxs = []
        while ((any(active_agents_idxs) == True) == False):
            if (config.opponent_selection == True):
                first_agent_idx = random.sample(range(config.n_agents), 1)[0]
                #print("first_agent_idx=", first_agent_idx)
                reputations = torch.Tensor([agent.reputation for ag_idx, agent in agents.items()])
                #print("reputations=", reputations)
                second_agent_idx = int(agents["agent_"+str(first_agent_idx)].select_opponent(reputations))
                while (second_agent_idx == first_agent_idx):
                    second_agent_idx = int(agents["agent_"+str(first_agent_idx)].select_opponent(reputations))
                #print("second_agent_idx=", second_agent_idx)
                active_agents_idxs = [first_agent_idx, second_agent_idx]
            else:
                active_agents_idxs = random.sample(range(config.n_agents), 2)
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
                mf = parallel_env.current_multiplier
                #print("\nmf=", mf.numpy()[0])

                messages = {}; actions = {}
                
                #for idx in active_agents_idxs:
                #    print("agent=",idx, "rep=", active_agents["agent_"+str(idx)].reputation)
                
                active_agents["agent_"+str(active_agents_idxs[0])].digest_input((observations["agent_"+str(active_agents_idxs[0])], active_agents["agent_"+str(active_agents_idxs[1])].reputation))
                active_agents["agent_"+str(active_agents_idxs[1])].digest_input((observations["agent_"+str(active_agents_idxs[1])], active_agents["agent_"+str(active_agents_idxs[0])].reputation))

                # speaking
                #print("\nspeaking")
                for agent in parallel_env.active_agents:
                    if (agents[agent].is_communicating):
                        messages[agent] = agents[agent].select_message(m_val=mf.numpy()[0]) # m value is given only to compute metrics
                    #print("messages=", messages)

                # listening
                #print("\nlistening")
                for agent in parallel_env.active_agents:
                    other = list(set(active_agents_idxs) - set([active_agents[agent].idx]))[0]
                    if (active_agents[agent].is_listening and len(messages) != 0):
                        active_agents[agent].get_message(messages["agent_"+str(other)])
                    if (active_agents[agent].is_listening and n_communicating_agents != 0 and len(messages) == 0):
                        message = torch.zeros(config.mex_size)
                        active_agents[agent].get_message(message)

                # acting
                #print("\nacting")
                for agent in parallel_env.active_agents:
                    actions[agent] = agents[agent].select_action(m_val=mf.numpy()[0]) # m value is given only to compute metrics
                #print("\nactions=", actions)
                observations, rewards, done, _ = parallel_env.step1(actions)
                #print("rewards=", rewards)

                if (mf > 1. and mf < 2.):
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
                if done:
                    break
            
        #print("act_batch=", act_batch)
        print("Avg act agent0=", np.mean(act_batch["agent_0"]))
        print("Avg act agent1=", np.mean(act_batch["agent_1"]))
        #print("last rewards=",rewards)

        #print("update reputation")
        #print("active_agents_idxs=",active_agents_idxs)
        #print("avg saved_actions=",[np.mean(social_norm.saved_actions[ag_idx]) for ag_idx in active_agents_idxs])
        if (config.binary_reputation == True):
            social_norm.rule09_binary(active_agents_idxs)
        else:    
            social_norm.rule09(active_agents_idxs)

        for ag_idx in active_agents_idxs:       
            agents["agent_"+str(ag_idx)].old_reputation = agents["agent_"+str(ag_idx)].reputation

        #if agents["agent_1"].reputation == 0.0:
        #    print("\n\n\n\n\n\n\n\n\n\n\nERROR!!!!!!!!!!!!!!!")

        #print("NEW REPUTATIONS=", "agent_0 = ", agents["agent_0"].reputation, ", agent_1 = ", agents["agent_1"].reputation)

        for ag_idx, agent in active_agents.items():
            if (agent.is_dummy == False and agent.is_communicating):
                #print("is communicating")
                for m_val in config.mult_fact:
                    if (m_val in agent.buffer.actions_given_m):
                        if (m_val in agent.sc_m):
                            agent.sc_m[m_val].append(U.calc_mutinfo(agent.buffer.actions_given_m[m_val], agent.buffer.messages_given_m[m_val], config.action_size, config.mex_size))
                        else:
                            agent.sc_m[m_val] = [U.calc_mutinfo(agent.buffer.actions_given_m[m_val], agent.buffer.messages_given_m[m_val], config.action_size, config.mex_size)]
                agent.sc.append(U.calc_mutinfo(agent.buffer.actions, agent.buffer.messages, config.action_size, config.mex_size))
            if (agent.is_dummy == False and agent.is_listening):
                #print("is listening")
                other = list(set(active_agents_idxs) - set([agent.idx]))[0] #active_agents_idxs.pop("agent_"+str(ag_idx))
                if (active_agents["agent_"+str(other)].is_communicating):
                    agent.mutinfo_listening.append(U.calc_mutinfo(agent.buffer.actions, agents['agent_'+str(other)].buffer.messages, config.action_size, config.mex_size))
           
        # update agents     
        for ag_idx, agent in active_agents.items():
            #print("update", ag_idx)
            agent.update()

        # =================== EVALUATION ===================
        # computing evaluation against the behavior of all the dummy agents
        #print("EVALUATION")
        rewards_eval_m = {}; rewards_eval_norm_m = {}; actions_eval_m = {}; mex_distrib_given_m = {}
        optimization_measure = []

        for m in config.mult_fact:
            act_eval, mex_distrib, act_distrib, rewards_eval = eval(config, parallel_env, active_agents, active_agents_idxs, m, device, False)
            mex_distrib_given_m[m] = mex_distrib # distrib dei messaggi per ogni agente, calcolata con dato input
            rewards_eval_m[m] = rewards_eval
            rewards_eval_norm_m[m] = {key: value/max_values[m] for key, value in rewards_eval.items()}
            actions_eval_m[m] = act_eval
            #print("act_distrib=", act_distrib)
        #print("act eval=", act_eval)
        #print("distrib actions learning agent:", act_distrib)

        #print("[agent.return_episode_old_norm.numpy() for _, agent in active_agents.items()]=",[agent.return_episode_old_norm.numpy().item() for _, agent in active_agents.items()])
        #print("np.mean([agent.return_episode_old_norm.numpy() for _, agent in active_agents.items()])=", np.mean([agent.return_episode_old_norm.numpy().item() for _, agent in active_agents.items()]))
        avg_norm_return = np.mean([agent.return_episode_old_norm.numpy().item() for _, agent in active_agents.items()])
        #print("avg_norm_return=",avg_norm_return)
        avg_norm_returns_train_list.append(avg_norm_return)

        rew_values = [(np.sum([rewards_eval_m[m_val][ag_idx] for m_val in config.mult_fact])) for ag_idx, _ in active_agents.items()]
        avg_rew_time.append(np.mean(rew_values))
        #print(avg_rew_time)
        measure = np.mean(avg_rew_time[-10:])
        #for ag_idx, agent in agents.items():
        #    if (agent.is_dummy == False):
        #        print("rep agent=", ag_idx, agent.reputation)
        avg_rep = np.mean([agent.reputation for ag_idx, agent in agents.items() if (agent.is_dummy == False)])
        #print("avg_rep=", avg_rep)
        measure = avg_rep
        #print("measure=", measure)
        trial.report(measure, epoch)
        
        if trial.should_prune():
            print("is time to pruneee")
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        if (config.wandb_mode == "online"):
            for ag_idx, agent in active_agents.items():
                if (agent.is_dummy == False):
                    df_actions = {ag_idx+"actions_eval_m_"+str(i): actions_eval_m[i][ag_idx] for i in config.mult_fact}
                    df_rew = {ag_idx+"rewards_eval_m"+str(i): rewards_eval_m[i][ag_idx] for i in config.mult_fact}
                    df_rew_norm = {ag_idx+"rewards_eval_norm_m"+str(i): rewards_eval_norm_m[i][ag_idx] for i in config.mult_fact}
                    df_agent = {**{
                        ag_idx+"_return_train_norm": agent.return_episode_old_norm,
                        ag_idx+"_reputation": agent.reputation,
                        ag_idx+"_return_train_"+str(mf[0]): agent.return_episode_old,
                        #ag_idx+"gmm_means": agent.means,
                        #ag_idx+"gmm_probabilities": agent.probs,
                        'epoch': epoch}, 
                        **df_actions, **df_rew, **df_rew_norm}
                else:
                    df_actions = {ag_idx+"N_actions_eval_m_"+str(i): actions_eval_m[i][ag_idx] for i in config.mult_fact}
                    df_rew = {ag_idx+"N_rewards_eval_m"+str(i): rewards_eval_m[i][ag_idx] for i in config.mult_fact}
                    df_rew_norm = {ag_idx+"N_rewards_eval_norm_m"+str(i): rewards_eval_norm_m[i][ag_idx] for i in config.mult_fact}
                    df_agent = {**{
                        ag_idx+"N_return_train_norm": agent.return_episode_old_norm,
                        ag_idx+"N_reputation": agent.reputation,
                        ag_idx+"N_return_train_"+str(mf[0]): agent.return_episode_old,
                        #ag_idx+"gmm_means": agent.means,
                        #ag_idx+"gmm_probabilities": agent.probs,
                        'epoch': epoch}, 
                        **df_actions, **df_rew, **df_rew_norm}
                
                if (config.communicating_agents[agent.idx] == 1. and agent.is_dummy == False):
                    df_mex = {ag_idx+"messages_prob_distrib_m_"+str(i): mex_distrib_given_m[i][ag_idx] for i in config.mult_fact}
                    df_sc = {ag_idx+"sc": agent.sc_old[-1]}
                    df_sc_m = {ag_idx+"sc_given_m"+str(i): agent.sc_m[i][0] for i in config.mult_fact}
                    df_ent = {ag_idx+"_avg_mex_entropy": np.mean(agent.buffer.comm_entropy)}
                    df_agent = {**df_agent, **df_mex, **df_sc, **df_ent, **df_sc_m}
                if (config.listening_agents[agent.idx] == 1. and agent.is_dummy == False):
                    other = list(set(active_agents_idxs) - set([agent.idx]))[0]
                    if (active_agents["agent_"+str(other)].is_communicating):
                        #df_listen = {ag_idx+"mutinfo_listening": agent.mutinfo_listening_old[-1]}
                        df_agent = {**df_agent}#, **df_listen}
                if ('df_agent' in locals() ):
                    wandb.log(df_agent, step=epoch, commit=False)

            wandb.log({
                "epoch": epoch,
                "avg_rep": avg_rep,
                "current_multiplier": mf,
                "avg_return_train": avg_norm_return,
                "avg_return_train_time": np.mean(avg_norm_returns_train_list[-10:]),
                "avg_rew_time": measure,
                ###"avg_loss": np.mean([agent.saved_losses[-1] for _, agent in active_agents.items()]),
                #"avg_loss_comm": np.mean([agent.saved_losses_comm[-1] for _, agent in agents.items()]),
                },
                step=epoch, commit=True)

        if (epoch%10 == 0):
            print("Epoch : {} \t Measure: {} ".format(epoch, measure))
    
    wandb.finish()
    return measure


def training_function(args):

    name_gmm = "_noGmm"
    if (args.gmm_ == 1):
        name_gmm = "_yesGmm"
    
    comm_string = "no_comm_"
    if (args.communicating_agents.count(1.) != 0):
        comm_string = "comm_"
    unc_string = "no_unc_"
    if (args.uncertainties.count(0.) != args.n_agents):
        unc_string = "unc_"

    repo_name = "ANAST_"+ str(args.n_agents) + "agents_" + comm_string + \
        unc_string + args.algorithm + "_dummy_population_" + str(args.proportion_dummy_agents) + "_b" + str(args.b_value)
    
    if (args.addition != ""):
        repo_name += "_"+ str(args.addition)
    print("repo_name=", repo_name)


    func = lambda trial: objective(trial, args, repo_name)

    # sql not optimized for paralel sync
    #storage = optuna.storages.RDBStorage(url="sqlite:///"+repo_name+"-db") 

    storage = JournalStorage(JournalFileStorage("optuna-journal"+repo_name+".log"))

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
        objective(study.best_trial, args, repo_name+"_BEST")
