from src.environments import pgg_parallel_v0
from src.algos.Reinforce import Reinforce
from src.algos.DQN import DQN
from src.algos.PPO import PPO
import numpy as np
import optuna
from optuna.trial import TrialState
import torch
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
import src.analysis.utils as U
from src.experiments_pgg_v0.utils_train_reinforce import eval, find_max_min

EPOCHS = 600
OBS_SIZE = 1
ACTION_SIZE = 2
WANDB_MODE = "online"
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

    all_params = dict(
        n_agents = args.n_agents,
        algorithm = args.algorithm,
        num_game_iterations = 1,
        n_epochs = EPOCHS,
        obs_size = OBS_SIZE,
        action_size = ACTION_SIZE,
        n_gmm_components = trial.suggest_categorical("n_gmm_components", [3, len(args.mult_fact)]),
        decayRate = trial.suggest_categorical("decay_rate", [0.99, 0.999]),
        mult_fact = args.mult_fact,
        uncertainties = args.uncertainties,
        gmm_ = args.gmm_,
        random_baseline = RANDOM_BASELINE,
        communicating_agents = args.communicating_agents,
        listening_agents = args.listening_agents,
        batch_size = trial.suggest_categorical("batch_size", [64, 128]),
        lr_actor = trial.suggest_float("lr_actor", 1e-3, 1e-1, log=True),
        lr_critic = trial.suggest_float("lr_critic", 1e-3, 1e-1, log=True),
        lr_actor_comm = trial.suggest_float("lr_actor_comm", 1e-3, 1e-1, log=True),
        lr_critic_comm = trial.suggest_float("lr_critic_comm", 1e-3, 1e-1, log=True),
        n_hidden_act = trial.suggest_int("n_hidden_act", 1, 2),
        n_hidden_comm = trial.suggest_int("n_hidden_comm", 1, 2),
        K_epochs = trial.suggest_int("K_epochs", 30, 80),
        eps_clip = trial.suggest_float("eps_clip", 0.1, 0.4),
        gamma = trial.suggest_float("gamma", 0.99, 0.999, log=True),
        c1 = trial.suggest_float("c1", 0.01, 0.5, log=True),
        c2 = trial.suggest_float("c2", 0.0001, 0.1, log=True),
        c3 = trial.suggest_float("c3", 0.01, 0.5, log=True),
        c4 = trial.suggest_float("c4", 0.0001, 0.1, log=True),
        hidden_size_act = trial.suggest_categorical("hidden_size_act", [8, 16, 32, 64]),
        hidden_size_comm = trial.suggest_categorical("hidden_size_comm", [8, 16, 32, 64]),
        mex_size = trial.suggest_int("mex_size", 2, 10),
        memory_size = trial.suggest_int("memory_size", 100, 1000),
        sign_lambda = trial.suggest_float("sign_lambda", 0.1, 0.8),
        list_lambda = trial.suggest_float("list_lambda", 0.1, 0.8),
        tau = trial.suggest_float("tau", 0.001, 0.5),
        wandb_mode = WANDB_MODE
    )

    return all_params


def define_agents(config):
    agents = {}
    for idx in range(config.n_agents):
        if (config.algorithm == "reinforce"):
            agents['agent_'+str(idx)] = Reinforce(config, idx)
        elif (config.algorithm == "PPO"):
            agents['agent_'+str(idx)] = PPO(config, idx)
        elif (config.algorithm == "dqn"):
            agents['agent_'+str(idx)] = DQN(config, idx)
    return agents

def objective(trial, args, repo_name):

    all_params = setup_training_hyperparams(trial, args)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=WANDB_MODE)#, sync_tensorboard=True)
    config = wandb.config
    print("config=", config)

    parallel_env = pgg_parallel_v0.parallel_env(config)
    max_values = find_max_min(config, 4)
    idx_comm_agents = [j for j,val in enumerate(config.communicating_agents) if val==1]
    num_comm_agents = config.communicating_agents.count(1)

    agents = define_agents(config)

    #### TRAINING LOOP
    avg_norm_returns_train_list = []; avg_rew_time = []
    for epoch in range(config.n_epochs): 
        [agent.reset_batch() for _, agent in agents.items()]
        
        for _ in range(config.batch_size):

            observations = parallel_env.reset()
            
            [agent.reset_episode() for _, agent in agents.items()]

            done = False
            while not done:
                mf = parallel_env.current_multiplier
                #print("\n\nmf=", mf.numpy()[0])

                messages = {}; actions = {}
                [agents[agent].set_observation(observations[agent]) for agent in parallel_env.agents]

                # speaking
                #print("\nspeaking")
                for agent in parallel_env.agents:
                    if (agents[agent].is_communicating):
                        messages[agent] = agents[agent].select_message(m_val=mf.numpy()[0]) # m value is given only to compute metrics

                # listening
                #print("\nlistening")
                if (num_comm_agents != 0):
                    message = torch.stack([v for _, v in messages.items()]).view(-1).to(device)
                    [agents[agent].get_message(message) for agent in parallel_env.agents if (agents[agent].is_listening)]

                # acting
                for agent in parallel_env.agents:
                    actions[agent] = agents[agent].select_action(m_val=mf.numpy()[0]) # m value is given only to compute metrics
                
                observations, rewards, done, _ = parallel_env.step(actions)
                rewards_norm = {key: value/max_values[float(parallel_env.current_multiplier[0])] for key, value in rewards.items()}
                #print("rewards=", rewards)
                #print("rewards_norm=", rewards_norm)

                for ag_idx, agent in agents.items():
                    
                    agent.buffer.rewards.append(rewards[ag_idx])
                    agent.buffer.next_states_a.append(observations[ag_idx])
                    agent.buffer.is_terminals.append(done)
                    agent.return_episode_norm += rewards_norm[ag_idx]
                    agent.return_episode =+ rewards[ag_idx]

                # break; if the episode is over
                if done:
                    break

        for ag_idx, agent in agents.items():
            if (agent.is_communicating):
                #print("\n",agent.buffer.actions_given_m)
                for m_val in config.mult_fact:
                    if (m_val in agent.sc_m):
                        agent.sc_m[m_val].append(U.calc_mutinfo(agent.buffer.actions_given_m[m_val], agent.buffer.messages_given_m[m_val], config.action_size, config.mex_size))
                    else:
                        agent.sc_m[m_val] = [U.calc_mutinfo(agent.buffer.actions_given_m[m_val], agent.buffer.messages_given_m[m_val], config.action_size, config.mex_size)]
                #print("\n\n==========>SC_m:",agent.sc_m)
                agent.sc.append(U.calc_mutinfo(agent.buffer.actions, agent.buffer.messages, config.action_size, config.mex_size))
            if (agent.is_listening):
                for i in idx_comm_agents:
                    agent.mutinfo_listening.append(U.calc_mutinfo(agent.buffer.actions, agents['agent_'+str(i)].buffer.messages, config.action_size, config.mex_size))

        # update agents     
        for ag_idx, agent in agents.items():
            agent.update()
        
        mex_distrib_given_m = {}; rewards_eval_m = {}; rewards_eval_norm_m = {}; actions_eval_m = {}
        for m in config.mult_fact:
            act_eval, mex_distrib, _, rewards_eval = eval(config, parallel_env, agents, m, device, False)
            mex_distrib_given_m[m] = mex_distrib # distrib dei messaggei per ogni agente, calcolata con dato input
            rewards_eval_m[m] = rewards_eval
            rewards_eval_norm_m[m] = {key: value/max_values[m] for key, value in rewards_eval.items()}
            actions_eval_m[m] = act_eval

        avg_norm_return = np.mean([agent.return_episode_old_norm.numpy() for _, agent in agents.items()])
        avg_norm_returns_train_list.append(avg_norm_return)

        rew_values = [(np.sum([rewards_eval_m[m_val][ag_idx] for m_val in config.mult_fact])) for ag_idx, _ in agents.items()]
        
        avg_rew_time.append(np.mean(rew_values))
        #print(avg_rew_time)
        measure = np.mean(avg_rew_time[-10:])
        
        trial.report(measure, epoch)
        
        if trial.should_prune():
            print("is time to pruneee")
            wandb.finish()
            raise optuna.exceptions.TrialPruned()
            break

        if (config.wandb_mode == "online"):
            for ag_idx, agent in agents.items():
                df_actions = {ag_idx+"actions_eval_m_"+str(i): actions_eval_m[i][ag_idx] for i in config.mult_fact}
                df_rew = {ag_idx+"rewards_eval_m"+str(i): rewards_eval_m[i][ag_idx] for i in config.mult_fact}
                df_rew_norm = {ag_idx+"rewards_eval_norm_m"+str(i): rewards_eval_norm_m[i][ag_idx] for i in config.mult_fact}
                df_agent = {**{
                    ag_idx+"_return_train_norm": agent.return_episode_old_norm,
                    ag_idx+"_return_train_"+str(mf[0]): agent.return_episode_old,
                    ag_idx+"gmm_means": agent.means,
                    ag_idx+"gmm_probabilities": agent.probs,
                    'epoch': epoch}, 
                    **df_actions, **df_rew, **df_rew_norm}
                
                if (config.communicating_agents[agent.idx] == 1.):
                    df_mex = {ag_idx+"messages_prob_distrib_m_"+str(i): mex_distrib_given_m[i][ag_idx] for i in config.mult_fact}
                    df_sc = {ag_idx+"sc": agent.sc_old[-1]}
                    df_sc_m = {ag_idx+"sc_given_m"+str(i): agent.sc_m[i][0] for i in config.mult_fact}
                    df_ent = {ag_idx+"_avg_mex_entropy": np.mean(agent.buffer.comm_entropy)}
                    df_agent = {**df_agent, **df_mex, **df_sc, **df_ent, **df_sc_m}
                if (config.listening_agents[agent.idx] == 1.):
                    df_listen = {ag_idx+"mutinfo_listening": agent.mutinfo_listening_old[-1]}
                    df_agent = {**df_agent, **df_listen}
                
                wandb.log(df_agent, step=epoch, commit=False)

            wandb.log({
                "epoch": epoch,
                "current_multiplier": mf,
                "avg_return_train": avg_norm_return,
                "avg_return_train_time": np.mean(avg_norm_returns_train_list[-10:]),
                "avg_rew_time": measure,
                "avg_loss": np.mean([agent.saved_losses[-1] for _, agent in agents.items()]),
                #"avg_loss_comm": np.mean([agent.saved_losses_comm[-1] for _, agent in agents.items()]),
                },
                step=epoch, commit=True)

        if (epoch%10 == 0):
            print("Epoch : {} \t Mult factor: {}  \t Measure: {} ".format(epoch, mf, measure))
    
    wandb.finish()
    return measure


def training_function(args):

    name_gmm = "_noGmm"
    if (1 in args.gmm_):
        name_gmm = "_yesGmm"

    repo_name = str(args.n_agents) + "agents_" + "comm" + str(args.communicating_agents) + \
        "_list" + str(args.listening_agents) + name_gmm + "_unc" + str(args.uncertainties) + \
        "_mfact" + str(args.mult_fact) + args.algorithm
    print("repo_name=", repo_name)

    func = lambda trial: objective(trial, args, repo_name)

    storage = optuna.storages.RDBStorage(url="sqlite:///"+repo_name+"-db")

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
        study.optimize(func, n_trials=100, timeout=600)

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
