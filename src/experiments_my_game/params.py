import torch

EPOCHS = 4000 # total episodes
# batch size are the number of episodes in which 2 agents interact with each other alone
#OBS_SIZE = 3 # input: multiplication factor (with noise), opponent reputation.
#OBS_SIZE_ANAST = 4 # opponent previous act, my previous act, opponent reputation, my reputation
# the opponent index is embedded in the agent class
ACTION_SIZE = 2
RANDOM_BASELINE = False

DEVICE = torch.device('cpu')
if(torch.cuda.is_available()): 
    DEVICE = torch.device('cuda:0') 
    torch.cuda.empty_cache()

def setup_training_hyperparams(args, trial):

    all_params = {}

    if (args.optuna_ == 1):
        lr_a = trial.suggest_float("lr_actor", 1e-4, 1e-1, log=True) #0.002 reinforce, 0.0002 PPO mixed motive
        lr_c = trial.suggest_float("lr_critic", 1e-4, 1e-1, log=True) #0.017 reinforce , #0.001 mixed motive PPO
    else:
        lr_a = 0.001 # 0.002 reinforce 0.0002
        lr_c = 0.001  # 0.001

    if (args.optuna_ == 1):
        o_r_t = trial.suggest_categorical("other_reputation_threshold", [0.4, 0.5, 0.6, 0.8, 1.0])
        c_t = trial.suggest_categorical("cooperation_threshold", [0.4, 0.5, 0.6, 0.8, 1.0])
    else:
        o_r_t = 0.9
        c_t = 0.4

    if (args.communicating_agents.count(1.) != 0):
        if (args.optuna_ == 1):
            comm_params = dict(
                n_hidden_comm = 2,
                hidden_size_comm = trial.suggest_categorical("hidden_size_comm", [8, 16, 32, 64]),
                lr_actor_comm = trial.suggest_float("lr_actor_comm", 1e-4, 1e-1, log=True),
                lr_critic_comm = trial.suggest_float("lr_critic_comm", 1e-4, 1e-1, log=True),
                mex_size = 5,
                sign_lambda = trial.suggest_float("sign_lambda", -5.0, 5.0),
                list_lambda = trial.suggest_float("list_lambda", -5.0, 5.0))
        else:
            comm_params = dict(
                n_hidden_comm = 2,
                hidden_size_comm = 16,
                lr_actor_comm = 0.005,
                lr_critic_comm = 0.01,
                mex_size = 5,
                sign_lambda = 0,
                list_lambda = 0
            )
    else:
        comm_params = dict()

    game_params = dict(
        n_agents = args.n_agents,
        mult_fact = args.mult_fact,
        uncertainties = args.uncertainties,
        algorithm = args.algorithm,
        coins_value = args.coins_value,
        wandb_mode = args.wandb_mode,
        proportion_dummy_agents = args.proportion_dummy_agents,
        gmm_ = args.gmm_,
        communicating_agents = args.communicating_agents,
        listening_agents = args.listening_agents,
        get_index = False,
        get_opponent_is_uncertain = False,
        opponent_selection = args.opponent_selection,
        #num_game_iterations = args.num_game_iterations,
        n_epochs = EPOCHS,
        obs_size = args.obs_size,
        action_size = ACTION_SIZE,
        random_baseline = RANDOM_BASELINE,
        embedding_dim = 1,
        binary_reputation = args.binary_reputation,
        other_reputation_threshold = o_r_t,
        cooperation_threshold = c_t,
        optuna_ = args.optuna_,
        device = DEVICE,
        reputation_in_reward = args.reputation_in_reward
    )
    if hasattr(args, 'b_value'):
        all_params = {**all_params,
            **{"b_value": args.b_value, 
            "c_value": args.c_value,
            "d_value": args.d_value}
            }

    if (args.algorithm == "reinforce"):
        if (args.optuna_ == 1): 
            num_hidden_a = trial.suggest_categorical("n_hidden_act", [1, 2])
            hidden_size_a = trial.suggest_categorical("hidden_size_act", [8, 16, 32, 64])
        else:
            num_hidden_a = 1
            hidden_size_a = 4
        obs_size = 2 # m factor and reputation
        algo_params = dict(
            obs_size = obs_size,
            n_episodes = 200000,
            num_game_iterations = 200, # K 
            gamma = 0.99,
            chi = 0.0001,
            epsilon = 0.01,
            lr_actor = 0.01,
            n_hidden_act = num_hidden_a,
            hidden_size_act = hidden_size_a,
            decayRate = 0.999,
            alpha = 0.1, # introspection level
            reputation_enabled = args.reputation_enabled,
            introspective = False
        )
    elif (args.algorithm == "PPO"):
        if (args.optuna_ == 1):
            c_1 = trial.suggest_float("c1", 0.001, 0.9, log=True)
            c_2 = trial.suggest_float("c1", 0.001, 0.9, log=True)
            num_hidden_a = trial.suggest_categorical("n_hidden_act", [1, 2])
            hidden_size_a = trial.suggest_categorical("hidden_size_act", [8, 16, 32, 64])
        else:
            c_1 = 0.5
            c_2 = 0.01
            num_hidden_a = 2
            hidden_size_a = 16

        algo_params = dict(
            lr_actor = lr_a,
            lr_critic = lr_c, 
            n_hidden_act = num_hidden_a,
            hidden_size_act = hidden_size_a, 
            batch_size = 5,
            decayRate = 0.999,
            K_epochs = 40,
            eps_clip = 0.2,
            gamma = 0.99,
            c1 = c_1,
            c2 = c_2,
            c3 = 0, #trial.suggest_float("c3", 0.01, 0.5, log=True),
            c4 = 0  #trial.suggest_float("c4", 0.0001, 0.1, log=True),
        )
    elif (args.algorithm == "dqn"):
        obs_size = 2  # m factor and reputation
        algo_params = dict(
            obs_size = obs_size, # mult factor and reputation of opponent
            n_episodes = 10000,
            num_game_iterations = 200, #200, # K 
            gamma = 0.99,
            chi = 0.0001,
            epsilon = 0.01,
            memory_size = 500,
            n_hidden_act = 1,
            hidden_size_act = 4,
            lr_actor = 0.001,
            decayRate = 0.999,
            target_net_update_freq = 30,
            reputation_enabled = args.reputation_enabled,
            alpha = 0.1, # introspection level
            introspective = False,
            decaying_epsilon = True
        )
    elif (args.algorithm == "q-learning"):
        obs_size = 2  # m factor and reputation
        algo_params = dict(
            obs_size = obs_size,
            n_episodes = 400000,
            num_game_iterations = 10, # K 
            gamma = 0.99,
            chi = 0.001,
            epsilon = 0.01,
            lr_actor = 0.01,
            alpha = 0.1, # introspection level
            reputation_enabled = args.reputation_enabled,
            introspective = False
        )

    n_dummy = int(args.proportion_dummy_agents*args.n_agents)
    is_dummy = list(reversed([1 if i<n_dummy else 0 for i in range(args.n_agents) ]))
    non_dummy_idxs = [i for i,val in enumerate(is_dummy) if val==0]

    all_params = {**all_params, **game_params, **algo_params, **comm_params, "n_dummy":n_dummy, "is_dummy":is_dummy, "non_dummy_idxs":non_dummy_idxs}
    print("all_params=", all_params)

    return all_params