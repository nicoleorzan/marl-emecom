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
        o_r_t = 0.4
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
        algorithm = args.algorithm,
        wandb_mode = args.wandb_mode,
        num_game_iterations = args.num_game_iterations,
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
            num_hidden_a = 2
            hidden_size_a = 16

        algo_params = dict(
            lr_actor = lr_a,
            n_hidden_act = num_hidden_a,
            hidden_size_act = hidden_size_a,
            batch_size = 1,
            decayRate = 0.999
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
        algo_params = dict(
            memory_size = 500, #trial.suggest_int("memory_size", 100, 1000)
            n_hidden_act = 2,
            hidden_size_act = 16,
            batch_size = 5,
            lr_actor = lr_a,
            decayRate = 0.999
        )

    all_params = {**all_params, **game_params, **algo_params, **comm_params}
    print("all_params=", all_params)

    return all_params