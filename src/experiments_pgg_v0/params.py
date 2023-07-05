EPOCHS = 400 # total episodes
# batch size are the number of episodes in which 2 agents interact with each other alone
OBS_SIZE = 3 # input: multiplication factor (with noise), opponent reputation.
# the opponent index is embedded in the agent class
ACTION_SIZE = 2
RANDOM_BASELINE = False

params = {

}

def setup_training_hyperparams(args, trial):

    if (args.optuna_ == 1): 
        lr_a = trial.suggest_float("lr_actor", 1e-4, 1e-1, log=True) #0.002 reinforce, 0.0002 PPO mixed motive
        lr_c = trial.suggest_float("lr_critic", 1e-4, 1e-1, log=True) #0.017 reinforce , #0.001 mixed motive PPO
        c_1 = trial.suggest_float("c1", 0.001, 0.9, log=True)
        c_2 = trial.suggest_float("c1", 0.001, 0.9, log=True)
        if (args.opponent_selection == 1):
            lr_opp = trial.suggest_float("lr_opponent", 1e-3, 1e-1, log=True) # 0.007 reinforce, 0 midex motive PPO
        else: 
            lr_opp = 0 
    else: 
        lr_a = 0.0002
        lr_c = 0.001 
        c_1 = 0.5
        c_2 = 0.01
        lr_opp = 0
    

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
        n_gmm_components = args.mult_fact,
        decayRate = 0.999,
        mult_fact = args.mult_fact,
        uncertainties = args.uncertainties,
        gmm_ = args.gmm_,
        random_baseline = RANDOM_BASELINE,
        communicating_agents = args.communicating_agents,
        listening_agents = args.listening_agents,
        batch_size = 128,
        lr_actor = lr_a,
        lr_critic = lr_c, 
        lr_opponent = lr_opp, 
        n_hidden_act = 1,
        hidden_size_act = 8, #trial.suggest_categorical("hidden_size_act", [8, 16, 32, 64]),
        embedding_dim = 1,
        binary_reputation = args.binary_reputation,
        get_index = False,
        get_opponent_is_uncertain = False,
        opponent_selection = args.opponent_selection,
        other_reputation_threshold = o_r_t,
        cooperation_threshold = c_t,
        optuna_ = args.optuna_
    )

    if (args.algorithm == "reinforce"):
        algo_params = dict()
    elif (args.algorithm == "PPO"):
        algo_params = dict(
            K_epochs = 40,
            eps_clip = 0.2,
            gamma = 0.99,
            c1 = c_1,
            c2 = c_2,
            c3 = 0, #trial.suggest_float("c3", 0.01, 0.5, log=True),
            c4 = 0 #, #trial.suggest_float("c4", 0.0001, 0.1, log=True),
        )
    elif (args.algorithm == "dqn"):
        algo_params = dict(
            memory_size = trial.suggest_int("memory_size", 100, 1000)
        )

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