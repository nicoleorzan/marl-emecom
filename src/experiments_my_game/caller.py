import argparse
import ast
from src.experiments_my_game.train_reinforce import train_reinforce
from src.experiments_my_game.train_q_learning import train_q_learning
from src.experiments_my_game.train_q_learning_reputation_assignment import train_q_learning_reputation_assignment
from src.experiments_my_game.train_dqn import train_dqn
from src.experiments_my_game.train_dqn_comm import train_dqn_comm
from src.experiments_my_game.train_dqn_reputation_assignment import train_dqn_reputation_assignment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int)
    parser.add_argument(
        "--mult_fact",
        nargs="*",
        type=float,
        default=[1.5])
    parser.add_argument( # to fill with values of uncertainties for every agent (can be 0.)
        "--uncertainties",
        nargs="*",
        type=float,
        default=[])
    parser.add_argument(
        "--communicating_agents", # to fill with 0 if agent does not comm, 1 is agent does comm
        nargs="*",
        type=int,
        default=[])
    parser.add_argument(
        "--listening_agents",
        nargs="*",
        type=int,
        default=[]) # to fill with 0 if agent does not list, 1 is agent does list
        
    parser.add_argument('--gmm_', type=int, default=0)
    parser.add_argument('--optuna_', type=int, default=0)
    parser.add_argument('--num_game_iterations', type=int, default=1)
    parser.add_argument('--reputation_in_reward', type=int, default=0)
    parser.add_argument('--reputation_enabled', type=int, default=0)
    parser.add_argument('--obs_size', type=int, default=4)
    parser.add_argument('--freq_counts', type=int, default=1)  # 1 for true 0 for false
    parser.add_argument('--coins_value', type=float, default=4.)
    parser.add_argument('--proportion_dummy_agents', type=float, default=0.)
    parser.add_argument('--binary_reputation', type=int, default=1) # 1 yes 0 no
    parser.add_argument('--wandb_mode', type=str, choices = ["online", "offline"], default="offline")
    parser.add_argument('--dqn_activation_function', type=str, choices = ["tanh", "relu"], default="tanh")
    parser.add_argument('--rule', type=str, choices = ["rule09", "rule11", "rule03", "rule00"], default="rule09")
    parser.add_argument('--algorithm', type=str, choices = ["reinforce", "PPO", "dqn", "q-learning"], default="reinforce")
    parser.add_argument('--random_baseline', type=str, default="False")
    parser.add_argument('--optimize', type=int, default=0) # 1 for true 0 for false
    parser.add_argument('--comm', type=int, default=0) # 1 for true 0 for false
    parser.add_argument('--threshold', type=float, default=0.9) # 1 for true 0 for false
    parser.add_argument('--mex_size', type=int, default=2)
    parser.add_argument('--mf_from_interval', type=int, default=0) # 1 for true 0 for false
    parser.add_argument('--introspective', type=int, default=0) # 1 for true 0 for false
    parser.add_argument('--continuous_games', type=int, default=0) # 1 for true 0 for false
    parser.add_argument('--opponent_selection', type=int, default=0) # 1 for true 0 for false
    parser.add_argument('--reputation_assignment', type=int, default=0) # 1 for true 0 for false
    parser.add_argument('--addition', type=str, default="")

    args = parser.parse_args()
    args.random_baseline = ast.literal_eval(args.random_baseline)
    n_certain_agents = args.uncertainties.count(0.)
    n_uncertain = args.n_agents - n_certain_agents
    
    assert(args.proportion_dummy_agents >= 0.)    
    assert(args.proportion_dummy_agents <= 1.)

    assert(args.n_agents > 1)
    if (args.algorithm == "q-learning"):
        assert(n_certain_agents == args.n_agents)
    assert(len(args.uncertainties) == args.n_agents)
    assert(len(args.communicating_agents) == args.n_agents)
    assert(len(args.listening_agents) == args.n_agents)
    if (args.reputation_enabled == 0):
        assert(args.proportion_dummy_agents == 0)

    if args.algorithm == "dqn":
        if (args.comm == True): 
            train_dqn_comm(args)
        if (args.reputation_assignment == True):
            if (args.reputation_enabled == True):
                train_dqn_reputation_assignment(args)
            else: 
                print("Reputation is not enabled but reputation assignment it. Fix it.")
        else:
            train_dqn(args)
    elif args.algorithm == "reinforce":
        train_reinforce(args)    
    elif args.algorithm == "q-learning":
        if (args.reputation_assignment == True):
            if (args.reputation_enabled == True):
                train_q_learning_reputation_assignment(args)
            else: 
                print("Reputation is not enabled but reputation assignment it. Fix it.")
        else:
            train_q_learning(args)
