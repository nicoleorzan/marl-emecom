import argparse
import ast
from train_optuna_dummy_pop import training_function

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int)
    parser.add_argument(
        "--mult_fact",
        nargs="*",
        type=float,
        default=[])
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
    parser.add_argument(
        "--gmm_",
        nargs="*",
        type=int,
        default=[])        
        
    parser.add_argument('--proportion_dummy_agents', type=float, default=0.)
    parser.add_argument('--binary_reputation', type=int) # 1 yes 0 no
    parser.add_argument('--wandb_mode', type=str, choices = ["online", "offline"], default="online")
    parser.add_argument('--algorithm', type=str, choices = ["reinforce", "PPO", "dqn"], default="reinforce")
    parser.add_argument('--random_baseline', type=str, default="False")
    parser.add_argument('--optimize', type=int) # 1 for true 0 for false
    parser.add_argument('--addition', type=str, default="")

    args = parser.parse_args()
    args.random_baseline = ast.literal_eval(args.random_baseline)
    n_certain_agents = args.uncertainties.count(0.)
    n_uncertain = args.n_agents - n_certain_agents
    
    assert(args.proportion_dummy_agents >= 0.)    
    assert(args.proportion_dummy_agents <= 1.)

    assert(args.n_agents > 1)
    assert(len(args.gmm_) == args.n_agents)
    assert(len(args.uncertainties) == args.n_agents)
    assert(len(args.communicating_agents) == args.n_agents)
    assert(len(args.listening_agents) == args.n_agents)

    training_function(args)
