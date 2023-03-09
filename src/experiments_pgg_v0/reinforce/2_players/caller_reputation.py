import argparse
import ast
from new_train import training_function

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
        "--communicating_agents", #to fill with 0 if agent does not comm, 1 is agent does comm
        nargs="*",
        type=int,
        default=[])
    parser.add_argument(
        "--listening_agents",
        nargs="*",
        type=int,
        default=[]) #to fill with 0 if agent does not list, 1 is agent does list
    parser.add_argument(
        "--gmm_",
        nargs="*",
        type=int,
        default=[])

    parser.add_argument('--partner_selection', type=int, default=0)
    parser.add_argument('--punishment', type=int, default=0)
    parser.add_argument('--random_baseline', type=int, default=0)
    parser.add_argument('--optimize', type=int) # 1 for true 0 for false

    args = parser.parse_args()
    n_certain_agents = args.uncertainties.count(0.)
    n_uncertain = args.n_agents - n_certain_agents

    assert(args.n_agents > 1)
    assert(len(args.gmm_) == args.n_agents)
    assert(len(args.uncertainties) == args.n_agents)
    assert(len(args.communicating_agents) == args.n_agents)
    assert(len(args.listening_agents) == args.n_agents)

    training_function(args)