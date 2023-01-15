import argparse
import ast
from train_comm import training_function

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--mult_fact",
    nargs="*",
    type=float,
    default=[])
    parser.add_argument('--episodes_per_experiment', type=int)
    parser.add_argument('--update_timestep', type=int)
    parser.add_argument('--n_agents', type=int)
    parser.add_argument('--n_uncertain', type=int)
    parser.add_argument('--uncertainty', type=float)
    parser.add_argument('--gmm_', type=str)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--lr_act', type=float)
    parser.add_argument('--lr_comm', type=float)
    parser.add_argument('--decay_rate', type=float)
    parser.add_argument('--mex_size', type=int)
    parser.add_argument('--sign_lambda', type=float)
    parser.add_argument('--list_lambda', type=float)
    parser.add_argument('--random_baseline', type=str, default="False")

    args = parser.parse_args()
    args.gmm_ = ast.literal_eval(args.gmm_)
    args.random_baseline = ast.literal_eval(args.random_baseline)
    print("args=", args)

    assert(args.n_uncertain <= args.n_agents)
    assert(args.n_agents > 1)
    assert(args.uncertainty >= 0.)
    assert(args.uncertainty <= 5.)
    assert(args.update_timestep >= 23)
    assert(args.decay_rate < 1.)

    if (args.n_uncertain != 0):
        repo_name = "new_"+str(args.n_agents)+"_agents_reinforce_pgg_v0_"+str(args.n_uncertain)+"_unc_comm"
    else:
        repo_name = "new_"+str(args.n_agents)+"_agents_reinforce_pgg_v0_comm"
    
    print("Setting up wandb repo:", repo_name)
    training_function(args, repo_name)