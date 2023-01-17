import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import torch
import wandb
from train_comm import train

hyperparameter_defaults = dict(
    n_experiments = 1,
    episodes_per_experiment = 160000,
    update_timestep = 64,         # update policy every n timesteps
    n_agents = 2,
    uncertainties = [0., 1.],
    mult_fact = [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.],        # list givin min and max value of mult factor
    num_game_iterations = 1,
    obs_size = 2,                        # we observe coins we have, and multiplier factor with uncertainty
    action_size = 2,
    hidden_size_comm = 16,
    hidden_size_act = 64,
    n_hidden_comm = 2,
    n_hidden_act = 1,
    lr_actor = 0.01,             # learning rate for actor network
    lr_critic = 0.01,
    lr_actor_comm = 0.01,       # learning rate for actor network
    lr_critic_comm = 0.05,
    decayRate = 0.99,
    comm = True,
    save_models = False,
    mex_size = 3,
    random_baseline = False,
    wandb_mode ="online",
    sign_lambda = 0.,
    list_lambda = 0.,
    gmm_ = False
)

wandb.init(project="new_2_agents_reinforce_pgg_v0_comm_1_unc", entity="nicoleorzan", config=hyperparameter_defaults, mode=hyperparameter_defaults["wandb_mode"])#, sync_tensorboard=True)
config = wandb.config
print("config=", config)


if __name__ == "__main__":
    train(config)