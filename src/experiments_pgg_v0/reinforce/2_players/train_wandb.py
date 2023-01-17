import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import wandb
from train import train

hyperparameter_defaults = dict(
    n_experiments = 1,
    episodes_per_experiment = 40000,
    update_timestep = 128,        # update policy every n timesteps: same as batch side in this case
    n_agents = 2,
    uncertainties = [0.,0.],
    mult_fact = [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.],
    num_game_iterations = 1,
    obs_size = 2,                 # we observe coins we have, and multiplier factor with uncertainty
    hidden_size = 8,              # power of two!
    n_hidden = 1,
    action_size = 2,
    lr_actor = 0.01,              # learning rate for actor network
    lr_critic = 0.01,             # learning rate for critic network
    decayRate = 0.995,
    comm = False,
    save_models = False,
    random_baseline = False,
    wandb_mode = "online",
    gmm_ = False
)

wandb.init(project="new_2_agents_reinforce_pgg_v0", entity="nicoleorzan", config=hyperparameter_defaults, mode=hyperparameter_defaults["wandb_mode"])
config = wandb.config
print("config=", config)


if __name__ == "__main__":
    train(config)