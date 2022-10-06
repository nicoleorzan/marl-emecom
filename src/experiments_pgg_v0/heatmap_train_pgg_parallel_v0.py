from src.environments import pgg_parallel_v0_bis
from src.algos.PPO import PPO
from src.nets.ActorCritic import ActorCritic
import numpy as np
import torch
import wandb
import json
import pandas as pd
import os
import src.analysis.utils as U


hyperparameter_defaults = dict(
    n_experiments = 100,
    episodes_per_experiment = 500,
    update_timestep = 100,       # update policy every n timesteps
    n_agents = 3,
    unc = 2.0, #[0.1, 0.2, 0.5, 1.],
    coins_mean = 4,
    mult_factors = [0., 0.1, 0.2, 0.5, 0.7, 1., 2., 3., 6., 8., 10.],
    num_game_iterations = 1,
    obs_size = 2,                # we observe coins we have, and multiplier factor with uncertainty
    hidden_size = 50,
    action_size = 2,
    K_epochs = 40,               # update policy for K epochs
    eps_clip = 0.2,              # clip parameter for PPO
    gamma = 0.99,                # discount factor
    c1 = -0.3,
    c2 = 0.1,
    lr_actor = 0.1,              # learning rate for actor network
    lr_critic = 0.001,           # learning rate for critic network
    decayRate = 0.99,
    fraction = False,
    comm = False,
    plots = False,
    save_models = False,
    save_data = True,
    save_interval = 20,
    print_freq = 3000,
    recurrent = False,
    random_baseline = False,
    wandb_mode = "offline"
)


wandb.init(project="HMP_pgg_v0_parallel", entity="nicoleorzan", config=hyperparameter_defaults, mode=hyperparameter_defaults["wandb_mode"])
config = wandb.config

folder = str(config.n_agents)+"agents/heatmap/"+"variating_m/"

path = "data/pgg_v0/"+folder
if not os.path.exists(path):
    os.makedirs(path)
    print("New directory is created!")

print("path=", path)

with open(path+'params.json', 'w') as fp:
    json.dump(hyperparameter_defaults, fp)

def train(config):
    print("uncertainty=", config.unc)
    parallel_env = pgg_parallel_v0_bis.parallel_env(n_agents=config.n_agents, num_iterations=config.num_game_iterations)

    if (config.save_data == True):
        df = pd.DataFrame(columns=['experiment', 'episode', 'mult_factor', 'uncertainty'] + \
            ["ret_ag"+str(i) for i in range(config.n_agents)] + \
            ["coop_ag"+str(i) for i in range(config.n_agents)])

    for experiment in range(len(config.mult_factors)):
        
        print("Mult_factor=", config.mult_factors[experiment])
        for times in range(config.n_experiments):
            #print("\nExperiment ", experiment)

            agents_dict = {}
            for idx in range(config.n_agents):
                model = ActorCritic(config, config.obs_size, config.action_size)
                optimizer = torch.optim.Adam([{'params': model.actor.parameters(), 'lr': config.lr_actor},
                        {'params': model.critic.parameters(), 'lr': config.lr_critic} ])

                agents_dict['agent_'+str(idx)] = PPO(model, optimizer, config)

            #### TRAINING LOOP
            # TVB
            avg_coop_time = []
            for ep_in in range(config.episodes_per_experiment):
                #print("\nEpisode=", ep_in)

                observations = parallel_env.reset('uniform', config.unc, config.mult_factors[experiment])
                #print("obs=", observations)
                    
                [agent.reset() for _, agent in agents_dict.items()]

                done = False
                while not done:

                    actions = {agent: agents_dict[agent].select_action(observations[agent]) for agent in parallel_env.agents}
                    
                    observations, rewards, done, _ = parallel_env.step(actions)

                    for ag_idx, agent in agents_dict.items():
                        
                        agent.buffer.rewards.append(rewards[ag_idx])
                        agent.buffer.is_terminals.append(done)
                        agent.tmp_return += rewards[ag_idx]
                        if (actions[ag_idx] is not None):
                            agent.tmp_actions.append(actions[ag_idx])
                        if done:
                            agent.train_returns.append(agent.tmp_return)
                            agent.coop.append(np.mean(agent.tmp_actions))

                    # break; if the episode is over
                    if done:
                        break

                if (ep_in+1) % config.print_freq == 0:
                    print("Experiment : {} \t Episode : {} \t Mult factor : {} \t Iters: {} ".format(experiment, \
                        ep_in, parallel_env.current_multiplier, config.num_game_iterations))
                    print("Episodic Reward:")
                    for ag_idx, agent in agents_dict.items():
                        print("Agent=", ag_idx, "action=", actions[ag_idx], "rew=", rewards[ag_idx])

                # update PPO agents
                if ep_in != 0 and ep_in % config.update_timestep == 0:
                    for ag_idx, agent in agents_dict.items():
                        agent.update()

                if (ep_in%config.save_interval == 0):

                    avg_coop_time.append(np.mean([np.mean(agent.tmp_actions) for _, agent in agents_dict.items()]))
                    if (config.wandb_mode == "online"):
                        for ag_idx, agent in agents_dict.items():
                            wandb.log({ag_idx+"_return": agent.tmp_return}, step=ep_in)
                            wandb.log({ag_idx+"_coop_level": np.mean(agent.tmp_actions)}, step=ep_in)
                        wandb.log({"episode": ep_in}, step=ep_in)
                        wandb.log({"avg_return": np.mean([agent.tmp_return for _, agent in agents_dict.items()])}, step=ep_in)
                        wandb.log({"avg_coop": avg_coop_time[-1]}, step=ep_in)
                        wandb.log({"avg_coop_time": np.mean(avg_coop_time[-10:])}, step=ep_in)

                if (ep_in==config.episodes_per_experiment-1):
                    if (config.save_data == True):
                        df_ret = {"ret_ag"+str(i): agents_dict["agent_"+str(i)].tmp_return.item() for i in range(config.n_agents)}
                        df_coop = {"coop_ag"+str(i): np.mean(agents_dict["agent_"+str(i)].tmp_actions) for i in range(config.n_agents)}
                        df_avg_coop = {"avg_coop": avg_coop_time[-1]}
                        df_avg_coop_time = {"avg_coop_time": np.mean(avg_coop_time[-10:])}
                        df_dict = {**{'experiment': experiment, 'episode': ep_in, 'mult_factor': parallel_env.current_multiplier, 'uncertainty':config.unc}, **df_ret, **df_coop, **df_avg_coop, **df_avg_coop_time}
                        df = pd.concat([df, pd.DataFrame.from_records([df_dict])])

        #end loop times
        if (config.plots == True):
            ### PLOT TRAIN RETURNS
            U.plot_train_returns(config, agents_dict, path, "train_returns_pgg")

            # COOPERATIVITY PERCENTAGE PLOT
            U.cooperativity_plot(config, agents_dict, path, "train_cooperativeness")

    if (config.save_data == True):
        if (config.random_baseline == True):
            df.to_csv(path+'heatmap_data_RND_unc'+str(config.unc)+'.csv')
        else: 
            df.to_csv(path+'heatmap_data_unc'+str(config.unc)+'.csv')

    # save models
    if (config.save_models == True):
        print("Saving models...")
        for ag_idx, ag in agents_dict.items():
            torch.save(ag.policy.state_dict(), path+"model_"+str(ag_idx))


if __name__ == "__main__":
    train(config)
