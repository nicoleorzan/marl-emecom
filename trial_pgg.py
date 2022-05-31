from src.environments import pgg_v0
#import supersuit as ss
from PPO import PPO
import numpy as np
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
from utils import plot_hist_returns, plot_train_returns, cooperativity_plot, evaluation, plot_avg_on_experiments

hyperparameter_defaults = dict(
    n_experiments = 60,
    episodes_per_experiment = 800,
    update_timestep = 40, # update policy every n timesteps
    n_agents = 2,
    uncertainties = [2., 10.],#, 10.],
    coins_per_agent = 4,
    mult_fact = [0, 0.1, 0.2, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
    num_game_iterations = 5,
    action_space = 2,
    input_dim_agent = 2,         # we observe coins we have, and multiplier factor with uncertainty
    K_epochs = 40,               # update policy for K epochs
    eps_clip = 0.2,              # clip parameter for PPO
    gamma = 0.99,                # discount factor
    c1 = 0.5,
    c2 = -0.01,
    lr_actor = 0.001,            # learning rate for actor network
    lr_critic = 0.001,           # learning rate for critic network
    comm = False,
    plots = False
)

wandb.init(project="pgg", entity="nicoleorzan", config=hyperparameter_defaults, mode="offline")
config = wandb.config

assert (config.n_agents == len(config.uncertainties))

if hasattr(config.mult_fact, '__len__'):
    folder = 'coop_variating_m/'#='+str(config.mult_fact)+'/'
else: 
    folder = 'coop_'+str(config.mult_fact)+'/'    

max_ep_len = 1                    # max timesteps in one episode
num_blocks = 10                   # number of blocks for moving average

print_freq = 100     # print avg reward in the interval (in num timesteps)


def evaluate_episode(agents_dict, agent_to_idx):
    env = pgg_v0.env(n_agents=config.n_agents, coins_per_agent=config.coins_per_agent, num_iterations=config.num_game_iterations, \
        mult_fact=config.mult_fact, uncertainties=config.uncertainties)
    env.reset()
    i = 0
    ag_rets = np.zeros(len(agents_dict))

    for id_agent in env.agent_iter():
        idx = agent_to_idx[id_agent]
        obs, reward, done, _ = env.last()
        acting_agent = agents_dict[id_agent]
        act = acting_agent.select_action(obs) if not done else None
        env.step(act)
        ag_rets[idx] += reward
        i += 1

        if (done and idx == config.n_agents-1):  
            break

    env.close()
    return ag_rets


def train(config):

    n_agents = config.n_agents

    env = pgg_v0.env(n_agents=config.n_agents, coins_per_agent=config.coins_per_agent, num_iterations=config.num_game_iterations, \
    mult_fact=config.mult_fact, uncertainties=config.uncertainties)

    all_returns = np.zeros((n_agents, config.n_experiments, config.episodes_per_experiment))
    all_cooperativeness = np.zeros((n_agents, config.n_experiments, config.episodes_per_experiment))
    average_returns = np.zeros((n_agents, config.episodes_per_experiment))
    average_cooperativeness = np.zeros((n_agents, config.episodes_per_experiment))
    print("start")

    for experiment in range(config.n_experiments):

        agents_dict = {}
        un_agents_dict = {}
        agent_to_idx = {}
        for idx in range(config.n_agents):
            agents_dict['agent_'+str(idx)] = PPO(config.input_dim_agent, config.action_space, config.lr_actor, config.lr_critic,  \
            config.gamma, config.K_epochs, config.eps_clip, config.c1, config.c2)
            un_agents_dict['agent_'+str(idx)] = PPO(config.input_dim_agent, config.action_space, config.lr_actor, config.lr_critic,  \
            config.gamma, config.K_epochs, config.eps_clip, config.c1, config.c2)
            agent_to_idx['agent_'+str(idx)] = idx

        if (config.plots == True):
            print("\nEVALUATION BEFORE LEARNING")
            rews_before = evaluation(un_agents_dict, config.eval_eps, agent_to_idx)

        #### TRAINING LOOP
        time_step = 0
        i_episode = 0

        # printing and logging variables
        print_running_reward = np.zeros(config.n_agents)
        print_running_episodes = np.zeros(config.n_agents)

        for ep_in in range(config.episodes_per_experiment):

            env.reset()

            current_ep_reward = np.zeros(config.n_agents)
            i_internal_loop = 0
                
            for ag_idx in range(config.n_agents):
                agents_dict['agent_'+str(ag_idx)].tmp_return = 0
                agents_dict['agent_'+str(ag_idx)].tmp_actions = []

            for id_agent in env.agent_iter():
                #print("\nTime=", time_step)
                idx = agent_to_idx[id_agent]
                acting_agent = agents_dict[id_agent]
                #print("idx=", idx, "agent_to_idx=", agent_to_idx[id_agent], "id_agent=", id_agent)
                
                obs, rew, done, _ = env.last()
                #print(obs, rew, done, info)
                act = acting_agent.select_action(obs) if not done else None
                #print("act=", act)
                #print("env step")
                env.step(act)

                if (i_internal_loop > config.n_agents-1):
                    acting_agent.buffer.rewards.append(rew)
                    acting_agent.buffer.is_terminals.append(done)
                    acting_agent.tmp_return += rew
                    if (act is not None):
                        acting_agent.tmp_actions.append(act)

                time_step += 1

                if rew != None:
                    current_ep_reward[idx] += rew

                # break; if the episode is over
                if (done):  
                    acting_agent.train_returns.append(acting_agent.tmp_return)
                    acting_agent.cooperativeness.append(np.mean(acting_agent.tmp_actions))
                    if (idx == config.n_agents-1):
                        break

                i_internal_loop += 1

            if time_step % print_freq == 0:
                print_avg_reward = np.zeros(config.n_agents)
                for k in range(config.n_agents):
                    print_avg_reward[k] = print_running_reward[k] / print_running_episodes[k]
                    print_avg_reward[k] = round(print_avg_reward[k], 2)

                print("Episode : {} \t Timestep : {} \t Mult factor : {} ".format(i_episode, time_step, env.env.env.current_multiplier))

                print("Average and Episodic Reward:")
                for i_print in range(config.n_agents):
                    print("Average rew agent",str(i_print),"=", print_avg_reward[i_print], "episodic reward=", agents_dict['agent_'+str(i_print)].buffer.rewards[-1])
                print("\n")

                for i in range(config.n_agents):
                    print_running_reward[i] = 0
                    print_running_episodes[i] = 0

            # update PPO agents
            if time_step % config.update_timestep == 0:
                for ag_idx in range(config.n_agents):
                    agents_dict['agent_'+str(ag_idx)].update()

            for i in range(config.n_agents):
                print_running_reward[i] += current_ep_reward[i]
                print_running_episodes[i] += 1

            if (i_episode%10 == 0):
                for ag_idx in range(config.n_agents):
                    wandb.log({"agent"+str(ag_idx)+"_return": agents_dict['agent_'+str(ag_idx)].tmp_return}, step=i_episode)
                    wandb.log({"agent"+str(ag_idx)+"_coop_level": np.mean(agents_dict['agent_'+str(ag_idx)].tmp_actions)}, step=i_episode)
                wandb.log({"episode": i_episode}, step=i_episode)

            i_episode += 1

        for ag_idx in range(n_agents):
            all_returns[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].train_returns
            all_cooperativeness[ag_idx,experiment,:] = agents_dict['agent_'+str(ag_idx)].cooperativeness


        if (config.plots == True):
            ### PLOT TRAIN RETURNS
            plot_train_returns(config, agents_dict, folder, "train_returns_pgg")

            # COOPERATIVITY PERCENTAGE PLOT
            cooperativity_plot(config, agents_dict, folder, "train_cooperativeness")
            plt.savefig("images/pgg/"+str(n_agents)+"_agents/"+folder+"train_returns.png")

            ### EVALUATION
            print("\n\nEVALUATION AFTER LEARNING")

            rews_after = evaluation(agents_dict, config.eval_eps, agent_to_idx)
            print("average rews before", np.average(rews_before[0]), np.average(rews_before[1]))
            print("average rews ater", np.average(rews_after[0]), np.average(rews_after[1]))

            #plot_hist_returns(rews_before, rews_after)

            # Print policy
            pox_coins = np.linspace(0, int(max(rews_after[0])), int(max(rews_after[0])))
            heat = np.zeros((n_agents, len(pox_coins), len(config.mult_fact)))
            for ag in range(config.n_agents):
                for ii in range(len(pox_coins)):
                    for jj in range(len(config.mult_fact)):
                        obs = np.array((pox_coins[ii], n_agents, config.mult_fact[jj]))
                        act = agents_dict['agent_'+str(ag_idx)].select_action(obs)
                        heat[ag, ii,jj] = act
                
            fig, ax = plt.subplots(1, n_agents, figsize=(n_agents*4, 4))
            for ag in range(config.n_agents):
                sns.heatmap(heat[ag], ax=ax[ag])
            print("Saving heatmap..")
            plt.savefig("images/pgg/"+str(n_agents)+"_agents/"+folder+"heatmap.png")
    
    #mean calculations
    plot_avg_on_experiments(config, all_returns, all_cooperativeness, folder, "")
            


if __name__ == "__main__":
    print("eh?")
    train(config)