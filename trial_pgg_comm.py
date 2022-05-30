from src.environments import pgg_v0
from PPOcomm import PPOcomm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb

hyperparameter_defaults = dict(
    eval_eps = 1000,
    max_training_steps = 50000, # break training loop if timeteps > max_training_timesteps
    update_timestep = 40, # update policy every n timesteps
    n_agents = 2,
    uncertainties = [0., 0.],
    coins_per_agent = 4,
    mult_fact = 10,
    num_game_iterations = 5,
    comm = True,
    action_space = 2,
    obs_dim = 3,         # we observe coins we have, num of agents, and multiplier factor with uncertainty
    mex_space = 2,               # vocabulary
    K_epochs = 40,               # update policy for K epochs
    eps_clip = 0.2,              # clip parameter for PPO
    gamma = 0.99,                # discount factor
    c1 = 0.5,
    c2 = -0.01,
    lr_actor = 0.001,       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
)

wandb.init(project="pgg", entity="nicoleorzan", config=hyperparameter_defaults)#, mode="offline")
config = wandb.config

assert (config.n_agents == len(config.uncertainties))

folder = 'coop_m='+str(config.mult_fact)+'/'

max_ep_len = 1                    # max timesteps in one episode
num_blocks = 10                   # number of blocks for moving average

print_freq = 100     # print avg reward in the interval (in num timesteps)

def evaluation(agents_dict, episodes, agent_to_idx):

    agentsr = np.zeros((episodes, len(agents_dict)))
    for e in range(episodes):
        if (e%100 == 0):
            print("Episode:", e)
        agentsr[e] = evaluate_episode(agents_dict, agent_to_idx)

    return agentsr

def evaluate_episode(agents_dict, agent_to_idx):
    env = pgg_v0.env(n_agents=config.n_agents, coins_per_agent=config.coins_per_agent, num_iterations=config.num_game_iterations, \
        mult_fact=config.mult_fact, uncertainties=config.uncertainties)
    env.reset()
    i = 0
    ag_rets = np.zeros(len(agents_dict))
    #mex_in_0 = torch.tensor([0.]).long()
    #mex_in_0 = F.one_hot(mex_in_0, num_classes=mex_space)[0]
    #mex_in = mex_in_0
    mex_in = torch.zeros((config.mex_space*config.n_agents), dtype=torch.int64)
    mex_out_aggreg = torch.zeros((config.mex_space*config.n_agents)).long()

    for id_agent in env.agent_iter():
        idx = agent_to_idx[id_agent]
        obs, reward, done, _ = env.last()
        obs = torch.FloatTensor(obs)
        acting_agent = agents_dict[id_agent]
        #print("obs=", obs, "mex=", mex_in)
        if (i > config.n_agents-1):
            mex_in = mex_out_aggreg
        state = torch.cat((obs, mex_in), dim=0)

        #act = acting_agent.select_action(state) if not done else None
        if not done:
            act, mex_out = acting_agent.select_action(state)
            mex_out = torch.Tensor([mex_out]).long()
            mex_out = F.one_hot(mex_out, num_classes=config.mex_space)[0]
            mex_out_aggreg[idx*config.mex_space:idx*config.mex_space+config.mex_space] = mex_out
            #mex_in = mex_out
            #print("mex_out=", mex_out)
        else:
            act = None
            mex_out = None

        env.step(act)
        ag_rets[idx] += reward

        i += 1
    env.close()
    return ag_rets

def plot_hist_returns(rews_before, rews_after):

    fig, ax = plt.subplots(config.n_agents, 2, figsize=(20,8))
    fig.suptitle("Distribution of Returns", fontsize=25)

    n_bins = 40

    for i in range(config.n_agents):
        ax[i,0].hist(rews_before[i], bins=n_bins, range=[-0., max(rews_before[i])], label='agent'+str(i)+' before')
        ax[i,0].legend(prop=dict(size=18))
        ax[i,1].hist(rews_after[i], bins=n_bins, range=[-0., max(rews_after[i])], label='agent'+str(i)+' after')
        ax[i,1].legend(prop=dict(size=18))

    print("Saving histogram..")
    plt.savefig("images/pgg/"+str(config.n_agents)+"_agents/"+folder+"hist_rewards_pgg_comm.png")

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def train(config):

    env = pgg_v0.env(n_agents=config.n_agents, coins_per_agent=config.coins_per_agent, num_iterations=config.num_game_iterations, \
        mult_fact=config.mult_fact, uncertainties=config.uncertainties)

    n_agents = config.n_agents
    mex_space = config.mex_space

    agents_dict = {}
    un_agents_dict = {}
    agent_to_idx = {}
    for idx in range(config.n_agents):
        agents_dict['agent_'+str(idx)] = PPOcomm(config.obs_dim, config.action_space, config.mex_space, config.n_agents, config.lr_actor, config.lr_critic,  \
        config.gamma, config.K_epochs, config.eps_clip, config.c1, config.c2)
        un_agents_dict['agent_'+str(idx)] = PPOcomm(config.obs_dim, config.action_space, config.mex_space, config.n_agents, config.lr_actor, config.lr_critic,  \
        config.gamma, config.K_epochs, config.eps_clip, config.c1, config.c2)
        agent_to_idx['agent_'+str(idx)] = idx

    print("\nEVALUATION BEFORE LEARNING")
    rews_before = evaluation(un_agents_dict, config.eval_eps, agent_to_idx)

    #### TRAINING LOOP
    print("\n===>Training")
    time_step = 0
    i_episode = 0

    # printing and logging variables
    print_running_reward = np.zeros(config.n_agents)
    print_running_episodes = np.zeros(config.n_agents)


    while time_step <= config.max_training_steps:

        env.reset()

        current_ep_reward = np.zeros(n_agents)
        i_internal_loop = 0
        agents_tmp_returns = np.zeros((n_agents))
        for ag_idx in range(n_agents):
            agents_dict['agent_'+str(ag_idx)].tmp_return = 0
            agents_dict['agent_'+str(ag_idx)].tmp_actions = []

        mex_in_0 = torch.tensor([0.]).long()
        mex_in_0 = F.one_hot(mex_in_0, num_classes=mex_space)[0]
        #print("mex_in_0=", mex_in_0)
        mex_in = torch.zeros((mex_space*n_agents), dtype=torch.int64)
        mex_out_aggreg = torch.zeros((mex_space*n_agents)).long()
        #print("mex out aggr=", mex_out_aggreg)

        for id_agent in env.agent_iter():
            #print("\nTime=", time_step)
            #print(agent_to_idx[id_agent], id_agent)
            idx = agent_to_idx[id_agent]
            acting_agent = agents_dict[id_agent]
            
            obs, rew, done, info = env.last()
            #print(obs, rew, done, info)
            obs = torch.FloatTensor(obs)#.to(device)
            if (i_internal_loop > n_agents-1):
                mex_in = mex_out_aggreg
            #print("mex_in =", mex_in)
            state = torch.cat((obs, mex_in), dim=0)
            
            if not done:
                act, mex_out = acting_agent.select_action(state)
                mex_out = torch.Tensor([mex_out]).long()
                mex_out = F.one_hot(mex_out, num_classes=mex_space)[0]
                #mex_in = mex_out
                mex_out_aggreg[idx*mex_space:idx*mex_space+mex_space] = mex_out
                #print("mex_out=", mex_out)
                #print("mex out aggr=", mex_out_aggreg)
            else:
                act = None
                mex_out = None
            #print("act=", act, "mex_out=", mex_out)
            env.step(act)

            #print("step done")
            if (i_internal_loop > n_agents-1):
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
                for ag_idx in range(n_agents):
                    agents_dict['agent_'+str(ag_idx)].train_returns.append(agents_dict['agent_'+str(ag_idx)].tmp_return)
                    agents_dict['agent_'+str(ag_idx)].train_actions.append(agents_dict['agent_'+str(ag_idx)].tmp_actions)
                if (idx == n_agents-1):
                    #print("======>exiting\n\n\n") 
                    break

            i_internal_loop += 1
        
        if time_step % print_freq == 0:
            #print("time_step % print_freq=",time_step % print_freq)
            print_avg_reward = np.zeros(n_agents)
            for k in range(n_agents):
                #print(print_running_reward[k],print_running_episodes[k])
                print_avg_reward[k] = print_running_reward[k] / print_running_episodes[k]
                print_avg_reward[k] = round(print_avg_reward[k], 2)

            print("Episode : {} \t\t Timestep : {} \t\t ".format(i_episode, time_step))
            print("Average and Episodic Reward:")
            for i_print in range(n_agents):
                print("Average rew agent",str(i_print),"=", print_avg_reward[i_print], "episodic reward=", agents_dict['agent_'+str(i_print)].buffer.rewards[-1])

            for i in range(n_agents):
                print_running_reward[i] = 0
                print_running_episodes[i] = 0

        # update PPO agent
        if time_step % config.update_timestep == 0:
            for ag_idx in range(n_agents):
                agents_dict['agent_'+str(ag_idx)].update()
        
        for i in range(n_agents):
            print_running_reward[i] += current_ep_reward[i]
            print_running_episodes[i] += 1

        i_episode += 1


    ### PLOT TRAIN RETURNS
    moving_avgs = []
    for ag_idx in range(n_agents):
        moving_avgs.append(moving_average(agents_dict['agent_'+str(ag_idx)].train_returns, num_blocks))

    fig, ax = plt.subplots(n_agents)
    fig.suptitle("Train Returns")
    for i in range(n_agents):
        ax[i].plot(np.linspace(0, len(agents_dict['agent_'+str(ag_idx)].train_returns), len(agents_dict['agent_'+str(ag_idx)].train_returns)), agents_dict['agent_'+str(ag_idx)].train_returns)
        ax[i].plot(np.linspace(0, len(agents_dict['agent_'+str(ag_idx)].train_returns), len(moving_avgs[i])), moving_avgs[i])
    plt.savefig("images/pgg/"+str(n_agents)+"_agents/"+folder+"train_returns_pgg_comm.png")


    ### COOPERATIVITY PERCENTAGE PLOT
    #moving_avgs = []
    #for ag_idx in range(n_agents):
    #    moving_avgs.append(moving_average(agents_dict['agent_'+str(ag_idx)].train_returns, num_blocks))
    fig, ax = plt.subplots(n_agents)
    fig.suptitle("Train Cooperativity mean over the iteractions")
    for i in range(n_agents):
        train_actions = agents_dict['agent_'+str(ag_idx)].train_actions
        train_act_array = np.array(train_actions)
        avgs = np.mean(train_act_array, axis=1)
        ax[i].plot(np.linspace(0, len(train_actions), len(train_actions)), avgs)
        #ax[i].plot(np.linspace(0, len(agents_dict['agent_'+str(ag_idx)].train_returns), len(moving_avgs[i])), moving_avgs[i])
    plt.savefig("images/pgg/"+str(n_agents)+"_agents/"+folder+"train_cooperativeness_comm.png")


    ### EVALUATION
    print("\n\nEVALUATION AFTER LEARNING")

    rews_after = evaluation(agents_dict, config.eval_eps, agent_to_idx)
    print("average rews before", np.average(rews_before[0]), np.average(rews_before[1]))
    print("average rews ater", np.average(rews_after[0]), np.average(rews_after[1]))

    plot_hist_returns(rews_before, rews_after)


if __name__ == "__main__":
    train(config)