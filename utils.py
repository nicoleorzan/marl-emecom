import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def evaluation(agents_dict, episodes, agent_to_idx):

    agents_returns = np.zeros((episodes, len(agents_dict)))
    for e in range(episodes):
        if (e%100 == 0):
            print("Episode:", e)
        agents_returns[e] = evaluate_episode(agents_dict, agent_to_idx)

    return agents_returns

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_train_returns(config, agents_dict, path, name):

    num_blocks = 40

    moving_avgs = []
    for ag_idx in range(config.n_agents):
        moving_avgs.append(moving_average(agents_dict['agent_'+str(ag_idx)].train_returns, num_blocks))

    fig, ax = plt.subplots(config.n_agents)
    fig.suptitle("Train Returns")
    for ag_idx in range(config.n_agents):
        ax[ag_idx].plot(np.linspace(0, len(agents_dict['agent_'+str(ag_idx)].train_returns), len(agents_dict['agent_'+str(ag_idx)].train_returns)), agents_dict['agent_'+str(ag_idx)].train_returns)
        ax[ag_idx].plot(np.linspace(0, len(agents_dict['agent_'+str(ag_idx)].train_returns), len(moving_avgs[ag_idx])), moving_avgs[ag_idx])
    plt.savefig(path+name+".png")

def cooperativity_plot(config, agents_dict, path, name):
    fig, ax = plt.subplots(config.n_agents)
    fig.suptitle("Train Cooperativity mean over the iteractions")
    for ag_idx in range(config.n_agents):
        train_actions = agents_dict['agent_'+str(ag_idx)].train_actions
        train_act_array = np.array(train_actions)
        avgs = np.mean(train_act_array, axis=1)
        ax[ag_idx].plot(np.linspace(0, len(train_actions), len(train_actions)), avgs)
    plt.savefig(path+name+".png")


def plot_avg_on_experiments(config, all_returns, all_cooperativeness, path, comm):

    average_returns = np.zeros((config.n_agents, config.episodes_per_experiment))
    average_cooperativeness = np.zeros((config.n_agents, config.episodes_per_experiment))

    for ag_idx in range(config.n_agents):
        average_returns[ag_idx, :] = np.mean(all_returns[ag_idx], axis=0) 
        average_cooperativeness[ag_idx, :] = np.mean(all_cooperativeness[ag_idx], axis=0)     

    fig, ax = plt.subplots(config.n_agents)
    fig.suptitle("AVG Train Returns")
    for ag_idx in range(config.n_agents):
        ax[ag_idx].plot(np.linspace(0, len(average_returns[ag_idx]), len(average_returns[ag_idx])), average_returns[ag_idx])
    plt.savefig(path+"AVG_train_returns"+comm+".png")

    fig, ax = plt.subplots(config.n_agents)
    fig.suptitle("AVG Cooperativity")
    for ag_idx in range(config.n_agents):
        #sns.lineplot(np.linspace(0, len(average_cooperativeness[ag_idx, :]), len(average_cooperativeness[ag_idx, :])), average_cooperativeness[ag_idx, :], ax=ax[ag_idx])
        ax[ag_idx].plot(np.linspace(0, len(average_cooperativeness[ag_idx, :]), len(average_cooperativeness[ag_idx, :])), average_cooperativeness[ag_idx, :])
        ax[ag_idx].set_ybound(0,1)
    plt.savefig(path+"AVG_coop"+comm+".png")    

    #df = pd.DataFrame(average_returns) 
    #df1 = pd.DataFrame(average_cooperativeness)

    #df.to_csv(path+"avg_returns"+comm+".csv")   
    #df1.to_csv(path+"avg_cooperativeness"+comm+".csv")