import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import torch
import math

def calc_entropy(comms, n_comm):
    # Calculates the entropy of the communication distribution
    # p(c) is calculated by averaging over episodes
    comms = [to_int(m) for m in comms]
    eps = 1e-9

    p_c = probs_from_counts(comms, n_comm, eps=eps)
    entropy = 0
    for c in range(n_comm):
        entropy += - p_c[c] * math.log(p_c[c])
    return entropy


def calc_mutinfo(acts, comms, n_acts, n_comm):
    # Calculate mutual information between actions and messages
    # Joint probability p(a, c) is calculated by counting co-occurences, *not* by performing interventions
    # If the actions and messages come from the same agent, then this is the speaker consistency (SC)
    # If the actions and messages come from different agents, this is the instantaneous coordinatino (IC)
    #print(acts)
    #print(comms)
    if (comms[0].size() != torch.Size()):
        comms = [torch.argmax(c) for c in comms]
    comms = [to_int(m) for m in comms]
    if (acts[0].size() != torch.Size()):
        acts = [torch.argmax(c) for c in acts]
    acts = [to_int(a) for a in acts]
    #print("acts=", acts)
    #print("comm=", comms)

    # Calculate probabilities by counting co-occurrences
    p_a = probs_from_counts(acts, n_acts)
    p_c = probs_from_counts(comms, n_comm)
    p_ac = bin_acts(comms, acts, n_comm, n_acts)
    p_ac /= np.sum(p_ac)  # normalize counts into a probability distribution

    # Calculate mutual information
    mutinfo = 0
    for c in range(n_comm):
        for a in range(n_acts):
            if p_ac[c][a] > 0:
                mutinfo += p_ac[c][a] * math.log(p_ac[c][a] / (p_c[c] * p_a[a]))
    return mutinfo

def probs_from_counts(l, ldim, eps=0):
    # Outputs a probability distribution (list) of length ldim, by counting event occurrences in l
    l_c = [eps] * ldim
    for i in l:
        l_c[i] += 1. / len(l)
    return l_c

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# from https://github.com/facebookresearch/measuring-emergent-comm/blob/master/measuring_emergent_comm/utils.py


def to_int(n):
    # Converts various things to integers
    if type(n) is int:
        return n
    elif type(n) is float:
        return int(n)
    else:
        return int(n.data.numpy())


def calc_stats(comms, acts, n_comm, n_acts, stats):
    # Produces a matrix ('stats') that counts co-occurrences of messages and actions
    # Can update an existing 'stats' matrix (=None if there is none)
    # Calls bin_acts to do the heavy lifting
    #print("comms=", comms)
    #print("actions=", acts)
    comms = [torch.argmax(c) for c in comms]
    comms = [to_int(m) for m in comms]
    acts = [to_int(a) for a in acts]
    #print("comms=", comms)
    #print("actions=", acts)
    stats = bin_acts(comms, acts, n_comm, n_acts, stats)
    return stats


def bin_acts(comms, acts, n_comm, n_acts, b=None):
    # Binning function that creates a matrix that counts co-occurrences of messages and actions
    if b is None:
        b = np.zeros((n_comm, n_acts))
    for a, c in zip(acts, comms):
        b[c][a] += 1
    return b


def plot_info(config, infos, path, name):
    #print(infos)
    # plot mutual information, entropy or speaker consistency depending on the input given

    fig, ax = plt.subplots(config.n_agents)
    fig.suptitle(name)
    for ag_idx in range(config.n_agents):
        ax[ag_idx].plot(np.linspace(0, len(infos[ag_idx]), len(infos[ag_idx])), infos[ag_idx])
        ax[ag_idx].grid()
    plt.savefig(path+name+".png")


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
        ax[ag_idx].grid()
    plt.savefig(path+name+".png")


def plot_losses(config, agents_dict, path, name, comm=False):

    fig, ax = plt.subplots(config.n_agents)
    fig.suptitle("Train Losses")
    for ag_idx in range(config.n_agents):
        if (comm == True):
            ax[ag_idx].plot(np.linspace(0, len(agents_dict['agent_'+str(ag_idx)].saved_losses_comm), len(agents_dict['agent_'+str(ag_idx)].saved_losses_comm)), agents_dict['agent_'+str(ag_idx)].saved_losses_comm)
        else:
            ax[ag_idx].plot(np.linspace(0, len(agents_dict['agent_'+str(ag_idx)].saved_losses), len(agents_dict['agent_'+str(ag_idx)].saved_losses)), agents_dict['agent_'+str(ag_idx)].saved_losses)
        ax[ag_idx].grid()
    plt.savefig(path+name+".png")


def cooperativity_plot(config, agents_dict, path, name):
    fig, ax = plt.subplots(config.n_agents)
    fig.suptitle("Train Cooperativity")
    for ag_idx in range(config.n_agents):
        train_actions = agents_dict['agent_'+str(ag_idx)].coop
        train_act_array = np.array(train_actions)
        ax[ag_idx].plot(np.linspace(0, len(train_actions), len(train_actions)), train_act_array)
        ax[ag_idx].grid()
    plt.savefig(path+name+".png")


def plots_experiments(config, all_returns, all_cooperativeness, path, comm):

    average_returns = np.zeros((config.n_agents, int(config.episodes_per_experiment/config.save_interval)))
    average_cooperativeness = np.zeros((config.n_agents, int(config.episodes_per_experiment/config.save_interval)))

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
        ax[ag_idx].plot(np.linspace(0, len(average_cooperativeness[ag_idx, :]), len(average_returns[ag_idx])), average_cooperativeness[ag_idx, :])
        ax[ag_idx].set_ybound(-0.01,1.01)
    plt.savefig(path+"AVG_coop"+comm+".png")