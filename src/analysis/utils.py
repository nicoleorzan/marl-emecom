import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import torch
import math
import itertools 

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

def get_p_a_given_do_c(config, possible_messages, agents_dict, parallel_env):
    # Calculates p(a | do(c)) for all agents, i.e. the probability distribution over other agent's messages given that
    p_a_given_do_c = [np.zeros((len(possible_messages), config.action_size)), np.zeros((len(possible_messages), config.action_size)), np.zeros((len(possible_messages), config.action_size))]

    # For both agents
    # Iterated over possible messages
    for idx_mex, message in enumerate(possible_messages):
        observations = parallel_env.reset()

        acts_distrib = {agent: agents_dict[agent].get_action_distribution(observations[agent], message) for agent in parallel_env.agents}
    
        for ag_idx, agent in enumerate(parallel_env.agents):
            p_a_given_do_c[ag_idx][idx_mex, :] = acts_distrib[agent].numpy()

    return p_a_given_do_c, observations

def define_all_possibe_messages(config):

    pox_mex = [i for i in range(config.mex_size)]
    pox_configurations = [p for p in itertools.product(pox_mex, repeat=config.n_agents)]
    possible_messages = []
    for c in pox_configurations:
        tens = torch.Tensor([])
        for ag_idx in range(config.n_agents):
            tens = torch.concat((tens,torch.nn.functional.one_hot(torch.Tensor([c[ag_idx]]).long(), num_classes=config.mex_size)[0]), dim=0)
        possible_messages.append(tens)
            
    return possible_messages 
"""   
def calc_cic(config, possible_messages, p_a_given_do_c, p_c):
    # Calculate the one-step causal influence of communication, i.e. the mutual information using p(a | do(c))
    print("possible_messages=", possible_messages)
    print("p_a_given_do_c=",p_a_given_do_c)
    print("pc=", p_c, "np.expand_dims(p_c, axis=1)=",  np.expand_dims(p_c, axis=1))
    p_ac = numpy.zeros((p_a_given_do_c.shape))
    print("p_ac",p_ac)
    p_ac[0:config.mex_size**(config.n_agents-1)] =  p_a_given_do_c[0:config.mex_size**(config.n_agents-1)]*p_c[0]
    p_ac[config.mex_size**(config.n_agents-1):config.mex_size**(config.n_agents-1)*2] = p_a_given_do_c[config.mex_size**(config.n_agents-1):config.mex_size**(config.n_agents-1)*2]*p_c[1]
    p_ac[config.mex_size**(config.n_agents-1)*2:config.mex_size**(config.n_agents-1)*3] = p_a_given_do_c[config.mex_size**(config.n_agents-1)*2:config.mex_size**(config.n_agents-1)*3]*p_c[2]
    
    #p_ac = p_a_given_do_c * np.expand_dims(p_c, axis=1)  # calculate joint probability p(a, c)
    p_ac /= np.sum(p_ac)  # re-normalize
    p_a = np.mean(p_ac, axis=0)  # compute p(a) by marginalizing over c

    # Calculate mutual information
    cic = 0
    for c in range(config.mex_size):
        for a in range(config.action_size):
            if p_ac[c][a] > 0:
                cic += p_ac[c][a] * math.log(p_ac[c][a] / (p_c[c] * p_a[a]))
    return cic

def calc_model_cic(config, possible_messages, agents_dict, parallel_env, num_games=1000):
    # Given trained model files in model_file_ag1 and model_file_ag2 (.txt files saved with torch.save), calculates
    # the one-step CIC, averaged over num_games training games
    # args are used to specify MCG game, and structure of the agen

    # Iterate over games
    cics = [[], [], []]
    for i in range(num_games):
        print("game=", i)
        # Get a new game (which is random even if args.game = fixed)
        #observations = parallel_env.reset()

        # Calculate p(a | do(c)) for both agents and messages c
        p_a_given_do_c, observations = get_p_a_given_do_c(config, possible_messages, agents_dict, parallel_env)
        # For each agent, calculate the one-step CIC
        # Calcualte p(c) of other agent (1-ag) by doing a forward pass through network
        #logits_c, logits_a, v = agents[1 - ag].forward(torch.Tensor(ob_c[ag]))
        #probs_c = F.softmax(logits_c, dim=0).data.numpy()
        probs_c = {agent: agents_dict[agent].get_message_distribution(observations[agent]) for agent in parallel_env.agents}
        print("probs_c=", probs_c)
        for ag_idx, agent in enumerate(parallel_env.agents):
            cic = calc_cic(config, possible_messages, p_a_given_do_c[ag_idx], probs_c[agent])
            cics[ag_idx].append(cic)

    return cics
"""
def calc_mutinfo2(possible_messages, parallel_env, agent_speaker, agent_speaker_idx, agent_listener, agent_listener_idx):
    cic = 0
    #define set of possible messages
    
    #states loop
    for i in range(100):
        observations = parallel_env.reset()
        print("obs=", observations)
        
        # loop over possible messages
        for mex_idx, mex in enumerate(possible_messages):
            print("mex=", mex)

            p_mj = agent_speaker.policy_comm.get_distribution(observations["agent_"+str(agent_speaker_idx)])#[mex_idx]
            print("p_mj=", p_mj)
            obs = torch.FloatTensor(observations["agent_"+str(agent_listener_idx)])
            # intervene forcing message "mex"
            state_mex = torch.cat((obs, mex))#.to(device)
            print("state_mex=", state_mex)
            p_a_given_mj = agent_listener.policy_act.get_distribution(state_mex)
            print("p_a_mj=", p_a_given_mj)
            p_a_and_mj = torch.matmul( torch.reshape(p_mj,(p_mj.shape[0],1)), torch.reshape(p_a_given_mj, (1,p_a_given_mj.shape[0])))
            print("p_a_and_mj= ", p_a_and_mj)
            p_a = torch.sum(p_a_and_mj, dim=0) # sum over axes of the action set
            print("p_a=", p_a)
            print(torch.matmul(torch.reshape(p_a, (p_a.shape[0], 1)),torch.reshape(p_mj, (1, p_mj.shape[0]))))
            #cic += (1/i+1)*torch.sum(p_a_and_mj*(torch.log(p_a_and_mj)/torch.matmul(log1,log2)), dim=1)



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