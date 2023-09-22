import numpy as np
import torch
import math

def bin_acts(comms, acts, n_comm, n_acts, b=None):
    # Binning function that creates a matrix that counts co-occurrences of messages and actions
    if b is None:
        b = np.zeros((n_comm, n_acts))
    for a, c in zip(acts, comms):
        b[c][a] += 1
    return b

def to_int(n):
    # Converts various things to integers
    if type(n) is int:
        return n
    elif type(n) is float:
        return int(n)
    else:
        return int(n.data.numpy())

def probs_from_counts(l, ldim, eps=0):
    # Outputs a probability distribution (list) of length ldim, by counting event occurrences in l
    l_c = [eps] * ldim
    for i in l:
        l_c[i] += 1. / len(l)
    return l_c

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def calc_mutinfo(acts, comms, n_acts, n_comm):
    # Calculate mutual information between actions and messages
    # Joint probability p(a, c) is calculated by counting co-occurences, *not* by performing interventions
    # If the actions and messages come from the same agent, then this is the speaker consistency (SC)
    # If the actions and messages come from different agents, this is the instantaneous coordinatino (IC)
    #print("acts=",acts)
    #print("comms=",comms)
    #print(acts == comms)
    #if (len(acts)>1):
    acts = torch.stack(acts, dim=1)[0]
    comms = torch.stack(comms, dim=1)[0]
    #print("acts=",acts)
    #print("comms=",comms)
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
    #print("mutinfo=", mutinfo)
    return mutinfo


class SocialNorm():

    def __init__(self, params, agents):

        for key, val in params.items(): setattr(self, key, val)

        self.agents = agents
        self.n_agents = len(self.agents)

        self.buffer_len = 600

        self.reset_saved_actions()
        self.reset_comm()

    def reset_comm(self):
        self.saved_mex_states = {key: [] for key in [i for i in range(self.n_agents)]}
        self.saved_actions = {key: [] for key in [i for i in range(self.n_agents)]}
        self.saved_messages = {key: [] for key in [i for i in range(self.n_agents)]}

    def save_comm(self, mex_st, mex, act, active_agents_idxs):
        for ag_idx in active_agents_idxs:
            if (ag_idx not in self.saved_actions.keys()):
                self.saved_actions[ag_idx] = []
            else:
                if (len(self.saved_actions[ag_idx]) >= self.buffer_len):
                    del self.saved_actions[ag_idx][0]
                    del self.saved_messages[ag_idx][0]
                    del self.saved_mex_states[ag_idx][0]

                self.saved_mex_states[ag_idx].append(mex_st["agent_"+str(ag_idx)])
                self.saved_messages[ag_idx].append(mex["agent_"+str(ag_idx)])
                self.saved_actions[ag_idx].append(act["agent_"+str(ag_idx)])

    def change_rep_mex(self, active_agents, active_agents_idxs):
        #print("active_agents_idxs=",active_agents_idxs)
        for ag_idx in active_agents_idxs:
            #print("agent=", ag_idx)
            #print('self.saved_mex_states=',self.saved_actions)
            agent = self.agents["agent_"+str(ag_idx)]
            agent.old_reputation = agent.reputation
            if (self.saved_actions[ag_idx] != []):
                #print("self.saved_actions[ag_idx]=",self.saved_actions[ag_idx])
                #print("self.saved_messages[ag_idx]=",self.saved_messages[ag_idx])
                #print("self.saved_actions[ag_idx] ",self.saved_actions[ag_idx] )
                if (self.freq_counts == True):
                    agent.reputation = (torch.stack(self.saved_actions[ag_idx],dim=0) == torch.stack(self.saved_messages[ag_idx],dim=0)).sum(dim=0)/len(self.saved_messages[ag_idx])
                else:
                    agent.reputation = torch.Tensor([calc_mutinfo(self.saved_actions[ag_idx], self.saved_messages[ag_idx], self.action_size, self.mex_size)])
                #print(agent.reputation)
            #print("rep=", agent.reputation)

    def reset_saved_actions(self):
        self.saved_actions = {key: [] for key in [i for i in range(self.n_agents)]}

    def save_actions(self, act, active_agents_idxs):
        for ag_idx in active_agents_idxs:
            if (ag_idx not in self.saved_actions.keys()):
                self.saved_actions[ag_idx] = []
            else:
                self.saved_actions[ag_idx].append(act["agent_"+str(ag_idx)])

    def rule09(self, active_agents_idxs):
        # agent that cooperates with good agents, and does not cooperate with bad ones is good
        for ag_idx in active_agents_idxs:
            agent = self.agents["agent_"+str(ag_idx)]
            agent.old_reputation = agent.reputation
            #if (agent.is_dummy == False and self.saved_actions[ag_idx] != []):
            if (self.saved_actions[ag_idx] != []):
                other_idx = list(set(active_agents_idxs) - set([agent.idx]))[0]
                other = self.agents["agent_"+str(other_idx)]
                avg_cooperation_level = np.mean(self.saved_actions[ag_idx])

                if (avg_cooperation_level >= self.cooperation_threshold):
                    if (other.old_reputation >= self.other_reputation_threshold):
                        agent.reputation = min(agent.reputation + 0.2, 1.)
                    else: 
                        agent.reputation = max(agent.reputation - 0.2, 0.)
                else: 
                    if (other.old_reputation >= self.other_reputation_threshold):
                        agent.reputation = max(agent.reputation - 0.2, 0.)
                    else: 
                        agent.reputation = min(agent.reputation + 0.2, 1.)

        self.reset_saved_actions()

    def rule09_binary_anast(self, agents, active_agents_idxs):
        for ag_idx in active_agents_idxs:
            #print("ag=", ag_idx)
            agent = self.agents["agent_"+str(ag_idx)]
            
            agent.old_reputation = agent.reputation
            if (self.saved_actions[ag_idx] != []):
                other_idx = list(set(active_agents_idxs) - set([agent.idx]))[0]
                other = self.agents["agent_"+str(other_idx)]

                avg_cooperation_level = np.mean(self.saved_actions[ag_idx])

                if (avg_cooperation_level == torch.Tensor([1.0])):
                    #print("I am cooperating")
                    if (other.old_reputation == 1.):
                        #print("other is cooperator")
                        agent.reputation = torch.Tensor([1.0])
                    else: 
                        #print("other is defector")
                        agent.reputation = torch.Tensor([0.0])
                else: 
                    #print("I am defecting")
                    if (other.old_reputation == 1.):
                        #print("other is cooperator")
                        agent.reputation = torch.Tensor([0.0])
                    else: 
                        #print("other is defector")
                        agent.reputation = torch.Tensor([1.0])
                
                var = torch.bernoulli(torch.Tensor([self.chi])) # tensor contaitning prob that we switch the new reputation assignement
                if (var == 1):
                    #print("FLIP! for agent", ag_idx)
                    if (agent.reputation == torch.Tensor([1.0])):
                        agent.reputation = torch.Tensor([0.0])
                    elif (agent.reputation == torch.Tensor([0.0])):
                        agent.reputation = torch.Tensor([1.0])

            #print("new reputation=", agent.reputation)

        #print("UPDATING ALL REPUTATIONS")
        for ag_idx in active_agents_idxs:     
            agents["agent_"+str(ag_idx)].old_reputation = agents["agent_"+str(ag_idx)].reputation

        self.reset_saved_actions()

    def rule09_binary_pgg(self, agents, active_agents_idxs, mf):
        for ag_idx in active_agents_idxs:
            agent = self.agents["agent_"+str(ag_idx)]

            agent.old_reputation = agent.reputation
            if (self.saved_actions[ag_idx] != []):
                other_idx = list(set(active_agents_idxs) - set([agent.idx]))[0]
                other = self.agents["agent_"+str(other_idx)]
                
                avg_cooperation_level = np.mean(self.saved_actions[ag_idx])

                if (mf >= 1.):
                    if (avg_cooperation_level == torch.Tensor([1.0])):
                        if (other.old_reputation == 1.):
                            agent.reputation = torch.Tensor([1.0])
                        else: 
                            agent.reputation = torch.Tensor([0.0])
                    else: 
                        if (other.old_reputation == 1.):
                            agent.reputation = torch.Tensor([0.0])
                        else: 
                            agent.reputation = torch.Tensor([1.0])
                
                #if (agents["agent_"+str(ag_idx)].is_dummy == False):
                var = torch.bernoulli(torch.Tensor([self.chi])) # tensor contaitning prob that we switch the new reputation assignement
                if (var == 1):
                    if (agent.reputation == torch.Tensor([1.0])):
                        agent.reputation = torch.Tensor([0.0])
                    elif (agent.reputation == torch.Tensor([0.0])):
                        agent.reputation = torch.Tensor([1.0])

            #print("new reputation=", agent.reputation)

        #print("UPDATING ALL REPUTATIONS")
        for ag_idx in active_agents_idxs:     
            agents["agent_"+str(ag_idx)].old_reputation = agents["agent_"+str(ag_idx)].reputation

        self.reset_saved_actions()

    def rule11_binary_pgg(self, agents, active_agents_idxs, mf):
        for ag_idx in active_agents_idxs:
            agent = self.agents["agent_"+str(ag_idx)]

            agent.old_reputation = agent.reputation
            if (self.saved_actions[ag_idx] != []):
                other_idx = list(set(active_agents_idxs) - set([agent.idx]))[0]
                other = self.agents["agent_"+str(other_idx)]
                
                avg_cooperation_level = np.mean(self.saved_actions[ag_idx])

                if (mf >= 1.):
                    if (avg_cooperation_level == torch.Tensor([1.0])):
                            agent.reputation = torch.Tensor([1.0])
                    else: 
                        if (other.old_reputation == 1.):
                            agent.reputation = torch.Tensor([0.0])
                        else: 
                            agent.reputation = torch.Tensor([1.0])
                
                #if (agents["agent_"+str(ag_idx)].is_dummy == False):
                var = torch.bernoulli(torch.Tensor([self.chi])) # tensor contaitning prob that we switch the new reputation assignement
                if (var == 1):
                    if (agent.reputation == torch.Tensor([1.0])):
                        agent.reputation = torch.Tensor([0.0])
                    elif (agent.reputation == torch.Tensor([0.0])):
                        agent.reputation = torch.Tensor([1.0])

            #print("new reputation=", agent.reputation)

        #print("UPDATING ALL REPUTATIONS")
        for ag_idx in active_agents_idxs:     
            agents["agent_"+str(ag_idx)].old_reputation = agents["agent_"+str(ag_idx)].reputation

        self.reset_saved_actions()

    def rule03_binary_pgg(self, agents, active_agents_idxs, mf):
        for ag_idx in active_agents_idxs:
            agent = self.agents["agent_"+str(ag_idx)]

            agent.old_reputation = agent.reputation
            if (self.saved_actions[ag_idx] != []):
                other_idx = list(set(active_agents_idxs) - set([agent.idx]))[0]
                other = self.agents["agent_"+str(other_idx)]
                
                avg_cooperation_level = np.mean(self.saved_actions[ag_idx])

                if (mf >= 1.):
                    if (avg_cooperation_level == torch.Tensor([1.0])):
                        agent.reputation = torch.Tensor([1.0])
                    else: 
                        agent.reputation = torch.Tensor([0.0])
                
                #if (agents["agent_"+str(ag_idx)].is_dummy == False):
                var = torch.bernoulli(torch.Tensor([self.chi])) # tensor contaitning prob that we switch the new reputation assignement
                if (var == 1):
                    if (agent.reputation == torch.Tensor([1.0])):
                        agent.reputation = torch.Tensor([0.0])
                    elif (agent.reputation == torch.Tensor([0.0])):
                        agent.reputation = torch.Tensor([1.0])

            #print("new reputation=", agent.reputation)

        #print("UPDATING ALL REPUTATIONS")
        for ag_idx in active_agents_idxs:     
            agents["agent_"+str(ag_idx)].old_reputation = agents["agent_"+str(ag_idx)].reputation

        self.reset_saved_actions()

    def rule00_binary_pgg(self, agents, active_agents_idxs, mf):
        for ag_idx in active_agents_idxs:
            agent = self.agents["agent_"+str(ag_idx)]

            agent.old_reputation = agent.reputation
            if (self.saved_actions[ag_idx] != []):
                other_idx = list(set(active_agents_idxs) - set([agent.idx]))[0]
                other = self.agents["agent_"+str(other_idx)]
                
                avg_cooperation_level = np.mean(self.saved_actions[ag_idx])

                agent.reputation = torch.Tensor([0.0])
                
                #if (agents["agent_"+str(ag_idx)].is_dummy == False):
                var = torch.bernoulli(torch.Tensor([self.chi])) # tensor contaitning prob that we switch the new reputation assignement
                if (var == 1):
                    if (agent.reputation == torch.Tensor([1.0])):
                        agent.reputation = torch.Tensor([0.0])
                    elif (agent.reputation == torch.Tensor([0.0])):
                        agent.reputation = torch.Tensor([1.0])

            #print("new reputation=", agent.reputation)

        #print("UPDATING ALL REPUTATIONS")
        for ag_idx in active_agents_idxs:     
            agents["agent_"+str(ag_idx)].old_reputation = agents["agent_"+str(ag_idx)].reputation

        self.reset_saved_actions()

    def rule09_binary(self, agents, active_agents_idxs):
        # agent that cooperates with good agents, and does not cooperate with bad ones is good
        for ag_idx in active_agents_idxs:
            #print("agent=", ag_idx)
            #if (agents["agent_"+str(ag_idx)].is_dummy == True):
            #    #print("agent is dummy")
            agent = self.agents["agent_"+str(ag_idx)]
            
            #print("reputation before=", agent.reputation)
            agent.old_reputation = agent.reputation
            if (self.saved_actions[ag_idx] != []):
                other_idx = list(set(active_agents_idxs) - set([agent.idx]))[0]
                other = self.agents["agent_"+str(other_idx)]
                #print("other.reputation=", other.old_reputation)
                #print("self.saved_actions[ag_idx]=",self.saved_actions[ag_idx])
                avg_cooperation_level = np.mean(self.saved_actions[ag_idx])
                #print("my_cooperation_level=",avg_cooperation_level)

                if (avg_cooperation_level >= self.cooperation_threshold):
                    if (other.old_reputation == 1.):
                        agent.reputation = torch.Tensor([1.0])
                    else: 
                        agent.reputation = torch.Tensor([0.0])
                else: 
                    if (other.old_reputation == 1.):
                        agent.reputation = torch.Tensor([0.0])
                    else: 
                        agent.reputation = torch.Tensor([1.0])
                
                var = torch.bernoulli(torch.Tensor([self.chi])) # tensor contaitning prob that we switch the new reputation assignement
                if (var == 1):
                    if (agent.reputation == torch.Tensor([1.0])):
                        agent.reputation = torch.Tensor([0.0])
                    elif (agent.reputation == torch.Tensor([0.0])):
                        agent.reputation = torch.Tensor([1.0])

            #print("new reputation=", agent.reputation)

        #print("UPDATING ALL REPUTATIONS")
        for ag_idx in active_agents_idxs:     
            agents["agent_"+str(ag_idx)].old_reputation = agents["agent_"+str(ag_idx)].reputation

        self.reset_saved_actions()

    def rule00_binary(self, agents, active_agents_idxs):
        # agent that never cooperates (baseline)
        for ag_idx in active_agents_idxs:
            agent = self.agents["agent_"+str(ag_idx)]
            agent.reputation = torch.Tensor([0.0])
        
        for ag_idx in active_agents_idxs:       
            agents["agent_"+str(ag_idx)].old_reputation = agents["agent_"+str(ag_idx)].reputation

        self.reset_saved_actions()

    def reputation_as_avg_cooperation(self, active_agents_idxs):
        for ag_idx in active_agents_idxs:
            if self.saved_actions[ag_idx] != []:
                self.agents["agent_"+str(ag_idx)].reputation = np.mean(self.saved_actions[ag_idx])
        self.reset_saved_actions()
