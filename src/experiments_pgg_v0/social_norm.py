import numpy as np
import torch

class SocialNorm():

    def __init__(self, params, agents):

        for key, val in params.items(): setattr(self, key, val)

        self.agents = agents
        self.n_agents = len(self.agents)

        if (self.binary_reputation == True):
            self.threshold = 0.5
        else: 
            self.threshold = self.cooperation_threshold

        self.reset_saved_actions()

    def reset_saved_actions(self):
        self.saved_actions = {key: [] for key in [i for i in range(self.n_agents)]} #{}

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
            if (agent.is_dummy == False and self.saved_actions[ag_idx] != []):
                other_idx = list(set(active_agents_idxs) - set([agent.idx]))[0]
                other = self.agents["agent_"+str(other_idx)]
                avg_cooperation_level = np.mean(self.saved_actions[ag_idx])

                if (avg_cooperation_level >= self.threshold):
                    if (other.reputation >= self.other_reputation_threshold):
                        agent.reputation = min(agent.reputation + 0.2, 1.)
                    else: 
                        agent.reputation = max(agent.reputation - 0.2, 0.)
                else: 
                    if (other.reputation >= self.other_reputation_threshold):
                        agent.reputation = max(agent.reputation - 0.2, 0.)
                    else: 
                        agent.reputation = min(agent.reputation + 0.2, 1.)

    def rule09_binary(self, active_agents_idxs):
        # agent that cooperates with good agents, and does not cooperate with bad ones is good
        for ag_idx in active_agents_idxs:
            agent = self.agents["agent_"+str(ag_idx)]
            if (agent.is_dummy == False and self.saved_actions[ag_idx] != []):
                other_idx = list(set(active_agents_idxs) - set([agent.idx]))[0]
                other = self.agents["agent_"+str(other_idx)]
                avg_cooperation_level = np.mean(self.saved_actions[ag_idx])

                if (avg_cooperation_level >= self.threshold):
                    if (other.reputation == 1.):
                        agent.reputation = 1.
                    else: 
                        agent.reputation = 0.
                else: 
                    if (other.reputation == 1.):
                        agent.reputation = 0.
                    else: 
                        agent.reputation = 1.

    def rule00(self, active_agents_idxs):
        # agent that never cooperates (baseline)
        for ag_idx in active_agents_idxs:
            agent = self.agents["agent_"+str(ag_idx)]
            agent.reputation = 0

    def reputation_as_avg_cooperation(self, active_agents_idxs):
        for ag_idx in active_agents_idxs:
            if self.saved_actions[ag_idx] != []:
                self.agents["agent_"+str(ag_idx)].reputation = np.mean(self.saved_actions[ag_idx])
        self.reset_saved_actions()
