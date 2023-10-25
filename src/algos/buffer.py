from collections import deque
import numpy as np

class RolloutBufferComm:
    def __init__(self):
        self.states_c = []
        self.states_a = []
        self.next_states_a = []
        self.messages = []
        self.actions = []
        self.actions_given_m = {}
        self.messages_given_m = {}
        self.act_logprobs = []
        self.comm_logprobs = []
        self.act_entropy = []
        self.comm_entropy = []
        self.rewards = []
        self.is_terminals = []
        self.mut_info = []

    def clear_batch(self):
        self.messages_given_m = {}
        self.actions_given_m = {}

    def len_(self):
        return len(self.states_a)
        
    def clear(self):
        del self.states_c[:]
        del self.states_a[:]
        del self.next_states_a[:]
        del self.messages[:]
        self.messages_given_m = {}
        self.actions_given_m = {}
        del self.actions[:]
        del self.act_logprobs[:]
        del self.comm_logprobs[:]
        del self.act_entropy[:]
        del self.comm_entropy[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.mut_info[:]