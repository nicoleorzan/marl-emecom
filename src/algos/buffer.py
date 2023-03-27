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
    

class DQNBuffer(RolloutBufferComm):

    def __init__(self, capacity=200):
        RolloutBufferComm.__init__(self)
        self.capacity = capacity

    def sample_comm(self, batch_size):
        print("len=", len(self.states_a))
        print("batch_size=", batch_size)
        ind = np.random.randint(0, len(self.states_a), size=batch_size)
        print("ind=",  ind)
        print("states=", self.states_a)
        s_c = [self.states_a[i] for i in ind]
        s_a = [self.states_a[i] for i in ind]
        m = [self.messages[i] for i in ind]
        a = [self.actions[i] for i in ind]
        r = [self.rewards[i] for i in ind]
        t = [self.is_terminals[i] for i in ind]
        return s_c, m, s_a, a, r, t
    
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.states_a), size=batch_size)
        #print("self.states_a=",self.states_a)
        #print("self.next_states_a",self.next_states_a)
        #print("ind=", ind)
        s_a = [self.states_a[i] for i in ind]
        next_s_a = [self.next_states_a[i] for i in ind]
        a = [self.actions[i] for i in ind]
        r = [self.rewards[i] for i in ind]
        return s_a, a, r, next_s_a

    def __len__(self):
        return len(self.states_a)
    


"""class RolloutBuffer:
    
    def __init__(self, recurrent = False):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.recurrent = recurrent
        if self.recurrent:
            self.hstates = []
            self.cstates = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        if self.recurrent:
            del self.hstates[:]
            del self.cstates[:]"""