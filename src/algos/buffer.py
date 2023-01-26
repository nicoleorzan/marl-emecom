
class RolloutBuffer:
    
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
            del self.cstates[:]

    def __print__(self):
        print("states=", len(self.states))
        if self.recurrent:
            print("hstates=", len(self.hstates))
            print("cstates=", len(self.cstates))
        print("actions=", len(self.actions))
        print("logprobs=", len(self.logprobs))
        print("rewards=", len(self.rewards))
        print("is_terminals=", len(self.is_terminals))
        
class RolloutBufferComm:
    def __init__(self, recurrent = False):
        self.states_c = []
        self.states_a = []
        self.messages = []
        self.actions = []
        self.actions_given_m = {}
        self.messages_given_m = {}
        self.act_logprobs = []
        self.comm_logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.mut_info = []
        self.recurrent = recurrent
        if self.recurrent:
            self.hstates_c = []
            self.cstates_c = []
            self.hstates_a = []
            self.cstates_a = []

    def clear_batch(self):
        self.messages_given_m = {}
        self.actions_given_m = {}
        
    def clear(self):
        del self.states_c[:]
        del self.states_a[:]
        del self.messages[:]
        self.messages_given_m = {}
        self.actions_given_m = {}
        del self.actions[:]
        del self.act_logprobs[:]
        del self.comm_logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.mut_info[:]
        if self.recurrent:
            del self.hstates_c[:]
            del self.cstates_c[:]
            del self.hstates_a[:]
            del self.cstates_a[:]

    def __print__(self):
        print("states_c=", len(self.states_c))
        print("states_a=", len(self.states_a))
        print("messages_out=", len(self.messages))
        print("actions=", len(self.actions))
        print("act logprobs=", len(self.act_logprobs))
        print("mex_logprobs=", len(self.comm_logprobs))
        print("rewards=", len(self.rewards))
        print("is_terminals=", len(self.is_terminals))
        print("mutinfo:", len(self.mut_info))
        if self.recurrent:
            print("hstates_c=", len(self.hstates_c))
            print("cstates_c=", len(self.cstates_c))
            print("hstates_a=", len(self.hstates_a))
            print("cstates_a=", len(self.cstates_a))