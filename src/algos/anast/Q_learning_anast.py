
import torch
import random

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def __len__(self):
        return len(self.memory)

class Q_learning_agent():

    def __init__(self, params, idx=0):

        for key, val in params.items(): setattr(self, key, val)

        self.reputation = torch.bernoulli(torch.Tensor([0.5])) #torch.Tensor([1.0])
        self.old_reputation = self.reputation

        self.is_dummy = False
        self.idx = idx
        print("\nAgent", self.idx)

        # Action Policy
        self.max_value = 0.
        if (self.optimistic_initial_values == 1):
            self.max_value = (self.b_value)/(1.-self.gamma)
        print("self.max_value=", self.max_value)

        input_Q = (self.obs_size, self.action_size)
        if (self.reputation_enabled == 0): 
            input_Q = (self.action_size,)
        else: 
            input_Q = (self.obs_size, self.action_size)
        self.Q = torch.full(input_Q, self.max_value, dtype=float)
        print("self.Q=", self.Q)
        self.actions_list = [i for i in range(self.action_size)]
    
        self.memory = ExperienceReplayMemory(self.num_game_iterations)

        self.reset()

    def append_to_replay(self, s, a, r, s_, d):
        self.memory.push((s, a, r, s_, d))

    def argmax(self, q_values):
        top = torch.Tensor([-10000000])
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return random.choice(ties) #self.rand_generator.choice(ties)

    def reset(self):
        self.previous_action = torch.Tensor([1.])
        self.return_episode = 0

    def digest_input_anast(self, input):
        opponent_reputation = input

        opponent_reputation = torch.Tensor([opponent_reputation])
        self.state_act = torch.Tensor([opponent_reputation])

    def select_action(self, _eval=False):
        #print("SELECT ACTION")
        state_to_act = self.state_act
        #print("state=", state_to_act)

        if (self.reputation_enabled == 0):
            current_q = self.Q
        else: 
            #print("Q=", self.Q)
            current_q = self.Q[state_to_act[0].long(),:]
        #print("current_q=", current_q)
        assert(current_q.shape == torch.Size([self.action_size]))

        if (_eval == True):
            #print("eval")
            action = self.argmax(current_q)
        elif (_eval == False):
            if torch.rand(1) < self.epsilon:
                #print("random")
                action = random.choice(self.actions_list)
            else:
                #print("argmax")
                action = self.argmax(current_q)
                
        return torch.Tensor([action])
    
    def get_action_distribution(self):
        return self.Q[self.state_act.long(),:]
        
    def select_opponent(self, reputations, _eval=False):
        pass
    
    def update(self):
        #print("\nIN UPDATE")
        
        for i in range(self.num_game_iterations):
            state, action, reward, next_state, done = self.memory.memory[i]
            state = state.long()
            action = action.long()
            next_state = next_state.long()
            #print("state=", state)
            #print("action=", action)
            #print("next_state=", next_state)
            #print("Q=", self.Q)
            #print("self.Q[state, action]=", self.Q[state, action])
            #print("self.Q[next_state,:]=",self.Q[next_state,:])
            if (self.reputation_enabled == 0):
                if (done):
                    self.Q[action] += self.lr_actor*(reward - self.Q[action])
                else:
                    self.Q[action] += self.lr_actor*(reward + self.gamma*torch.max(self.Q[:]) - self.Q[action])
            else:
                if (done):
                    self.Q[state, action] += self.lr_actor*(reward - self.Q[state, action])
                else:
                    self.Q[state, action] += self.lr_actor*(reward + self.gamma*torch.max(self.Q[next_state,:][0]) - self.Q[state, action])

        self.memory.memory = []
        self.reset()
        #print("agent=", self.idx, "Q=", self.Q)
        #print("Q=", self.Q)