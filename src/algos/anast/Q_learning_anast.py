
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

        self.reputation = torch.Tensor([1.0])
        self.old_reputation = self.reputation

        self.is_dummy = False
        self.idx = idx
        print("\nAgent", self.idx)

        # Action Policy
        self.max_value = self.b_value+self.c_value

        print("hasattr=",hasattr(self, "mult_fact"))
        if (hasattr(self, "mult_fact")): 
            input_Q = (len(self.mult_fact), self.obs_size, self.action_size)
        else:
            input_Q = (self.obs_size, self.action_size)
        self.Q = torch.full(input_Q, self.max_value, dtype=float)
        print("self.Q=", self.Q)
    
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
        
        state_to_act = self.state_act
        current_q = self.Q[state_to_act[0].long(),:]
        #current_q = torch.take(self.Q, state_to_act.long())
        #print("current_q1=", current_q1)
        #print("current_q=", current_q)
        #print(torch.equal(current_q1,current_q))

        if (_eval == True):
            action = self.argmax(current_q)
            #assert(action == self.argmax(current_q1))
        elif (_eval == False):   

            if torch.rand(1) < self.epsilon:
                action = random.choice([i for i in range(self.action_size)])
            else:
                action = self.argmax(current_q)
                #print("action=", action, type(action))
                #a1 = self.argmax(current_q1)
                #print("self.argmax(current_q1)=", a1, type(a1))
                #print(action, a1)
                #assert(action == a1)
                
        return torch.Tensor([action]), current_q
    
    def get_action_distribution(self):
        return self.Q[self.state_act.long(),:]
        
    def select_opponent(self, reputations, _eval=False):
        pass
    
    def update(self):
        
        for i in range(self.num_game_iterations):
            state, action, reward, next_state, done = self.memory.memory[i]
            #print("state, action, reward, next_state, done=",state, action, reward, next_state, done)
            state = state.long()
            action = action.long()
            next_state = next_state.long()
            if (done):
                self.Q[state, action] += self.lr_actor*(reward - self.Q[state, action])
            else:
                #self.Q[state, action] += self.lr_actor*(reward + self.gamma*self.argmax(self.Q[next_state,:][0]) - self.Q[state, action])
                self.Q[state, action] += self.lr_actor*(reward + self.gamma*self.argmax(torch.take(self.Q, next_state.long())) - self.Q[state, action])

        #print("self.Q=",self.Q)

        self.memory.memory = []
        self.reset()