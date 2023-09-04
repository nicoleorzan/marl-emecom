
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
        self.max_value = max(self.mult_fact)*self.coins_value#/(1.-self.gamma)
        if (len(self.mult_fact) == 1):
            input_Q = (self.obs_size, self.action_size)
        else: 
            input_Q = (len(self.mult_fact), self.obs_size, self.action_size)
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

        return random.choice(ties)

    def reset(self):
        self.previous_action = torch.Tensor([1.])
        self.return_episode = 0

    def digest_input_anast(self, input):
        opponent_reputation = input

        opponent_reputation = torch.Tensor([opponent_reputation])
        self.state_act = torch.Tensor([opponent_reputation])

    def select_action(self, _eval=False):
        
        state_to_act = self.state_act
        #print("self.state_act=",self.state_act)
        if (self.obs_size == 1):
            current_q = self.Q[state_to_act[0].long(),:]
        else:
            current_q = torch.take(self.Q, state_to_act.long())
        #print("current_q=",current_q, current_q.shape)
        assert(current_q.shape == torch.Size([self.action_size]))

        if (_eval == True):
            #print("eval")
            action = self.argmax(current_q)
        elif (_eval == False):  
            #print("no eval") 

            #print("self.epsilon",self.epsilon)
            #print("torch.rand(1)=",torch.rand(1))
            if torch.rand(1) < self.epsilon:
                #print("RANDOM")
                action = random.choice([i for i in range(self.action_size)])
            else:
                #print("argmax")
                action = self.argmax(current_q)
        #print("action=", action)
        
        return torch.Tensor([action])
    
    def get_action_distribution(self):
        return self.Q[self.state_act.long(),:]
        
    def select_opponent(self, reputations, _eval=False):
        pass
    
    def update(self):
        #print("memory=", self.memory.memory)
        
        for i in range(self.num_game_iterations):
            state, action, reward, next_state, done = self.memory.memory[i]
            state = state.long()
            action = action.long()
            next_state = next_state.long()
            #print("state=", state)
            #print("action=", action)
            #print("next_state=", next_state)
            if (done):
                #print("done")
                if (len(self.mult_fact) == 1 or self.obs_size == 1): 
                    #print("mfs 1")
                    self.Q[state[0], action] += self.lr_actor*(reward - self.Q[state[0], action])
                else:
                    #print("mfs more than 1")
                    self.Q[state[0], state[1], action] += self.lr_actor*(reward - self.Q[state[0], state[1], action])
            else:
                if (len(self.mult_fact) == 1 or self.obs_size == 1):
                    #print("here") 
                    self.Q[state[0], action] += self.lr_actor*(reward + self.gamma*torch.max(self.Q[next_state,:][0])  - self.Q[state[0], action])
                else:
                    #print("self.Q=",self.Q)
                    #print("self.Q[state[0], state[1], action]=",self.Q[state[0], state[1], action])
                    #print("self.Q[state[0], state[1], action]=",self.Q[state[0], state[1], action])
                    self.Q[state[0], state[1], action] += self.lr_actor*(reward - self.Q[state[0], state[1], action])
                    #self.Q[state[0], state[1], action] += self.lr_actor*(reward + self.gamma*torch.max(torch.take(self.Q, next_state)) - self.Q[state[0], state[1], action])

        self.memory.memory = []
        self.reset()