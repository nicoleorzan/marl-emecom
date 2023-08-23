from src.algos.anast.Actor import Actor
import copy
import torch
import random
import torch.nn as nn
from collections import namedtuple

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self._states = torch.empty((self.capacity,1))
        self._actions = torch.empty((self.capacity,1))
        self._rewards = torch.empty((self.capacity,1))
        self._next_states = torch.empty((self.capacity,1))
        self._dones = torch.empty((self.capacity,1), dtype=torch.bool)
        self.i = 0

    """def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None, None"""

    def __len__(self):
        return len(self._states)


class DQN(nn.Module):
    def __init__(self, params, idx=0):
        super(DQN, self).__init__()

        for key, val in params.items(): setattr(self, key, val)

        self.input_act = self.obs_size

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(device)
        self.policy_act_target = copy.deepcopy(self.policy_act).to(device)
        self.policy_act_target.load_state_dict(self.policy_act.state_dict())

        self.optimizer = torch.optim.RMSprop(self.policy_act.parameters())
        self.memory = ExperienceReplayMemory(self.num_game_iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.reputation = torch.Tensor([1.])
        self.old_reputation = self.reputation
        self.previous_action = torch.Tensor([1.])

        self.is_dummy = False
        self.idx = idx
        self.batch_size = self.num_game_iterations

        self.action_selections = [0 for _ in range(self.action_size)]
        self.action_log_frequency = 1.

        self.learn_start = 10
        self.update_freq = 1
        self.update_count = 0
        self.target_net_update_freq = 10

    def reset(self):
        self.memory.reset()
        self.memory.i = 0
            
    def select_action(self, _eval=False):
        self.state_act = self.state_act.view(-1,1)

        greedy = False
        if (_eval == True):
            greedy = True

        with torch.no_grad():
            action, _, _, dist = self.policy_act.act(state=self.state_act, greedy=greedy, get_distrib=True)
            if torch.rand(1) < self.epsilon:
                action = torch.randint(0, self.action_size, (1,))[0]

            return action, dist
        
    def argmax(self, q_values):
        top = torch.Tensor([-10000000])
        ties = []

        #print("q_values=",q_values)
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return random.choice(ties)
        
    def select_action_eps_greedy(self, _eval=False):
        
        self.state_act = self.state_act.view(-1,1)

        with torch.no_grad():
            dist = self.policy_act.get_distribution(state=self.state_act)

        if (_eval == True):
            action = self.argmax(dist)
        elif (_eval == False):   
            if torch.rand(1) < self.epsilon:
                action = random.choice([i for i in range(self.action_size)])
            else:
                action = self.argmax(dist)
                
        return torch.Tensor([action]), dist
    
    def get_action_distribution(self, state):

        with torch.no_grad():
            out = self.policy_act.get_distribution(state)
            return out

    def append_to_replay(self, s, a, r, s_, d):
        #self.memory.push((s, a, r, s_, d))
        self.memory._states[self.memory.i] = s
        self.memory._actions[self.memory.i] = a
        self.memory._rewards[self.memory.i] = r
        self.memory._next_states[self.memory.i] = s_
        self.memory._dones[self.memory.i] = d
        self.memory.i += 1

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory (nope, take all memory!)
        #transitions, indices, weights = self.memory.sample(self.batch_size)
        
        #batch_state, batch_action, batch_reward, batch_next_state, batch_dones = zip(*transitions)
        #batch_state = torch.cat(batch_state).view(-1,self.obs_size)
        #print("0 batch state=", batch_state, type(batch_state))
        #print("0 batch action=", batch_action)
        batch_state = self.memory._states
        batch_action = self.memory._actions
        batch_reward = self.memory._rewards
        batch_next_state = self.memory._next_states
        #batch_dones = self.memory._dones
        
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(-1,1)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values
    
    def read_q_matrix(self):
        possible_states = torch.Tensor([[0.], [1.]])
        Q = torch.full((2, 2),  0.)
        for state in possible_states:
            Q[state.long(),:] = self.policy_act.get_values(state.view(-1,1))
            #Q[state.long(),:] = self.policy_act.get_distribution(state.view(-1,1)).detach()
        return Q

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars
    
        current_q_values = self.policy_act.get_values(batch_state)
        current_q_values = torch.gather(current_q_values, dim=1, index=batch_action)
        #print("current_q_values=",torch.mean(current_q_values))
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                _, _, _, dist = self.policy_act_target.act(state=non_final_next_states, greedy=False, get_distrib=True)
                max_next_q_values[non_final_mask] = dist.gather(1, max_next_action)# (non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + self.gamma*max_next_q_values
            #print("expected_q_values=",torch.mean(expected_q_values))

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()
        #print("loss=", loss)

        return loss

    def update(self, frame=0):
        #print("\n\n\n\n\n\n===>UPDATING!")

        #if frame < self.learn_start or frame % self.update_freq != 0:
        #    return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_act.parameters():
        #    print("param=", param)
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        #self.save_td(loss.item(), frame)
        #self.save_sigma_param_magnitudes(frame)

        self.reset()

        return loss.detach()

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.policy_act_target.load_state_dict(self.policy_act.state_dict())

    def get_max_next_state_action(self, next_states):
        _, _, _, dist = self.policy_act_target.act(state=next_states, greedy=False, get_distrib=True)
        max_vals = dist.max(dim=1)[1].view(-1, 1)
        return max_vals

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)