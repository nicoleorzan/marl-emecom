from src.algos.anast.Actor import Actor
import copy
import torch
import numpy as np
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
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None, None

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, params, idx=0):
        super(DQN, self).__init__()

        for key, val in params.items(): setattr(self, key, val)

        opt_params = []

        self.gamma = 0.99
        self.input_act = self.obs_size

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act).to(device)
        self.policy_act_target = copy.deepcopy(self.policy_act).to(device)
        self.policy_act_target.load_state_dict(self.policy_act.state_dict())

        opt_params.append({'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor})

        self.optimizer = torch.optim.RMSprop(self.policy_act.parameters())
        self.memory = ExperienceReplayMemory(10000)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.reputation = torch.Tensor([1.])
        self.old_reputation = self.reputation
        self.previous_action = torch.Tensor([1.])

        self.is_dummy = False
        self.idx = idx

        self.action_selections = [0 for _ in range(self.action_size)]
        self.action_log_frequency = 1.

        self.learn_start = 10
        self.update_freq = 1
            
    def select_action(self, state, eps=0.1):
        with torch.no_grad():
            if np.random.random() >= eps:
                action, _, _ = self.policy_act.act(state)
            else: 
                action = torch.randint(0, self.action_size, (1,))[0]
            return action
    
    def get_action_distribution(self, state):

        with torch.no_grad():
            out = self.policy_act.get_distribution(state)
            return out

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        print("batch state=", batch_state, type(batch_state))

        #shape = (-1,)+self.obs_size
        batch_state = torch.cat(batch_state).view(-1,self.obs_size)
        print("batch state=", batch_state)
        #batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(-1)
        #print("batch state=", batch_state)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        print("batch action=", batch_action)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        print("batch_reward=", batch_reward)
        print("batch next state BEFORE=", batch_next_state)

        #batch_next_state = torch.cat(batch_next_state).view(-1,self.obs_size)
        #print("batch next state=", batch_next_state)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        print("non_final_mask=",non_final_mask)
        print("[s for s in batch_next_state if s is not None]=",[s for s in batch_next_state if s is not None])
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float)#.view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
        print("non_final_next_states=", non_final_next_states)

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars
        print("compute loss")
        print("empty_next_state_values=",empty_next_state_values)
        #estimate
        
        current_q_values, _, _ = self.policy_act.act(batch_state)#.gather(1, batch_action)
        print("current q vals=", current_q_values)
        print("current q shape=", current_q_values.shape)
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            print("max_next_q_values=",max_next_q_values)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                print("max_next_action=", max_next_action)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
                print("max_next_q_values[non_final_mask]=",max_next_q_values[non_final_mask])
            expected_q_values = batch_reward + self.gamma*max_next_q_values

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, frame=0):
        print("frame=", frame)

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None
        print("\n\n\n\n\n\n===>UPDATING!")

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)