
from src.algos.agent import Agent
from src.algos.buffer import DQNBuffer
from src.nets.Actor import Actor
import copy
import torch
import torch.nn as nn

#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class DQN(Agent):

    def __init__(self, params, idx=0):
        Agent.__init__(self, params, idx)

        opt_params = []

        self.policy_act = Actor(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, gmm=self.gmm_).to(device)
        self.policy_act_target = copy.deepcopy(self.policy_act).to(device)
        self.policy_act_target.load_state_dict(self.policy_act.state_dict())

        opt_params.append({'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor})
        #opt_params.append({'params': self.policy_act.critic.parameters(), 'lr': self.lr_critic})

        if (self.is_communicating):
            self.policy_comm = Actor(params=params, input_size=self.input_comm, output_size=self.mex_size, \
                n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, gmm=self.gmm_).to(device)
            self.policy_comm_target = copy.deepcopy(self.policy_comm).to(device)
            self.policy_comm_target.load_state_dict(self.policy_comm.state_dict())
            
            opt_params.append({'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm})
            #opt_params.append({'params': self.policy_comm.critic.parameters(), 'lr': self.lr_critic_comm})

        self.buffer = DQNBuffer(self.memory_size)

        self.optimizer = torch.optim.Adam(opt_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

    def update(self):
        #print("update")

        if self.buffer.len_() < self.batch_size:
            return
    
        transitions = self.buffer.sample(self.batch_size)
        #state_batch, action_batch, reward_batch, next_state_batch, is_terminal = transitions
        state_batch, action_batch, reward_batch, next_state_batch = transitions

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        #batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                        next_state_batch)), device=device, dtype=torch.bool)
        #non_final_next_states = torch.cat([s for s in next_state_batch
        #                                            if s is not None])
        #print("before state_batch=", state_batch)
        #state_batch = torch.Tensor(state_batch)
        #print("after cat=", state_batch)

        #b = torch.Tensor(len(state_batch), 3, 1)
        #torch.cat(state_batch, out=b)
        #print("b=", b )

        state_batch = torch.stack(state_batch) #torch.cat(state_batch.unsqueeze(0))
        #print("\nstate_batch=", state_batch, state_batch.shape)
        #next_state_batch = torch.stack(next_state_batch) #torch.cat(next_state_batch).unsqueeze(1)
        #print("next_state_batch=", next_state_batch, next_state_batch.shape)
        action_batch = torch.stack(action_batch)
        #print("action_batch", action_batch, action_batch.shape)
        #print("before reward_batch=", reward_batch)
        #reward_batch = torch.cat(reward_batch)
        # Normalizing the rewards
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(device)
        reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-7)
        #print("after reward_batch=", reward_batch, reward_batch.shape)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_act.get_values(state_batch).gather(1, action_batch.unsqueeze(1))
        #print("intermed state act val=", state_action_values, state_action_values.shape)
        #state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1))
        #print("state act val=", state_action_values, state_action_values.shape)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        #next_state_values = torch.zeros(self.batch_size, device=device)
        #with torch.no_grad():
        #    next_state_values = self.policy_act_target.get_values(next_state_batch).max(1)[0]
        # Compute the expected Q values
        #print("next_state_values=",next_state_values)
        #next_state_values = torch.zeros(next_state_values.shape)
        #print("next_state_values=",next_state_values)
        expected_state_action_values = reward_batch # = (next_state_values * self.gamma) + reward_batch
        #print("expected_state_action_values=",expected_state_action_values)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.saved_losses.append(torch.mean(loss.detach()))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_act.parameters(), 100)
        self.optimizer.step()

        target_net_state_dict = self.policy_act_target.state_dict()
        policy_net_state_dict = self.policy_act.state_dict()
        #for key in policy_net_state_dict:
        #    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.policy_act_target.load_state_dict(target_net_state_dict)

        self.scheduler.step()
        #print(self.scheduler.get_lr())

        #self.reset()