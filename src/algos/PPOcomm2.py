
from dis import disco
from src.nets.ActorCritic import ActorCriticDiscrete
import torch
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBufferComm:
    def __init__(self):
        self.states = []
        self.messages_out = []
        self.state_mex = []
        self.actions = []
        self.act_logprobs = []
        self.comm_logprobs = []
        self.rewards = []
        self.mut_info = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.messages_out[:]
        del self.state_mex[:]
        del self.actions[:]
        del self.act_logprobs[:]
        del self.comm_logprobs[:]
        del self.rewards[:]
        del self.mut_info[:]
        del self.is_terminals[:]

    def __print__(self):
        print("states=", len(self.states))
        print("messages_out=", len(self.messages_out))
        print("state_mex=", len(self.state_mex))
        print("actions=", len(self.actions))
        print("act logprobs=", len(self.act_logprobs))
        print("mex_logprobs=", len(self.comm_logprobs))
        print("rewards=", len(self.rewards))
        print("mut info=", len(self.mut_info))
        print("is_terminals=", len(self.is_terminals))


class PPOcomm2():

    def __init__(self, n_agents, obs_dim, action_dim, mex_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, c1, c2, c3, c4):

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.mex_dim = mex_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.c1 = c1 
        self.c2 = c2
        self.c3 = 0
        self.c4 = c4

        self.buffer = RolloutBufferComm()
    
        # Communication Policy
        self.policy_comm = ActorCriticDiscrete(obs_dim, mex_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy_comm.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy_comm.critic.parameters(), 'lr': lr_critic},
                    ])

        self.policy_comm_old = ActorCriticDiscrete(obs_dim, mex_dim).to(device)
        self.policy_comm_old.load_state_dict(self.policy_comm.state_dict())

        # Action Policy
        self.policy_act = ActorCriticDiscrete(obs_dim+mex_dim*n_agents, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy_act.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy_act.critic.parameters(), 'lr': lr_critic},
                    ])

        self.policy_act_old = ActorCriticDiscrete(obs_dim + mex_dim*n_agents, action_dim).to(device)
        self.policy_act_old.load_state_dict(self.policy_act.state_dict())

        self.MseLoss = nn.MSELoss()

        self.train_returns = []
        self.tmp_return = 0
        self.train_actions = []
        self.tmp_actions = []
        self.coop = []

    def select_mex(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            message, message_logprob = self.policy_comm_old.act(state)

            self.buffer.states.append(state)
            self.buffer.messages_out.append(message)
            self.buffer.comm_logprobs.append(message_logprob)

        message = torch.Tensor([message.item()]).long()
        message = F.one_hot(message, num_classes=self.mex_dim)[0]
        return message  

    def select_action(self, state, message):
    
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            state_mex = torch.cat((state, message))
            action, action_logprob = self.policy_act_old.act(state_mex)

            self.buffer.state_mex.append(state_mex)
            self.buffer.actions.append(action)
            self.buffer.act_logprobs.append(action_logprob)

        return action.item()

    def eval_mex(self, state):
    
        messages_logprobs = []
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            for mex in range(self.mex_dim):
                m = torch.Tensor([mex])
                _, mex_logprob, _= self.policy_comm_old.evaluate(state, m)
                messages_logprobs.append(mex_logprob.item())

        return messages_logprobs

    def eval_action(self, state, message):
    
        actions_logprobs = []
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            for action in range(self.action_dim):
                a = torch.Tensor([action])
                _, action_logprob, _= self.policy_act_old.evaluate(state, message, a)
                actions_logprobs.append(action_logprob.item())

        return actions_logprobs

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):#, reversed(self.buffer.mut_info)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            #discounted_reward = reward + m_info + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        #print("self.policy_comm.input_dim=", self.policy_comm.input_dim)
        #print("self.policy_act.input_dim=", self.policy_act.input_dim)
        if (self.policy_comm.input_dim == 1):
            old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
            old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
            old_messages_out = torch.stack(self.buffer.messages_out, dim=0).detach().to(device)
            old_states_mex = torch.stack(self.buffer.state_mex, dim=0).detach().to(device)
            old_logprobs_act = torch.stack(self.buffer.act_logprobs, dim=0).detach().to(device)
            old_logprobs_comm = torch.stack(self.buffer.comm_logprobs, dim=0).detach().to(device)
        else:
            old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            old_messages_out = torch.squeeze(torch.stack(self.buffer.messages_out, dim=0)).detach().to(device)
            old_states_mex = torch.squeeze(torch.stack(self.buffer.state_mex, dim=0)).detach().to(device)
            old_logprobs_act = torch.squeeze(torch.stack(self.buffer.act_logprobs, dim=0)).detach().to(device)
            old_logprobs_comm = torch.squeeze(torch.stack(self.buffer.comm_logprobs, dim=0)).detach().to(device)

        #print("actions=", old_actions.shape)
        #print("states=", old_states.shape)
        #print("messagess_out=", old_messages_out.shape)
        #print("states_mex=", old_states_mex.shape)

        for _ in range(self.K_epochs):
  
            logprobs_comm, dist_entropy_mex, state_values_comm = self.policy_comm.evaluate(old_states, old_messages_out)
            logprobs_act, dist_entropy_act, state_values_act = self.policy_act.evaluate(old_states_mex, old_actions)

            state_values_act = torch.squeeze(state_values_act)
            #print("states values act", state_values_act.shape)
            state_values_comm = torch.squeeze(state_values_comm)
            #print("states values comm", state_values_comm.shape)
            #print("rewards=", rewards.shape)

            ratios_act = torch.exp(logprobs_act - old_logprobs_act.detach())
            ratios_comm = torch.exp(logprobs_comm - old_logprobs_comm.detach())

            advantages_act = rewards - state_values_act.detach()
            # here I have to undrstand if it is better to use rewards or the entropy distribution as "communication reward"
            advantages_comm = -dist_entropy_mex - state_values_comm.detach()

            surr1 = ratios_act*advantages_act + ratios_comm*advantages_comm
            surr2 = torch.clamp(ratios_act, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_act
            surr3 = torch.clamp(ratios_comm, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_comm
            
            surr12 = torch.min(surr1, surr2)

            loss = (-torch.min(surr12, surr3) + \
                self.c1*self.MseLoss(state_values_act, rewards) + self.c2*dist_entropy_act + \
                self.c3*self.MseLoss(state_values_comm, rewards) + self.c4*dist_entropy_mex)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_comm_old.load_state_dict(self.policy_comm.state_dict())
        self.policy_act_old.load_state_dict(self.policy_act.state_dict())

        # clear buffer
        self.buffer.clear()