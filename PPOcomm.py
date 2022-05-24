
from dis import disco
import torch
import torch.nn as nn
from torch.distributions import Categorical
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
        self.messages = []
        self.actions = []
        self.act_logprobs = []
        self.mex_logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.messages[:]
        del self.actions[:]
        del self.act_logprobs[:]
        del self.mex_logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def __print__(self):
        print("states=", len(self.states))
        print("messages=", len(self.messages))
        print("actions=", len(self.actions))
        print("act logprobs=", len(self.act_logprobs))
        print("mex_logprobs=", len(self.mex_logprobs))
        print("rewards=", len(self.rewards))
        print("is_terminals=", len(self.is_terminals))


class ActorCriticComm(nn.Module):

    def __init__(self, obs_dim, mex_dim, action_dim): # input = obs + mex
        super(ActorCriticComm, self).__init__()

        self.obs_dim = obs_dim
        self.mex_dim = mex_dim
        self.input_dim = self.obs_dim + self.mex_dim
        self.hidden_dim = 64
        self.action_dim = action_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.action_actor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.action_critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.comm_actor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.mex_dim),
        )
        self.comm_critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def act(self, state): # state is state + mex already concat

        print(state)
        #out = torch.cat((state, mex), dim=0)
        #print("out=", out)
        out = self.layers(state)
        mex_probs = self.comm_actor(out)
        act_probs = self.action_actor(out)
        
        dist_mex = Categorical(logits=mex_probs) # here I changed probs with logits!!!
        dist_act = Categorical(logits=act_probs) # here I changed probs with logits!!!
        
        mex = dist_mex.sample()
        act = dist_act.sample()
        
        logprob_mex = dist_mex.log_prob(mex)
        logprob_act = dist_act.log_prob(act)

        return act.detach(), mex.detach(), logprob_act.detach(), logprob_mex.detach()

    def evaluate(self, state, mex_out, action):

        out = self.layers(state)
        mex_probs = self.comm_actor(out)
        act_probs = self.action_actor(out)

        dist_mex = Categorical(logits=mex_probs) # here I changed probs with logits!!!
        dist_act = Categorical(logits=act_probs) # here I changed probs with logits!!!

        logprob_mex = dist_mex.log_prob(mex_out)
        logprob_act = dist_act.log_prob(action)

        dist_entropy_mex = dist_mex.entropy()
        dist_entropy_act = dist_act.entropy()

        state_values_mex = self.action_critic(out)
        state_values_act = self.comm_critic(out)
        return logprob_mex, logprob_act, dist_entropy_mex, dist_entropy_act, state_values_mex, state_values_act


class PPOcomm():

    def __init__(self, obs_dim, action_dim, mex_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, c1, c2):

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

        self.buffer = RolloutBufferComm()
    
        self.policy = ActorCriticComm(obs_dim, mex_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.layers.parameters(), 'lr': lr_actor},
                        {'params': self.policy.action_actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.comm_actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.action_critic.parameters(), 'lr': lr_critic},
                        {'params': self.policy.comm_critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCriticComm(obs_dim, mex_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.train_returns = []
        self.tmp_return = 0

    def select_action(self, state, done=False):
    
        print("inside")
        #if not done:
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, message, action_logprob, message_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.messages.append(message)
            self.buffer.actions.append(action)
            self.buffer.act_logprobs.append(action_logprob)
            self.buffer.mex_logprobs.append(message_logprob)
        return action.item(), message.item()

    def eval_action(self, state, mex_in, done=False):
    
        actions_logprobs = []
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            for action in range(self.action_dim):
                a = torch.Tensor([action])
                _, action_logprob, _, _, _, _ = self.policy_old.evaluate(state, mex_in, a)
                print("action_logprob=",action_logprob)
                actions_logprobs.append(action_logprob.item())

        return actions_logprobs

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_messages = torch.squeeze(torch.stack(self.buffer.messages, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.act_logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
  
            logprob_mex, logprob_act, dist_entropy_mex, dist_entropy_act, state_values_mex, state_values_act = self.policy.evaluate(old_states, old_messages, old_actions)
            #logprobs, dist_entropy, state_values = self.policy.evaluate(old_states, old_messages, old_actions)

            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios*advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages

            loss = (-torch.min(surr1, surr2) + self.c1*self.MseLoss(state_values, rewards) + self.c2*dist_entropy)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()