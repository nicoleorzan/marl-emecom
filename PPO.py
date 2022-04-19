
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

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):

    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = 64
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def act(self, state):
        #state = torch.from_numpy(state).view(1,-1)
        #print("ACT")
        #print("state=", state.shape)
        out = self.actor(state)
        #print("probs=", out, torch.sum(out), out.shape, out.size(-1))
        dist = Categorical(logits=out) # here I changed probs with logits!!!
        #print("dist=", dist)
        act = dist.sample()
        #print("sampled act=", act)
        logprob = dist.log_prob(act)

        return act.detach(), logprob.detach()

    def evaluate(self, state, action):
        #print("EVALUATE")
        #print("state=", state.shape)
        action_probs = self.actor(state)
        #print("probs=", action_probs.shape)
        dist = Categorical(logits=action_probs)  # here I changed probs with logits!!!
        action_logprobs = dist.log_prob(action)
        #print("action_loprobs=", action_logprobs.shape)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, dist_entropy, state_values


class PPO():

    def __init__(self, input_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, c1, c2):

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.c1 = c1 
        self.c2 = c2

        self.buffer = RolloutBuffer()
    
        self.policy = ActorCritic(input_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(input_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.train_returns = []
        self.tmp_return = 0

    def select_action(self, state, done=False):
    
        #if not done:
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)

            #print("state=", state.shape)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
        return action.item()
        
        #else: 
        #    return None

    def update(self):
        # Monte carlo estimate of returns
        #print("=====>UPDATE")
        rewards = []
        #print("buffer rewards", len(self.buffer.rewards))
        #print("buffer is terminal", len(self.buffer.is_terminals))
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            #print("term=", is_terminal)
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        #print("rewards=", rewards)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        #print("rewards normalized=", rewards.shape)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            #print("old_states=", old_states.shape)
            #print("old_actions=", old_actions.shape)

            logprobs, dist_entropy, state_values = self.policy.evaluate(old_states, old_actions)

            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            #print("state_values=", state_values.shape)
            #print("rewards=", rewards.shape)
            advantages = rewards - state_values.detach()
            surr1 = ratios*advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages

            loss = (-torch.min(surr1, surr2) - self.c1*self.MseLoss(state_values, rewards) - self.c2*dist_entropy)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()