
from src.algos.agent import Agent
from src.nets.ActorCritic import ActorCritic
import torch
import copy

#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

class PPO(Agent):

    def __init__(self, params, idx=0):
        Agent.__init__(self, params, idx)

        opt_params = []

        self.policy_act = ActorCritic(params=params, input_size=self.input_act, output_size=self.action_size, \
            n_hidden=self.n_hidden_act, hidden_size=self.hidden_size_act, gmm=self.gmm_).to(device)
        self.policy_act_old = copy.deepcopy(self.policy_act).to(device)
        self.policy_act_old.load_state_dict(self.policy_act.state_dict())
        opt_params.append({'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor})
        opt_params.append({'params': self.policy_act.critic.parameters(), 'lr': self.lr_critic})

        if (self.is_communicating):
            self.policy_comm = ActorCritic(params=params, input_size=self.input_comm, output_size=self.mex_size, \
                    n_hidden=self.n_hidden_comm, hidden_size=self.hidden_size_comm, gmm=self.gmm_).to(device)
            self.policy_comm_old = copy.deepcopy(self.policy_comm).to(device)
            self.policy_comm_old.load_state_dict(self.policy_comm.state_dict())
            opt_params.append({'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm})
            opt_params.append({'params': self.policy_comm.critic.parameters(), 'lr': self.lr_critic_comm})

        self.optimizer = torch.optim.Adam(opt_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)
        self.MseLoss = torch.nn.MSELoss()

    def eval_mex(self, state_c):
    
        messages_logprobs = []
        with torch.no_grad():
            state_c = torch.FloatTensor(state_c).to(device)
            for mex in range(self.mex_size):
                m = torch.Tensor([mex])
                _, mex_logprob, _= self.policy_comm_old.evaluate(state_c, m)
                messages_logprobs.append(mex_logprob.item())

        return messages_logprobs

    def eval_action(self, state_a, message):
    
        actions_logprobs = []
        with torch.no_grad():
            state_a = torch.FloatTensor(state_a).to(device)
            for action in range(self.action_size):
                a = torch.Tensor([action])
                _, action_logprob, _= self.policy_act_old.evaluate(state_a, message, a)
                actions_logprobs.append(action_logprob.item())

        return actions_logprobs

    def update(self):
        rewards = []
        discounted_reward = 0
        #for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        if (self.policy_act.input_size == 1):
            old_states_a = torch.stack(self.buffer.states_a, dim=0).detach().to(device)
            old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
            old_logprobs_act = torch.stack(self.buffer.act_logprobs, dim=0).detach().to(device)
        else:
            old_states_a = torch.squeeze(torch.stack(self.buffer.states_a, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            old_logprobs_act = torch.squeeze(torch.stack(self.buffer.act_logprobs, dim=0)).detach().to(device)

        if (self.is_communicating):
            if (self.policy_comm.input_size == 1):
                old_states_c = torch.stack(self.buffer.states_c, dim=0).detach().to(device)
                old_messages = torch.stack(self.buffer.messages, dim=0).detach().to(device)
                old_logprobs_comm = torch.stack(self.buffer.comm_logprobs, dim=0).detach().to(device)
            else:
                old_states_c = torch.squeeze(torch.stack(self.buffer.states_c, dim=0)).detach().to(device)
                old_messages = torch.squeeze(torch.stack(self.buffer.messages, dim=0)).detach().to(device)
                old_logprobs_comm = torch.squeeze(torch.stack(self.buffer.comm_logprobs, dim=0)).detach().to(device)
    
        for _ in range(self.K_epochs):

            if (self.is_communicating):
                logprobs_comm, dist_entropy_comm, state_values_comm = self.policy_comm.evaluate(old_states_c, old_messages)
                state_values_comm = torch.squeeze(state_values_comm)
                ratios_comm = torch.exp(logprobs_comm - old_logprobs_comm.detach())
                mutinfo = torch.tensor(self.buffer.mut_info, dtype=torch.float32).to(device)
                advantages_comm = rewards - state_values_comm.detach()
                surr1c = ratios_comm*advantages_comm
                surr2c = torch.clamp(ratios_comm, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_comm

                loss_c = (-torch.min(surr1c, surr2c) + \
                    self.c3*self.MseLoss(state_values_comm, rewards) - \
                    self.c4*dist_entropy_comm)

            #print(old_states_a[0], old_actions[0])
            logprobs_act, dist_entropy_act, state_values_act = self.policy_act.evaluate(old_states_a, old_actions)
            state_values_act = torch.squeeze(state_values_act)
            ratios_act = torch.exp(logprobs_act - old_logprobs_act.detach())
            advantages_act = rewards - state_values_act.detach()
            surr1a = ratios_act*advantages_act
            surr2a = torch.clamp(ratios_act, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_act

            loss_a = (-torch.min(surr1a, surr2a) + \
                self.c1*self.MseLoss(state_values_act, rewards) - \
                self.c2*dist_entropy_act)
            #print("loss_a=", loss_a)

            if (self.is_communicating):
                loss_a += loss_c
                #self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in self.comm_logprobs])))
                #tmp = [torch.ones(a.data.shape) for a in self.comm_logprobs]
                #autograd.backward(self.comm_logprobs, tmp, retain_graph=True)
            self.saved_losses.append(torch.mean(loss_a.detach()))

            self.optimizer.zero_grad()
            loss_a.mean().backward()
            self.optimizer.step()
            #print(self.scheduler.get_lr())

        self.scheduler.step()
        #print(self.scheduler.get_lr())

        # Copy new weights into old policy
        self.policy_act_old.load_state_dict(self.policy_act.state_dict())
        if (self.is_communicating):
            self.policy_comm_old.load_state_dict(self.policy_comm.state_dict())

        # clear buffer
        self.buffer.clear()
        self.reset()