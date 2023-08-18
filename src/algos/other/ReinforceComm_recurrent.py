
from src.algos.buffer import RolloutBufferComm
from src.nets.ActorCritic import ActorCriticRNNcomm
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch.autograd as autograd

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class ReinforceComm():

    def __init__(self, params):

        for key, val in params.items(): setattr(self, key, val)

        self.buffer = RolloutBufferComm()
    
        # Communication and Action Policy
        self.policy = ActorCriticRNNcomm(params).to(device)
        self.optimizer = torch.optim.Adam([{'params': self.policy.parameters(), 'lr': params.lr} ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.policy_old = copy.deepcopy(self.policy).to(device)

        self.MseLoss = nn.MSELoss()

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.train_returns = []
        self.return_episode = 0
        self.tmp_actions = []
        self.coop = []

        self.ent = True

        self.saved_losses_comm = []
        self.saved_losses = []

        self.param_entropy = 0.1

        self.eps_norm = 0.0001

        self.mutinfo_signaling = []
        self.mutinfo_listening = []
        self.mutinfo_signaling_old = []
        self.mutinfo_listening_old = []

        self.reset()

    def reset(self):
        self.comm_logprobs = []
        self.act_logprobs = []
        self.comm_entropy = []
        self.act_entropy = []
        self.rewards = []
        self.mutinfo_signaling_old = self.mutinfo_signaling
        self.mutinfo_listening_old = self.mutinfo_listening
        self.mutinfo_signaling = []
        self.mutinfo_listening = []
        self.sc = []

    def reset_episode(self):
        self.return_episode_old = self.return_episode
        self.return_episode = 0
        self.tmp_actions_old = self.tmp_actions
        self.tmp_actions = []

    def select_message(self, state, eval=False):

        if (eval == True):
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                self.policy_old.observe(state)
                message, message_logprob, entropy = self.policy_old.speak()

        else:
            state = torch.FloatTensor(state).to(device)
            self.policy_old.observe(state)
            message, message_logprob, entropy = self.policy_old.speak()

            self.buffer.hstates_c.append(self.policy_old.hState)
            self.buffer.cstates_c.append(self.policy_old.cState)
            self.buffer.states_c.append(state)
            self.buffer.messages.append(message)
            self.comm_logprobs.append(message_logprob)
            self.comm_entropy.append(entropy)

        return message

    def random_messages(self, state):

        state = torch.FloatTensor(state).to(device)
        message = torch.randint(0, self.mex_size, (self.mex_size-1,))[0]

        self.buffer.states_c.append(state)
        self.buffer.messages.append(message)

        self.comm_logprobs.append(torch.tensor(0.0001))

        message = torch.Tensor([message.item()]).long().to(device)
        message = F.one_hot(message, num_classes=self.mex_size)[0]

        return message

    def select_action(self, state, eval=False):
    
        if (eval == True):
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                self.policy_old.observe(state)
                action, action_logprob,entropy = self.policy_old.act()

        else:

            state = torch.FloatTensor(state).to(device)
            self.policy_old.observe(state)
            action, action_logprob,entropy = self.policy_old.act()
                
            self.buffer.hstates_a.append(self.policy_old.hState)
            self.buffer.cstates_a.append(self.policy_old.cState)

            self.buffer.states_a.append(state)
            self.buffer.actions.append(action)

            self.act_logprobs.append(action_logprob)
            self.act_entropy.append(entropy)
        
        return action

    def get_action_distribution(self, state, message):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            state_mex = torch.cat((state, message))
            out = self.policy_act.get_distribution(state_mex)

            return out

    def get_message_distribution(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            out = self.policy_comm.get_distribution(state)

            return out.detach()

    def update(self):

        rewards =  self.rewards
        rew_norm = [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]
    
        for i in range(len(self.comm_logprobs)):
            self.comm_logprobs[i] = -self.comm_logprobs[i] * rew_norm[i]
            self.act_logprobs[i] = -self.act_logprobs[i] * rew_norm[i]

        self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in self.comm_logprobs])))
        self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in self.act_logprobs])))

        self.optimizer.zero_grad()
        tmp = [torch.ones(a.data.shape) for a in self.comm_logprobs]
        autograd.backward(self.comm_logprobs, tmp, retain_graph=True)

        tmp1 = [torch.ones(a.data.shape) for a in self.act_logprobs]
        autograd.backward(self.act_logprobs, tmp1, retain_graph=True)
        self.optimizer.step()

        #diminish learning rate
        self.scheduler.step()

        self.reset()