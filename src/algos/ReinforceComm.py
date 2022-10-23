
from src.algos.buffer import RolloutBufferComm
from src.nets.ActorCritic import ActorCritic
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

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
    
        # Communication Policy and Action Policy
        input_comm = self.obs_size
        output_comm = self.mex_size
        self.policy_comm = ActorCritic(params, input_comm, output_comm).to(device)

        input_act = self.obs_size + self.n_agents*self.mex_size
        output_act = self.action_size
        self.policy_act = ActorCritic(params, input_act, output_act).to(device)
        print("policy act input size=", input_act)

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm},
                        {'params': self.policy_comm.critic.parameters(), 'lr': self.lr_critic_comm},
                        {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy_act.critic.parameters(), 'lr': self.lr_critic}])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.train_returns = []
        self.coop = []
        self.reset()

        self.ent = True

        self.saved_losses_comm = []
        self.saved_losses = []

        self.param_entropy = 0.1

        self.eps_norm = 0.0001

    def reset(self):
        self.comm_logprobs = []
        self.act_logprobs = []
        self.comm_entropy = []
        self.act_entropy = []
        self.rewards = []
        self.mutinfo = []
        self.sc = []

    def reset_episode(self):
        self.return_episode = 0
        self.tmp_actions = []

    def select_message(self, state, eval=False):

        if (eval == True):
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                message, message_logprob, entropy = self.policy_comm.act(state, self.ent)

        elif (eval == False):
            state = torch.FloatTensor(state).to(device)
            message, message_logprob, entropy = self.policy_comm.act(state, self.ent)

            self.buffer.states_c.append(state)
            self.buffer.messages.append(message)

            self.comm_logprobs.append(message_logprob)
            self.comm_entropy.append(entropy)

        message = torch.Tensor([message.item()]).long().to(device)
        message = F.one_hot(message, num_classes=self.mex_size)[0]
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

    def select_action(self, state, message, eval=False):
    
        if (eval == True):
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                state_mex = torch.cat((state, message)).to(device)
                print("state_mex=", state_mex, "type=", type(state_mex))
                action, action_logprob, entropy = self.policy_act.act(state_mex, self.ent)

        elif (eval == False):
            state = torch.FloatTensor(state).to(device)
            state_mex = torch.cat((state, message)).to(device)
            action, action_logprob, entropy = self.policy_act.act(state_mex, self.ent)
            
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

            return out

    def update(self):

        rewards =  self.rewards
        rew_norm = [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]
    
        for i in range(len(self.comm_logprobs)):
            #print("rews=", rew_norm[i])
            #print("self.mutinfo=", self.mutinfo[i])
            #print("self.entropy=", self.comm_entropy[i])
            self.comm_logprobs[i] = -self.comm_logprobs[i] * rew_norm[i] + self.mutinfo_param*self.mutinfo[i] #- self.param_entropy*self.comm_entropy[i]
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
        #self.scheduler.step()

        self.reset()