
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

    def __init__(self, params, sign_lambda=0.0, list_lambda=0.0):

        for key, val in params.items(): setattr(self, key, val)

        self.buffer = RolloutBufferComm()
    
        # Communication Policy and Action Policy
        input_comm = self.obs_size
        output_comm = self.mex_size
        self.policy_comm = ActorCritic(params, input_comm, output_comm).to(device)

        input_act = self.obs_size + self.n_agents*self.mex_size
        output_act = self.action_size
        self.policy_act = ActorCritic(params, input_act, output_act).to(device)

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy_comm.actor.parameters(), 'lr': self.lr_actor_comm},
                        {'params': self.policy_comm.critic.parameters(), 'lr': self.lr_critic_comm},
                        {'params': self.policy_act.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy_act.critic.parameters(), 'lr': self.lr_critic}])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayRate)

        self.train_returns = []
        self.return_episode = 0
        self.tmp_actions = []
        self.coop = []

        self.ent = True
        self.htarget = np.log(self.action_size)/2.

        #self.sign_lambda = sign_lambda
        #self.list_lambda = list_lambda

        self.saved_losses_comm = []
        self.saved_losses = []
        self.saved_sign_loss_list = []

        self.param_entropy = 0.1

        self.eps_norm = 0.0001
        self.comm_loss = True

        self.mutinfo_signaling = []
        self.mutinfo_listening = []
        self.mutinfo_signaling_old = []
        self.mutinfo_listening_old = []

        self.reset()

    def reset(self):
        self.comm_logprobs = []
        self.act_logprobs = []
        self.sign_loss_list = []
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
                #print("state_mex=", state_mex, "type=", type(state_mex))
                action, action_logprob, entropy = self.policy_act.act(state_mex, self.ent)

        elif (eval == False):
            state = torch.FloatTensor(state).to(device)
            state_mex = torch.cat((state, message)).to(device)
            action, action_logprob, entropy = self.policy_act.act(state_mex, self.ent)
            
            if (self.comm_loss == True):
                state_no_mex = torch.cat((state, torch.zeros_like(message))).to(device)
                dist_nomex = self.policy_act.get_distribution(state_no_mex).detach()
                dist_mex = self.policy_act.get_distribution(state_mex)
                #print("dist_no_mex=", dist_nomex)
                #print("dist_mex=", dist_mex)
                #print("diff=", dist_nomex - dist_mex)
                sign_loss = -torch.sum(torch.abs(dist_nomex - dist_mex))
                #print("sign_loss=", sign_loss)
                self.sign_loss_list.append(sign_loss)


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

        entropy = torch.FloatTensor([self.policy_comm.get_dist_entropy(state).detach() for state in self.buffer.states_c])
        hloss = (torch.full(entropy.size(), self.htarget) - entropy) * (torch.full(entropy.size(), self.htarget) - entropy)
        #print("hloss=", hloss.shape)

        for i in range(len(self.comm_logprobs)):
            #print(" -self.comm_logprobs[i] * rew_norm[i]=",  -self.comm_logprobs[i] * rew_norm[i])
            self.comm_logprobs[i] = -self.comm_logprobs[i] * rew_norm[i] + self.sign_lambda*hloss[i] + self.list_lambda*self.sign_loss_list[i]
            self.act_logprobs[i] = -self.act_logprobs[i] * rew_norm[i]

        #print("mean logits loss=", )
        #print("mean signloss=",torch.mean(torch.Tensor([i.detach() for i in self.sign_loss_list])))
        #print("mean entloss=",torch.mean(torch.Tensor([i.detach() for i in hloss])))
        self.saved_sign_loss_list.append(torch.mean(torch.Tensor([i.detach() for i in self.sign_loss_list])))
        self.saved_losses_comm.append(torch.mean(torch.Tensor([i.detach() for i in self.comm_logprobs])))
        self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in self.act_logprobs])))
        #print("self.comm_logp=", len(self.comm_logprobs))

        self.optimizer.zero_grad()
        tmp = [torch.ones(a.data.shape) for a in self.comm_logprobs]
        autograd.backward(self.comm_logprobs, tmp, retain_graph=True)

        tmp1 = [torch.ones(a.data.shape) for a in self.act_logprobs]
        autograd.backward(self.act_logprobs, tmp1, retain_graph=True)
        self.optimizer.step()

        #diminish learning rate
        self.scheduler.step()

        self.reset()