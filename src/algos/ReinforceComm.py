
from src.algos.buffer import RolloutBufferComm
from src.nets.ActorCritic import ActorCritic
import torch
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
        self.coop = []
        self.reset()

    def reset(self):
        self.comm_logprobs = []
        self.act_logprobs = []
        self.rewards = []

    def reset_episode(self):
        self.return_episode = 0
        self.tmp_actions = []

    def select_message(self, state):

        state = torch.FloatTensor(state).to(device)
        message, message_logprob = self.policy_comm.act(state)

        self.buffer.messages.append(message)

        self.comm_logprobs.append(message_logprob)

        message = torch.Tensor([message.item()]).long()
        message = F.one_hot(message, num_classes=self.mex_size)[0]
        return message

    def random_messages(self, state):

        state = torch.FloatTensor(state).to(device)
        message = torch.randint(0, self.mex_size, (self.mex_size-1,))[0]

        self.buffer.messages.append(message)

        self.comm_logprobs.append(torch.tensor(0.0001))

        message = torch.Tensor([message.item()]).long()
        message = F.one_hot(message, num_classes=self.mex_size)[0]

        return message

    def select_action(self, state, message):
    
        state = torch.FloatTensor(state).to(device)
        state_mex = torch.cat((state, message))
        action, action_logprob = self.policy_act.act(state_mex)

        self.buffer.actions.append(action)

        self.act_logprobs.append(action_logprob)

        return action

    def update(self):
    
        for i in range(len(self.comm_logprobs)):
            self.comm_logprobs[i] = -self.comm_logprobs[i] * self.rewards[i]
            self.act_logprobs[i] = -self.act_logprobs[i] * self.rewards[i]

        self.optimizer.zero_grad()
        tmp = [torch.ones(a.data.shape) for a in self.comm_logprobs]
        autograd.backward(self.comm_logprobs, tmp, retain_graph=True)
        autograd.backward(self.act_logprobs, tmp, retain_graph=True)
        self.optimizer.step()

        #diminish learning rate
        self.scheduler.step()

        self.reset()