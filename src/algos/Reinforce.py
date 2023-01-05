
import torch
import torch.autograd as autograd
from sklearn.mixture import GaussianMixture as GMM
import torch.nn.functional as F

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class Reinforce():

    def __init__(self, model, optimizer, params, idx):

        for key, val in params.items():  setattr(self, key, val)
    
        self.policy = model.to(device)
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decayRate)
        
        self.train_returns = []
        self.train_returns_norm = []
        self.return_episode = 0
        self.return_episode_norm= 0
        self.tmp_actions = []
        self.coop = []
        self.saved_losses = []

        self.reset()

        self.eps_norm = 0.0001

        self.idx = idx
        self.gmm_ = self.policy.gmm_
        #print("agent idx=", self.idx)
        #print("gmm is=", self.gmm_)
        
    def reset(self):
        self.logprobs = []
        self.rewards = []

    def reset_episode(self):
        self.return_episode_old = self.return_episode
        self.return_episode_old_norm = self.return_episode_norm
        self.return_episode = 0
        self.return_episode_norm = 0
        self.tmp_actions_old = self.tmp_actions
        self.tmp_actions = []

    def select_action(self, state, eval=False):

        state = torch.FloatTensor(state).to(device)
        #print("idx=", self.idx, "state before=", state)
        if (self.gmm_):
            state = self.get_gmm_state(state, eval)
            #print("state after=", state)

        if (eval == True):
            with torch.no_grad():
                action, action_logprob = self.policy.act(state)
        elif (eval == False):
            action, action_logprob = self.policy.act(state)

            self.logprobs.append(action_logprob)

        return action

    def get_distribution(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if (self.gmm_):
                state = self.get_gmm_state(state, eval=True)
            out = self.policy.get_distribution(state)

            return out

    def update(self):

        rewards =  self.rewards
        rew_norm = [(i - min(rewards))/(max(rewards) - min(rewards) + self.eps_norm) for i in rewards]

        for i in range(len(self.logprobs)):
            self.logprobs[i] = -self.logprobs[i] * rew_norm[i]

        self.saved_losses.append(torch.mean(torch.Tensor([i.detach() for i in self.logprobs])))
        self.optimizer.zero_grad()
        tmp = [torch.ones(a.data.shape) for a in self.logprobs]
        autograd.backward(self.logprobs, tmp, retain_graph=True)
        self.optimizer.step()

        self.scheduler.step()
        print(self.scheduler.get_lr())

        self.reset()

    def get_gmm_state(self, state, eval=False):

        if hasattr(self, 'mf_history'):
            if (len(self.mf_history) < 50000 and eval == False):
                self.mf_history = torch.cat((self.mf_history, state[1].reshape(1)), 0)
        else:
            self.mf_history = state[1].reshape(1)

        if (len(self.mf_history) >= len(self.mult_fact)):
            self.gmm = GMM(n_components = len(self.mult_fact), max_iter=1000, random_state=0, covariance_type = 'full')
            input_ = self.mf_history.reshape(-1, 1)
            self.gmm.fit(input_)
            p = torch.Tensor(self.gmm.predict(state[1].reshape(1).reshape(-1, 1))).long()
            state_in = F.one_hot(p, num_classes=len(self.mult_fact))[0].to(torch.float32)
        else: 
            state_in = torch.zeros(len(self.mult_fact)).to(device)

        return state_in