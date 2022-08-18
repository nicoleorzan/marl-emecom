from src.environments import pgg_parallel_v1
from src.nets.ActorCriticRNN import ActorCriticRNN
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F


device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

path = "/home/nicole/marl-emecom/src/experiments_pgg_v1/data/pgg_v1/2agents/3iters_[0.0, 0.0]uncertainties/"

with open(path+'params.json') as json_file:
    config = json.load(json_file)

config = dotdict(config)

class MiniAgent():

    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        self.tmp_return = 0
        self.tmp_reward = 0
        self.tmp_rewards = []
        self.tmp_actions = []
        self.returns = []
        self.coop = []
    
agents_dict = {}
for idx in range(config.n_agents):
    policy_act = ActorCriticRNN(config)
    policy_act.load_state_dict(torch.load(path+"model_memory_agent_"+str(idx)))
    policy_act.eval()
    agents_dict['agent_'+str(idx)] = MiniAgent(policy_act)

parallel_env = pgg_parallel_v1.parallel_env(n_agents=config.n_agents, threshold=config.threshold, \
        num_iterations=config.num_game_iterations, uncertainties=config.uncertainties)

# ===============================================================================
#   Calcolo le probab che ogni agente cooperi dato input at the first (num of coins + zero actions)
# ===============================================================================
def probab_coop_one_iter():
    print("Calcolo le probab che ogni agente cooperi dato input (num of coins)")

    fig, ax = plt.subplots(config.n_agents, figsize=(7,6))
    fig.suptitle('Probability of Cooperation for every agent', fontsize=13)
    possible_inputs = np.linspace(0,1.5,100)
    dist0 = []; dist1 = []; dist2 = []

    for _, state in enumerate(possible_inputs):
        state = torch.FloatTensor([state]).to(device)
        
        [agent.reset() for _, agent in agents_dict.items()]

        actions_cat = torch.zeros((config.n_agents*config.action_size), dtype=torch.int64)

        distributions = {agent: agents_dict[agent].policy.get_distribution(torch.cat((state, actions_cat))).detach().numpy()[1] for agent in parallel_env.agents}
        dist0.append(distributions[ "agent_0"])
        dist1.append(distributions[ "agent_1"])
        dist2.append(distributions[ "agent_2"])

    ax[0].plot(possible_inputs, dist0, label='agent 0')
    ax[1].plot(possible_inputs, dist1, label='agent 1')
    ax[2].plot(possible_inputs, dist2, label='agent 2')
           
    for ag_idx in range(config.n_agents):
        ax[ag_idx].set_ylim(-0.05,1.05)
        ax[ag_idx].set_xlabel("Coins")
        ax[ag_idx].set_ylabel("P(coop) agent "+str(ag_idx))
        ax[ag_idx].grid()

    plt.savefig("prob_coop_"+str(config.n_agents)+"agents_memory.png")

def dependence(agent_in):
    prob = np.zeros((config.action_size, config.action_size*(config.n_agents-1)))
    possible_actions_input = []
    possible_inputs = np.linspace(0,1.5,100)
    actions_00 = torch.Tensor([1, 0, 1, 0]) 
    actions_10 = torch.Tensor([0, 1, 1, 0]) 
    actions_01 = torch.Tensor([1, 0, 0, 1]) 
    actions_11 = torch.Tensor([0, 1, 0, 1]) 
    acts = [actions_00, actions_10, actions_01, actions_11]
    for act in acts:
        # calcolo p(0|0,0,0) e p(1|0,0,0) mediato sui coins
        vals = []
        for _, state in enumerate(possible_inputs):
            state = torch.FloatTensor([state]).to(device)
            [agents_dict[ag].policy.observe(torch.FloatTensor(torch.cat((torch.Tensor(state), act))).to(device)) for ag in parallel_env.agents]
            actions = {ag: agents_dict[ag].policy.act() for ag in parallel_env.agents}
            vals.append(actions[agent_in])
        plt.plot(vals)
        plt.show()

    # compute, for single agent
    # p(0|0,0) p(0|0,1) p(0|1,0) p(0|1,1)
    # p(1|0,0) p(1|0,1) p(1|1,0) p(1|1,1)

def strategy_dependence():
    for agent in parallel_env.agents:
        dependence(agent)
        break
    """eval_eps = 1
    for _ in range(eval_eps):
        observations = parallel_env.reset()
            
        [agent.reset() for _, agent in agents_dict.items()]

        done = False
        actions_cat = torch.zeros((config.n_agents*config.action_size), dtype=torch.int64)
        i = 0
        actions_agent0 = []
        actions_agent1 = []
        actions_agent2 = []
        while not done:
            print("\n",observations)

            [agents_dict[agent].policy.observe(torch.FloatTensor(torch.cat((torch.Tensor(observations[agent]), actions_cat))).to(device)) for agent in parallel_env.agents]
            actions = {agent: agents_dict[agent].policy.act() for agent in parallel_env.agents}
            observations, rewards, done, _ = parallel_env.step(actions)
            print(actions)
            print(rewards)
            actions_cat = torch.stack([F.one_hot(torch.Tensor([v.item()]).long(), num_classes=config.action_size)[0] for _, v in actions.items()]).view(-1)
            i += 1
             

    """

if __name__ == "__main__":
    #probab_coop_one_iter()
    strategy_dependence()
