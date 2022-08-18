from src.environments import pgg_parallel_v1
from src.nets.ActorCritic import ActorCritic
from src.nets.ActorCriticRNN import ActorCriticRNN
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd


device = torch.device('cpu')
if (torch.cuda.is_available()): 
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

path = "/home/nicole/marl-emecom/src/experiments_pgg_v1/data/pgg_v1/2agents/1iters_[0.0, 0.0]uncertainties/"

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
    if config.recurrent == True:
        policy_act = ActorCriticRNN(config)
    else:
        policy_act = ActorCritic(config)
    policy_act.load_state_dict(torch.load(path+"model_agent_"+str(idx)))
    policy_act.eval()
    agents_dict['agent_'+str(idx)] = MiniAgent(policy_act)


def eval_episodes():
    parallel_env = pgg_parallel_v1.parallel_env(n_agents=config.n_agents, threshold=config.threshold, \
            num_iterations=config.num_game_iterations, uncertainties=config.uncertainties)

    df = pd.DataFrame(columns=['experiment', 'episode'] + \
            ["ret_ag"+str(i) for i in range(config.n_agents)] + \
            ["coop_ag"+str(i) for i in range(config.n_agents)])

    n_experiments = 100
    eval_episodes = 5000
    with torch.no_grad():

        for experiment in range(n_experiments):
            #print("experiment=", experiment)
            for ep in range(eval_episodes):
                #print("\nEpisode=", ep)

                observations = parallel_env.reset()
                    
                for ag_idx, agent in agents_dict.items():
                    agent.tmp_return = 0
                    agent.tmp_actions = []

                done = False
                while not done:

                    actions = {agent: agents_dict[agent].policy.act(torch.FloatTensor([observations[agent]]).to(device), True)[0] for agent in parallel_env.agents}
                    #print("actions=", actions)
                    observations, rewards, done, _ = parallel_env.step(actions)
                    #print("observations, rewards, done", observations, rewards, done)

                    for ag_idx, agent in agents_dict.items():
                        agent.tmp_return += rewards[ag_idx]
                        if (actions[ag_idx] is not None):
                            agent.tmp_actions.append(actions[ag_idx])
                        if done:
                            agent.returns.append(agent.tmp_return)
                            agent.coop.append(np.mean(agent.tmp_actions))

                    if done:
                        break

                if (ep%config.save_interval == 0):
                    df_ret = {"ret_ag"+str(i): agents_dict["agent_"+str(i)].tmp_return for i in range(config.n_agents)}
                    df_coop = {"coop_ag"+str(i): np.mean(agents_dict["agent_"+str(i)].tmp_actions) for i in range(config.n_agents)}
                    df_dict = {**{'experiment': experiment, 'episode': ep}, **df_ret, **df_coop}
                    df = pd.concat([df, pd.DataFrame.from_records([df_dict])])

    df.to_csv(path+'data_no_comm_eval.csv')

# ===============================================================================
#   Calcolo le probab che ogni agente cooperi dato input (num of coins)
# ===============================================================================
def probab_coop():
    print("Calcolo le probab che ogni agente cooperi dato input (num of coins)")
    fig, ax = plt.subplots(config.n_agents, figsize=(7,6))
    fig.suptitle('Probability of Cooperation for every agent', fontsize=13)
    possible_inputs = np.linspace(0,1.5,100)
    for ag_idx in range(config.n_agents):
        outputs = []
        print("agent=", ag_idx)
        for _, state in enumerate(possible_inputs):
            state = torch.FloatTensor([state]).to(device)
            out = agents_dict['agent_'+str(ag_idx)].policy.get_distribution(state)
            outputs.append(out.detach().numpy()[1])
        ax[ag_idx].plot(possible_inputs, outputs, label='agent '+str(ag_idx))
        ax[ag_idx].set_ylim(-0.05,1.05)
        ax[ag_idx].set_xlabel("Coins")
        ax[ag_idx].set_ylabel("P(coop) agent "+str(ag_idx))
        ax[ag_idx].grid()
    plt.savefig("prob_coop_"+str(config.n_agents)+"agents.png")


if __name__ == "__main__":
    probab_coop()
    #eval_episodes()

