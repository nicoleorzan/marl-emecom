from src.environments import pgg_parallel_v1
from src.nets.ActorCritic import ActorCritic
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd

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

path = "/home/nicole/marl-emecom/src/experiments_pgg_v1/data/pgg_v1/3agents/1iters_[0.0, 0.0, 0.0]uncertainties/comm/"
#path = "/home/nicole/marl-emecom/src/experiments_pgg_v1/data/pgg_v1/5agents/1iters_[0.0, 0.0, 0.0, 0.0, 0]uncertainties/comm/"

with open(path+'params.json') as json_file:
    config = json.load(json_file)

config = dotdict(config)
print(config)

class MiniAgent():

    def __init__(self, policy_comm, policy_act):
        self.policy_comm = policy_comm
        self.policy_act = policy_act
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
    policy_act = ActorCritic(config.obs_dim + config.mex_space*config.n_agents, config.action_space)
    policy_act.load_state_dict(torch.load(path+"policy_act_agent_"+str(idx)))
    policy_act.eval()
    policy_comm = ActorCritic(config.obs_dim, config.action_space)
    policy_comm.load_state_dict(torch.load(path+"policy_comm_agent_"+str(idx)))
    policy_comm.eval()
    agents_dict['agent_'+str(idx)] = MiniAgent(policy_comm, policy_act) #{"policy_comm": policy_comm, "policy_act": policy_act}

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
            print("experiment=", experiment)
            for ep in range(eval_episodes):
                #print("\nEpisode=", ep)

                observations = parallel_env.reset()
                    
                for ag_idx, agent in agents_dict.items():
                    agent.tmp_return = 0
                    agent.tmp_actions = []

                done = False
                while not done:

                    messages = {}
                    for agent in parallel_env.agents:
                        mex, _ = agents_dict[agent].policy_comm.act(torch.FloatTensor(observations[agent]).to(device))
                        message = torch.Tensor([mex.item()]).long()
                        message = F.one_hot(message, num_classes=config.mex_space)[0]
                        messages[agent] = message
                    message = torch.stack([v for _, v in messages.items()]).view(-1)
                    actions = {agent: agents_dict[agent].policy_act.act(torch.cat((torch.FloatTensor(observations[agent]).to(device), message)))[0] for agent in parallel_env.agents}
                    
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

    df.to_csv(path+'data_comm_eval.csv')


# ===============================================================================
#   Calcolo le probab che ogni agente mandi messaggio 0 dato input (num of coins) OK
# ===============================================================================
def probab_mex0():
    print("Calcolo le probab che ogni agente mandi messaggio 0 dato input (num of coins)")
    fig, ax = plt.subplots(config.n_agents, figsize=(7,2*config.n_agents))
    fig.suptitle('Probability of Message 1 for every agent', fontsize=13)
    possible_inputs = np.linspace(0,1,110)
    eval_eps = 1
    
    for ag_idx in range(config.n_agents):
        #print("agent=", ag_idx)
        outputs = []
        for _, state in enumerate(possible_inputs):
            for ep in range(eval_eps):
                state = torch.FloatTensor([state]).to(device)
                val = agents_dict['agent_'+str(ag_idx)].policy_comm.get_distribution(state)
                #print("val=", val)
            outputs.append(val[0].detach().numpy())
        ax[ag_idx].plot(possible_inputs, outputs, label='agent '+str(ag_idx))
        ax[ag_idx].set_ylim(-0.05,1.05)
        ax[ag_idx].set_xlabel("Coins")
        ax[ag_idx].set_ylabel("P(mex) agent "+str(ag_idx))
        ax[ag_idx].grid()
    plt.savefig("prob_mex_0_"+str(config.n_agents)+"agents.png")


# ===============================================================================
#   Calcolo le probab che ogni agente cooperi dato input (num of coins)
# ===============================================================================
def probab_coop():
    print("Calcolo le probab che ogni agente cooperi dato input (num of coins)")
    fig, ax = plt.subplots(config.n_agents, figsize=(7,2*config.n_agents))
    fig.suptitle('Probability of Cooperation for every agent given input', fontsize=13)
    possible_inputs = np.linspace(0,1,100)

    parallel_env = pgg_parallel_v1.parallel_env(n_agents=config.n_agents, threshold=config.threshold, \
            num_iterations=config.num_game_iterations, uncertainties=config.uncertainties)

    eval_eps = 50
    for ag_idx in range(config.n_agents): # per ogni agent faccio il conto dipendente dall'input, dato che gli altri agenti fanno cose che non so
        coop_act = []
        for _, state in enumerate(possible_inputs):
            print("state=", state)
            coop_per_input = []
            for i in range(eval_eps): # loop molte volte x creare statistica
                observations = parallel_env.reset()
                observations["agent_"+str(ag_idx)] = np.array([state])
                #print("obs=", observations)
                messages = {}
                for agent in parallel_env.agents:
                    mex, _ = agents_dict[agent].policy_comm.act(torch.FloatTensor(observations[agent]).to(device))
                    message = torch.Tensor([mex.item()]).long()
                    message = F.one_hot(message, num_classes=config.mex_space)[0]
                    messages[agent] = message
                    #print(messages)
                message = torch.stack([v for _, v in messages.items()]).view(-1)
                actions = {agent: agents_dict[agent].policy_act.act(torch.cat((torch.FloatTensor(observations[agent]).to(device), message)))[0] for agent in parallel_env.agents}
                #print("az che ttacc",actions["agent_"+str(ag_idx)].long())
                coop_per_input.append(actions["agent_"+str(ag_idx)].long())
            #print("coop inp=",np.sum(coop_per_input)/eval_eps)
            coop_act.append(np.sum(coop_per_input)/eval_eps)
        print(coop_act, len(coop_act))

        message = torch.stack([v for _, v in messages.items()]).view(-1)
        ax[ag_idx].plot(possible_inputs, coop_act, label='agent '+str(ag_idx))
        ax[ag_idx].set_ylim(-0.05,1.05)
        ax[ag_idx].set_xlabel("Coins")
        ax[ag_idx].set_ylabel("P(coop) agent "+str(ag_idx))
        ax[ag_idx].grid()
    plt.savefig("prob_coop_comm_"+str(config.n_agents)+"agents.png")

# ===============================================================================
#   Vedo se il messaggio che un agente manda influenza la sua azione successiva
# ===============================================================================
def probab_actions():

    print("Vedo se il messaggio che un agente manda influenza la sua azione successiva")

    parallel_env = pgg_parallel_v1.parallel_env(n_agents=config.n_agents, threshold=config.threshold, \
            num_iterations=config.num_game_iterations, uncertainties=config.uncertainties)

    probs = np.zeros((config.n_agents, 2, 2)) # p(0|0), p(0|1), p(1|0), p(1|1)
    episodes = 1000
    for _ in range(episodes):
        observations = parallel_env.reset()

        messages = {}
        for agent in parallel_env.agents:
            mex, _ = agents_dict[agent].policy_comm.act(torch.FloatTensor(observations[agent]).to(device))
            message = torch.Tensor([mex.item()]).long()
            message = F.one_hot(message, num_classes=config.mex_space)[0]
            messages[agent] = message
        message = torch.stack([v for _, v in messages.items()]).view(-1)
        actions = {agent: agents_dict[agent].policy_act.act(torch.cat((torch.FloatTensor(observations[agent]).to(device), message)))[0] for agent in parallel_env.agents}

        for idx, ag in enumerate(parallel_env.agents):
            # messaggio e` 0
            if ( torch.all(messages[ag].eq(torch.Tensor([1,0])))):
                if( actions[ag].long() == 0 ):
                    probs[idx, 0, 0] += 1
                elif ( actions[ag].long() == 1  ):
                    probs[idx, 0, 1] += 1
            # messaggio e` 1
            elif ( torch.all(messages[ag].eq(torch.Tensor([0,1])))):
                if( actions[ag].long() == 0 ):
                    probs[idx, 1,0] += 1
                elif ( actions[ag].long() == 1 ):
                    probs[idx, 1,1] += 1

        observations, rewards, done, _ = parallel_env.step(actions)

    for ag_idx in range(config.n_agents):
        print("\nAgent=", ag_idx)
        print("Probab che prendo azioni 0 ed 1 dato messaggio 0:")
        print("P(a=0|m=0)=", probs[ag_idx, 0, 0]/(probs[ag_idx, 0, 0] + probs[ag_idx, 0, 1]))
        print("P(a=1|m=0)=", probs[ag_idx, 0, 1]/(probs[ag_idx, 0, 0] + probs[ag_idx, 0, 1]))

        print("Probab che prendo azioni 0 ed 1 dato messaggio 1:")
        print("P(a=0|m=1)=", probs[ag_idx, 1, 0]/(probs[ag_idx, 1, 0] + probs[ag_idx, 1, 1]))
        print("P(a=1|m=1)=", probs[ag_idx, 1, 1]/(probs[ag_idx, 1, 0] + probs[ag_idx, 1, 1]))

# ===============================================================================
#   Vedo se il messaggio che mandano gli altri influenza la mia azione successiva
# ===============================================================================
def prob_act_mex_others():

    print("Vedo se il messaggio che mandano gli altri influenza la mia azione successiva")

    parallel_env = pgg_parallel_v1.parallel_env(n_agents=config.n_agents, threshold=config.threshold, \
            num_iterations=config.num_game_iterations, uncertainties=config.uncertainties)

    probs = np.zeros((config.n_agents, config.action_space, config.action_space*config.n_agents)) 
    # Per ogni agente mi calcolo:
    # p(a=0|0,1), p(a=0|1,0), p(a=0|0,0), p(a=0|1,1)
    # p(a=1|0,1), p(a=1|1,0), p(a=1|0,0), p(a=1|1,1)
    possible_messages1 = [torch.Tensor([1.,0.]), torch.Tensor([0.,1.]), torch.Tensor([1.,0.]), torch.Tensor([0.,1.])]
    possible_messages2 = [torch.Tensor([1.,0.]), torch.Tensor([0.,1.]), torch.Tensor([0.,1.]), torch.Tensor([1.,0.])]
    for mex2, mex3 in zip(possible_messages1, possible_messages2):
        print("mex2=", mex2, "mex3=", mex3)
    episodes = 1000
    # and I want to do this for every possible input
    possible_inputs = np.linspace(0,1,100)

    for ag_idx in range(config.n_agents): # per ogni agent faccio il conto dipendente dall'input, dato che gli altri agenti fanno cose che non so
        for _, state in enumerate(possible_inputs):
            print("state=", state)
            observations = parallel_env.reset()
            observations["agent_"+str(ag_idx)] = np.array([state])

            messages = {}
            for mex2, mex3 in zip(possible_messages1, possible_messages2):
                print("mex2=", mex2, "mex3=", mex3)
                messages = {}
                for idx, agent in enumerate(parallel_env.agents):
                    print("agent_in=", idx)
                    #print("idx+1%config.n_agents=", idx+1%config.n_agents)
                    if (idx == ag_idx):
                        print("idx1=", idx)
                        mex, _ = agents_dict[agent].policy_comm.act(torch.FloatTensor(observations[agent]).to(device))
                        message = torch.Tensor([mex.item()]).long()
                        message = F.one_hot(message, num_classes=config.mex_space)[0]
                        messages[agent] = message
                    if (idx == ag_idx+1%config.n_agents):
                        print("idx2=", idx)
                        messages[agent] = mex2
                    if (idx == ag_idx+2%config.n_agents):
                        print("idx3=", idx)
                        messages[agent] = mex3
                messages = torch.stack([v for _, v in messages.items()]).view(-1)
                print("messages=", messages)
                actions = {agent: agents_dict[agent].policy_act.act(torch.cat((torch.FloatTensor(observations[agent]).to(device), messages)))[0] for agent in parallel_env.agents}
                print("actions=", actions[agent])

                probs[idx, actions[agent], ]                
            break
        break
                    




if __name__ == "__main__":
    #prob_act_mex_others()
    probab_mex0()
    probab_actions()
    #probab_coop()
    #eval_episodes()
