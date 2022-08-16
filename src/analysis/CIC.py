from src.environments import pgg_parallel_v1
from src.nets.ActorCritic import ActorCriticDiscrete
import torch.nn.functional as F
import math
import numpy as np
import torch
import json

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

with open(path+'params.json') as json_file:
    config = json.load(json_file)

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
    policy_act = ActorCriticDiscrete(config.obs_dim + config.mex_space*config.n_agents, config.action_space)
    policy_act.load_state_dict(torch.load(path+"policy_act_agent_"+str(idx)))
    policy_act.eval()
    policy_comm = ActorCriticDiscrete(config.obs_dim, config.action_space)
    policy_comm.load_state_dict(torch.load(path+"policy_comm_agent_"+str(idx)))
    policy_comm.eval()
    agents_dict['agent_'+str(idx)] = MiniAgent(policy_comm, policy_act)

config = dotdict(config)

def calc_model_cic(model_file_ag1, model_file_ag2, num_games=1000):
    # Given trained model files in model_file_ag1 and model_file_ag2 (.txt files saved with torch.save), calculates
    # the one-step CIC, averaged over num_games training games
    # args are used to specify MCG game, and structure of the agent

    # Instantiate MCG and agents
    parallel_env = pgg_parallel_v1.parallel_env(n_agents=config.n_agents, threshold=config.threshold, \
        num_iterations=config.num_game_iterations, uncertainties=config.uncertainties)

    # Iterate over games
    cics = [[], []]
    for i in range(num_games):
        # Get a new game (which is random even if args.game = fixed)
        observations = parallel_env.reset()

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

        #env.payoffs_a = [3 * np.random.randn(env.n_acts, env.n_acts)]
        #env.payoffs_b = [3 * np.random.randn(env.n_acts, env.n_acts)]
        #ob_c = env.reset()

        agents = [agents_dict['agent_0'], agents_dict['agent_1']]
        # Calculate p(a | do(c)) for both agents and messages c
        p_a_given_do_c = get_p_a_given_do_c(agents, parallel_env, n_comm=config.mex_dim)

        # For each agent, calculate the one-step CIC
        for idx_ag, agent in enumerate(parallel_env.agents):
            # Calcualte p(c) of other agent (1-ag) by doing a forward pass through network
            logits_c, logits_a, v = agents[1 - idx].forward(torch.Tensor(observations[agent]))
            probs_c = F.softmax(logits_c, dim=0).data.numpy()
            cic = calc_cic(p_a_given_do_c[agent], probs_c, config.mex_dim, config.action_dim)
            cics[agent].append(cic)

    return cics

def calc_cic(p_a_given_do_c, p_c, n_comm, n_acts):
    # Calculate the one-step causal influence of communication, i.e. the mutual information using p(a | do(c))
    p_ac = p_a_given_do_c * np.expand_dims(p_c, axis=1)  # calculate joint probability p(a, c)
    p_ac /= np.sum(p_ac)  # re-normalize
    p_a = np.mean(p_ac, axis=0)  # compute p(a) by marginalizing over c

    # Calculate mutual information
    cic = 0
    for c in range(n_comm):
        for a in range(n_acts):
            if p_ac[c][a] > 0:
                cic += p_ac[c][a] * math.log(p_ac[c][a] / (p_c[c] * p_a[a]))
    return cic

def get_p_a_given_do_c(agents, parallel_env, config, self=False):
    # Calculates p(a | do(c)) for both agents, i.e. the probability distribution over agent 1's actions given that
    # we intervene at agent 2 to send message c (and vice-versa)
    # If self = True, calculates p(a | do(c)) if we intervene at agent 1 to send message c, i.e. the effect of
    # agent 1's message on its own action (and similarly for agent 2)

    # Cache payoff matrices to ensure they are kept consistent
    #payoff_a = env.payoff_mat_a
    #payoff_b = env.payoff_mat_b
    p_a_given_do_c = [np.zeros((config.mex_dim, config.action_dim)), np.zeros((config.mex_dim, config.action_dim))]

    # For all agents
    for ag in range(3):
        # Iterated over this agent's possible messages
        for i in range(config.mex_dim): # n_comm = 2
            observations = parallel_env.reset()  # get rid of any existing messages in the observation

            # agent in control intervenes in environment with message i
            messages = []
            for agent in parallel_env.agents:
                mex, _ = agents_dict[agent].policy_comm.act(torch.FloatTensor(observations[agent]).to(device))
                message = torch.Tensor([mex.item()]).long()
                message = F.one_hot(message, num_classes=config.mex_space)[0]
                messages[agent] = message

            env.payoff_mat_a = payoff_a  # restore payoffs undone by .reset()
            env.payoff_mat_b = payoff_b
            ob_c, _ = env.step_c_single(i, ag)  # intervene in environment with message i
            if self:
                # Calculate p(a|do(c)) of same agent
                logits_c, logits_a, v = agents[ag].forward(torch.Tensor(ob_c[ag]))
            else:
                # Calculate p(a|do(c)) of other agent
                logits_c, logits_a, v = agents[1 - ag].forward(torch.Tensor(ob_c[1 - ag]))

            # Convert logits to probability distribution
            probs_a = F.softmax(logits_a, dim=0)
            p_a_given_do_c[ag][i, :] = probs_a.data.numpy()

    return p_a_given_do_c