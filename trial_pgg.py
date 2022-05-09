from src.environments import pgg_v0
import supersuit as ss
from PPO import PPO
import numpy as np
import matplotlib.pyplot as plt

eval_eps = 1000
max_training_timesteps = 10000  #int(1e3)   # break training loop if timeteps > max_training_timesteps
update_timestep = 40          # update policy every n timesteps

n_agents = 2
n_total_coins = 6

action_space = 2
input_dim_agent = 3         # we observe coins we have, num of agents, and multiplier factor with uncertainty
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor
c1 = 0.5
c2 = -0.01
lr_actor = 0.001       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

max_ep_len = 1                    # max timesteps in one episode
num_blocks = 10                   # number of blocks for moving average

print_freq = 100     # print avg reward in the interval (in num timesteps)

env = pgg_v0.env(n_agents = n_agents, n_total_coins = n_total_coins)

agent0_PPO = PPO(input_dim_agent, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)
agent1_PPO = PPO(input_dim_agent, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)

#untrained agents for comparison
un_agent0_PPO = PPO(input_dim_agent, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)
un_agent1_PPO = PPO(input_dim_agent, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)

agents_dict = {"agent_0": agent0_PPO, "agent_1": agent1_PPO}
un_agents_dict = {"agent_0": un_agent0_PPO, "agent_1": un_agent1_PPO}
agent_to_idx = {"agent_0": 0, "agent_1": 1}

def evaluation(agents_dict, episodes):

    agent0r = np.zeros((episodes))
    agent1r = np.zeros((episodes))
    for e in range(episodes):
        if (e%100 == 0):
            print("Episode:", e)
        agent0r[e], agent1r[e] = evaluate_episode(agents_dict)

    #print(agent0r, agent1r)
    return [agent0r, agent1r]

def evaluate_episode(agents_dict):
    env = pgg_v0.env(n_agents = n_agents, n_total_coins = n_total_coins)
    env.reset()
    #agent0r = np.zeros(1); agent1r = np.zeros(1)
    i = 0
    for agent in env.agent_iter():
        obs, reward, done, _ = env.last()
        acting_agent = agents_dict[agent]
        act = acting_agent.select_action(obs) if not done else None
        env.step(act)
        if (agent == 'agent_0'):
            agent0r = reward
        else:
            agent1r = reward
        i += 1
    env.close()
    return agent0r, agent1r

def plot_hist_returns(rews_before, rews_after):

    agent0rb, agent1rb = rews_before
    agent0ra, agent1ra = rews_after

    n_bins = 40
    fig, ax = plt.subplots(n_agents, 2, figsize=(20,8))
    fig.suptitle("Distribution of Returns", fontsize=25)
    ax[0,0].hist(agent0rb, bins=n_bins, range=[-30., 60.])
    ax[0,1].hist(agent1rb, bins=n_bins, range=[-30., 60.])
    ax[1,0].hist(agent0ra, bins=n_bins, range=[-30., 60.])
    ax[1,1].hist(agent1ra, bins=n_bins, range=[-30., 60.])
    print("Saving histogram..")
    plt.savefig("images/pgg/hist_rewards_pgg.png")

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


print("\nEVALUATION BEFORE LEARNING")
rews_before = evaluation(un_agents_dict, eval_eps)

#### TRAINING LOOP
print("\n===>Training")
time_step = 0
i_episode = 0

# printing and logging variables
print_running_reward = np.zeros(n_agents)
print_running_episodes = np.zeros(n_agents)


while time_step <= max_training_timesteps:

    env.reset()

    current_ep_reward = np.zeros(n_agents)
    i_internal_loop = 0
    agent0_PPO.tmp_return = 0; agent1_PPO.tmp_return = 0

    for id_agent in env.agent_iter():
        #print("\nTime=", time_step)
        #print(agent_to_idx[id_agent], id_agent)
        idx = agent_to_idx[id_agent]
        acting_agent = agents_dict[id_agent]
        
        obs, rew, done, info = env.last()
        #print(obs, rew, done, info)
        act = acting_agent.select_action(obs) if not done else None
        #print("act=", act)
        #print("env step")
        env.step(act)

        #print("step done")
        if (i_internal_loop > 1):
            acting_agent.buffer.rewards.append(rew)
            acting_agent.buffer.is_terminals.append(done)
            acting_agent.tmp_return += rew

        time_step += 1

        if rew != None:
            current_ep_reward[idx] += rew

        #print("==>acting_agent.buffer.__print__()=")
        #acting_agent.buffer.__print__()   

        # break; if the episode is over
        if done:
            agent0_PPO.train_returns.append(agent0_PPO.tmp_return)
            agent1_PPO.train_returns.append(agent1_PPO.tmp_return)
        if (done and idx==n_agents-1):  
            #print("======>exiting") 
            break

        i_internal_loop += 1

    # update PPO agent
    if time_step % update_timestep == 0:
        acting_agent.update()
    
    if time_step % print_freq == 0:
        #print("time_step % print_freq=",time_step % print_freq)
        print_avg_reward = np.zeros(n_agents)
        for k in range(n_agents):
            #print(print_running_reward[k],print_running_episodes[k])
            print_avg_reward[k] = print_running_reward[k] / print_running_episodes[k]
            print_avg_reward[k] = round(print_avg_reward[k], 2)

        print("Episode : {} \t\t Timestep : {} \t\t ".format(i_episode, time_step))
        print("Average Reward : agent0=",print_avg_reward[0],", angent1=",print_avg_reward[1])

        for i in range(n_agents):
            print_running_reward[i] = 0
            print_running_episodes[i] = 0
    
    for i in range(n_agents):
        print_running_reward[i] += current_ep_reward[i]
        print_running_episodes[i] += 1

    i_episode += 1


### PLOT TRAIN RETURNS
mov_avg_agent0 = moving_average(agent0_PPO.train_returns, num_blocks)
mov_avg_agent1 = moving_average(agent1_PPO.train_returns, num_blocks)

fig, (ax1, ax2) = plt.subplots(n_agents)
fig.suptitle("Train Returns")
ax1.plot(np.linspace(0, len(agent0_PPO.train_returns), len(agent0_PPO.train_returns)), agent0_PPO.train_returns)
ax1.plot(np.linspace(0, len(agent0_PPO.train_returns), len(mov_avg_agent0)), mov_avg_agent0)
ax2.plot(np.linspace(0, len(agent1_PPO.train_returns), len(agent1_PPO.train_returns)), agent1_PPO.train_returns)
ax2.plot(np.linspace(0, len(agent1_PPO.train_returns), len(mov_avg_agent1)), mov_avg_agent1)
plt.savefig("images/pgg/train_returns_pgg.png")

### EVALUATION
print("\n\nEVALUATION AFTER LEARNING")

rews_after = evaluation(agents_dict, eval_eps)
print("average rews before", np.average(rews_before[0]), np.average(rews_before[1]))
print("average rews ater", np.average(rews_after[0]), np.average(rews_after[1]))

plot_hist_returns(rews_before, rews_after)