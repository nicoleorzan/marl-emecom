from pettingzoo.mpe import simple_adversary_v2
from PPO import PPO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cycles = 30

n_agents = 3
action_space = 5
input_dim_agent = 10
input_dim_adversary = 8
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor
c1 = 0.5
c2 = -0.01

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

max_ep_len = 500                    # max timesteps in one episode
max_training_timesteps = int(5e5)   # break training loop if timeteps > max_training_timesteps
num_blocks = 10                   # number of blocks for moving average
eval_eps = 1000

update_timestep = max_ep_len * 4      # update policy every n timesteps
print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)


agent0_PPO = PPO(input_dim_agent, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)
agent1_PPO = PPO(input_dim_agent, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)
adversary0_PPO = PPO(input_dim_adversary, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)

un_agent0_PPO = PPO(input_dim_agent, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)
un_agent1_PPO = PPO(input_dim_agent, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)
un_adversary0_PPO = PPO(input_dim_adversary, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)

env = simple_adversary_v2.env(N=2, max_cycles=cycles, continuous_actions=False)

agents_dict = {"agent_0": agent0_PPO, "agent_1": agent1_PPO, "adversary_0": adversary0_PPO}
un_agents_dict = {"agent_0": un_agent0_PPO, "agent_1": un_agent1_PPO, "adversary_0": un_adversary0_PPO}
agent_to_idx = {"adversary_0": 0, "agent_0": 1, "agent_1":2}

time_step = 0
i_episode = 0

# printing and logging variables
print_running_reward = np.zeros(3)
print_running_episodes = np.zeros(3)

def evaluation(agents_dict, episodes):

    agent0r = np.zeros((episodes, cycles))
    agent1r = np.zeros((episodes, cycles))
    adv0r = np.zeros((episodes, cycles))
    for e in range(episodes):
        if (e%100 == 0):
            print("Episode:", e)
        agent0r[e], agent1r[e], adv0r[e] = evaluate_episode(agents_dict)

    return [agent0r, agent1r, adv0r]


def evaluate_episode(agents_dict):
    env = simple_adversary_v2.env(max_cycles=cycles, continuous_actions=False)
    env.reset()
    agent0r = np.zeros(cycles); agent1r = np.zeros(cycles); adv0r = np.zeros(cycles)
    i = 0
    for agent in env.agent_iter():
        obs, reward, done, _ = env.last()
        acting_agent = agents_dict[agent]
        act = acting_agent.select_action(obs) if not done else None
        env.step(act)
        if (i > 2):
            if (agent == 'adversary_0'):
                adv0r[int(i/n_agents)-2] = reward
            elif (agent == 'agent_0'):
                agent0r[int(i/n_agents)-n_agents] = reward
            elif (agent == 'agent_1'):
                agent1r[int(i/n_agents)-4] = reward
        i += 1
    env.close()
    return agent0r, agent1r, adv0r


def plot_hist_returns(rews_before, rews_after):

    agent0r, agent1r, adv0r = rews_before
    returns_ag0b = np.sum(agent0r, axis=1)
    returns_ag1b = np.sum(agent1r, axis=1)
    returns_adv0b = np.sum(adv0r, axis=1)
    

    agent0r, agent1r, adv0r = rews_after
    returns_ag0a = np.sum(agent0r, axis=1)
    returns_ag1a = np.sum(agent1r, axis=1)
    returns_adv0a = np.sum(adv0r, axis=1)

    n_bins = 40
    fig, ax = plt.subplots(n_agents, 2, figsize=(20,8))
    fig.suptitle("Distribution of Returns", fontsize=25)
    ax[0,0].hist(returns_ag0b, bins=n_bins, range=[-30., 60.])
    ax[0,1].hist(returns_ag0a, bins=n_bins, range=[-30., 60.])
    ax[1,0].hist(returns_ag1b, bins=n_bins, range=[-30., 60.])
    ax[1,1].hist(returns_ag1a, bins=n_bins, range=[-30., 60.])
    ax[2,0].hist(returns_adv0b, bins=n_bins, range=[-120., 0.])
    ax[2,1].hist(returns_adv0a, bins=n_bins, range=[-120., 0.])
    print("Saving histogram..")
    plt.savefig("images/simple_adv/hist_rewards_simple_adv.png")
    print("done")


def plotting_rews(rews_b, rews_a): # rews is a LIST OF LISTS for every agent of the num of episodes to plot

    fig, ax = plt.subplots(n_agents, 2, figsize=(20,8))
    fig.suptitle("Rewards of agent in time for one episode", fontsize=25)

    meanrews_b = np.zeros((n_agents, len(rews_b[0][0])))
    sdrews_b = np.zeros((n_agents, len(rews_b[0][0])))
    meanrews_a = np.zeros((n_agents, len(rews_a[0][0])))
    sdrews_a = np.zeros((n_agents, len(rews_a[0][0])))

    for i in range(n_agents):
        meanrews_b[i] = [np.mean(k) for k in zip(*rews_b[i])]
        sdrews_b[i] = [np.std(k) for k in zip(*rews_b[i])]
        meanrews_a[i] = [np.mean(k) for k in zip(*rews_a[i])]
        sdrews_a[i] = [np.std(k) for k in zip(*rews_a[i])]

        ax[i,0].errorbar(np.linspace(0, len(meanrews_b[i]), len(meanrews_b[i])), meanrews_b[i], sdrews_b[i], \
            linewidth=3, label="before_learning, agent"+ str(i))
        ax[i,1].errorbar(np.linspace(0, len(meanrews_a[i]), len(meanrews_a[i])), meanrews_a[i], sdrews_a[i], \
            linewidth=3, label="after_learning, agent"+ str(i))
        ax[i,0].tick_params(axis='both', which='major', labelsize=18)
        ax[i,1].tick_params(axis='both', which='major', labelsize=18)
        ax[i,0].legend(fontsize=18)
        ax[i,1].legend(fontsize=18)

    ax[2,0].set_xlabel('Time', fontsize=18)
    ax[2,1].set_xlabel('Time', fontsize=18)
    ax[0,0].set_ylabel('Reward', fontsize=18)

    for i,j in zip(range(0,2), range(0,2)):
        ax[i,j].set_ylim(-2,1)
    plt.savefig("images/simple_adv/rewards_simple_adversary_v2.png")

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


print("\nEVALUATION BEFORE LEARNING")
rews_before = evaluation(un_agents_dict, eval_eps)

#### TRAINING LOOP

while time_step <= max_training_timesteps:

    env.reset()
    obs, rew, done, info = env.last()

    current_ep_reward = np.zeros(3)
    i_internal_loop = 0
    agent0_PPO.tmp_return = 0; agent1_PPO.tmp_return = 0; adversary0_PPO.tmp_return = 0

    for id_agent in env.agent_iter(max_ep_len):
        #print("id_agent=", id_agent)
        #print(agent_to_idx[id_agent], id_agent)
        idx = agent_to_idx[id_agent]

        time_step += 1
        acting_agent = agents_dict[id_agent]

        act = acting_agent.select_action(obs) if not done else None
        env.step(act)

        obs, rew, done, info = env.last()

        acting_agent.buffer.rewards.append(rew)
        acting_agent.buffer.is_terminals.append(done)
        acting_agent.tmp_return += rew

        time_step +=1
        if rew != None:
            current_ep_reward[idx] += rew

        # update PPO agent
        if time_step % update_timestep == 0:
            acting_agent.update()

        if time_step % print_freq == 0:
            print_avg_reward = np.zeros(3)
            for k in range(3):
                #print average reward till last episode
                #print("print_running_reward=", print_running_reward, "print_running_reward[idx]=", print_running_reward[idx])
                #print("print_running_episodes=", print_running_episodes, "print_running_episodes[idx]", print_running_episodes[idx])
                print_avg_reward[k] = print_running_reward[k] / print_running_episodes[k]
                #print(print_avg_reward[k])
                print_avg_reward[k] = round(print_avg_reward[k], 2)

            print("Episode : {} \t\t Timestep : {} \t\t ".format(i_episode, time_step))
            print("Average Reward : agent0={}, angent1={}, adv0={}".format(print_avg_reward[0], print_avg_reward[1], print_avg_reward[2]))

            print_running_reward[0] = 0; print_running_reward[1] = 0; print_running_reward[2] = 0
            print_running_episodes[0] = 0; print_running_episodes[1] = 0; print_running_episodes[2] = 0

        # break; if the episode is over
        if done:
            #print("acting_agent=", acting_agent)
            agent0_PPO.train_returns.append(agent0_PPO.tmp_return)
            agent1_PPO.train_returns.append(agent1_PPO.tmp_return)
            adversary0_PPO.train_returns.append(adversary0_PPO.tmp_return)
        if (done and idx==2):   
            break

        i_internal_loop += 1
    
    print_running_reward[0] += current_ep_reward[0]
    print_running_reward[1] += current_ep_reward[1]
    print_running_reward[2] += current_ep_reward[2]
    print_running_episodes[0] += 1; print_running_episodes[1] += 1; print_running_episodes[2] += 1

    i_episode += 1

### PLOT TRAIN RETURNS
mov_avg_agent0 = moving_average(agent0_PPO.train_returns, num_blocks)
mov_avg_agent1 = moving_average(agent1_PPO.train_returns, num_blocks)
mov_avg_adv0 = moving_average(adversary0_PPO.train_returns, num_blocks)

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle("Train Returns")
ax1.plot(np.linspace(0, len(agent0_PPO.train_returns), len(agent0_PPO.train_returns)), agent0_PPO.train_returns)
ax1.plot(np.linspace(0, len(agent0_PPO.train_returns), len(mov_avg_agent0)), mov_avg_agent0)
ax2.plot(np.linspace(0, len(agent1_PPO.train_returns), len(agent1_PPO.train_returns)), agent1_PPO.train_returns)
ax2.plot(np.linspace(0, len(agent1_PPO.train_returns), len(mov_avg_agent1)), mov_avg_agent1)
ax3.plot(np.linspace(0, len(adversary0_PPO.train_returns), len(adversary0_PPO.train_returns)), adversary0_PPO.train_returns)
ax3.plot(np.linspace(0, len(adversary0_PPO.train_returns), len(mov_avg_adv0)), mov_avg_adv0)
plt.savefig("images/simple_adv/train_returns_simple_adv.png")

### EVALUATION
print("\n\nEVALUATION \n\n")

rews_after = evaluation(agents_dict, eval_eps)
plot_hist_returns(rews_before, rews_after)
plotting_rews(rews_before, rews_after)

print("computing test episodes and saving csv")
### TEST EPISODES AND SAVE CSV
total_test_episodes = 1000
test_running_reward = np.zeros(3)
df = pd.DataFrame(columns=['episode', 't', 'agent0_ret', 'agent1_ret', 'adv0_ret'])
line = 0
for ep in range(0, total_test_episodes):

    ep_reward = np.zeros(3)
    env.reset()
    obs, rew, done, info = env.last()
    i_internal_loop = 1
    step = 0
    
    for id_agent in env.agent_iter(max_ep_len):
        idx = agent_to_idx[id_agent]

        acting_agent = agents_dict[id_agent]
        act = acting_agent.select_action(obs) if not done else None

        env.step(act)
        obs, rew, done, info = env.last()
        #env.render(mode='human')
        ep_reward[idx] += rew
        
        if done:
            #print("done")
            break

        if (i_internal_loop % 3 == 0):
            line += 1
            df.loc[line] = [ep, step, ep_reward[0], ep_reward[1], ep_reward[2]]
            step += 1

        i_internal_loop +=1

    # clear buffer    
    acting_agent.buffer.clear()

    test_running_reward[0] += ep_reward[0]
    test_running_reward[1] += ep_reward[1]
    test_running_reward[2] += ep_reward[2]
    #print('Episode: {} \t\t Reward agent0: {}, agent1: {}, adv0: {}'.format(ep, round(ep_reward[0], 2), round(ep_reward[1], 2), round(ep_reward[2], 2)))
    ep_reward = 0

df.to_csv("simple_adversary.csv")
print("done")

env.close()