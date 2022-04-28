from pettingzoo.mpe import simple_v2
from PPO import PPO
import matplotlib.pyplot as plt
import numpy as np

cycles = 30
env = simple_v2.env(max_cycles=cycles, continuous_actions=False)

max_ep_len = 100                    # max timesteps in one episode
max_training_timesteps = int(5e4)   # break training loop if timeteps > max_training_timesteps
eval_eps = 1000

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
c1 = 0.5
c2 = -0.01

input_dim = 4
action_space = 5

agent = PPO(input_dim, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)
untrained_agent = PPO(input_dim, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1, c2)

test_running_reward = 0

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

time_step = 0
i_episode = 0

def evaluation(agent, episodes):

    rews = np.zeros((episodes, cycles+1))
    for e in range(episodes):
        rews[e] = evaluate_episode(agent)

    return rews

def evaluate_episode(agent):
    env = simple_v2.env(max_cycles=cycles, continuous_actions=False)
    env.reset()
    rews_ep = []
    for _ in env.agent_iter():
        obs, reward, done, _ = env.last()
        rews_ep.append(reward)
        act = agent.select_action(obs) if not done else None
        env.step(act)
    env.close()
    return rews_ep

def plot_hist_returns(rews_before, rews_after):

    returns_agb = np.sum(rews_before, axis=1)
    returns_aga = np.sum(rews_after, axis=1)

    n_bins = 40
    fig, ax = plt.subplots(1, 2, figsize=(20,8))
    fig.suptitle("Distribution of Returns", fontsize=25)
    ax[0].hist(returns_agb, bins=n_bins, range=[-120., 100.])
    ax[1].hist(returns_aga, bins=n_bins, range=[-120., 100.])
    print("Saving histogram")
    plt.savefig("images/simple/hist_rewards_simple.png")


def plotting_rews(rews_b, rews_a): # rews is a np array with the all the rewards for n episodes
    
    meanrews_b = np.average(rews_b, axis=0)
    sdrews_b = np.std(rews_b, axis=0)
    meanrews_a = np.average(rews_a, axis=0)
    sdrews_a = np.std(rews_a, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    fig.suptitle("Rewards of agent in time for one episode", fontsize=20)
    
    for i in range(rews_a.shape[0]):
        ax1.plot(np.linspace(0, rews_b.shape[1], rews_b.shape[1]), rews_b[i,:], linestyle='dashed')
        ax2.plot(np.linspace(0, rews_a.shape[1], rews_a.shape[1]), rews_a[i,:], linestyle='dashed')
    ax1.errorbar(np.linspace(0, rews_b.shape[1], rews_b.shape[1]), meanrews_b, sdrews_b, color='black', linewidth=3, label="before_learning")
    ax2.errorbar(np.linspace(0, rews_a.shape[1], rews_a.shape[1]), meanrews_a, sdrews_a, color='black', linewidth=3, label="after_learning")
    ax1.set_xlabel('Time', fontsize=18)
    ax2.set_xlabel('Time', fontsize=18)
    ax1.set_ylabel('Reward', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax1.legend(fontsize=18)
    ax2.legend(fontsize=18)
    plt.savefig("images/simple/rewards_simple_v2.png")


print("\nEVALUATION BEFORE LEARNING")
rews_before = evaluation(untrained_agent, eval_eps)

#### TRAINING LOOP

while time_step <= max_training_timesteps:

    env.reset()
    obs, rew, done, info = env.last()

    current_ep_reward = 0 
    i_internal_loop = 0
    agent.tmp_return = 0

    for id_agent in env.agent_iter(max_ep_len):

        time_step += 1

        act = agent.select_action(obs) if not done else None
        env.step(act)

        obs, rew, done, info = env.last()
        
        agent.buffer.rewards.append(rew)
        agent.buffer.is_terminals.append(done)
        agent.tmp_return += rew

        time_step +=1
        current_ep_reward += rew

        # update PPO agent
        if time_step % update_timestep == 0:
            agent.update()

        # printing average reward
        if time_step % print_freq == 0:
            # print average reward till last episode
            print("print_running_episodes=", print_running_episodes)
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0

        # break; if the episode is over
        if done:
            agent.train_returns.append(agent.tmp_return)
            break

        i_internal_loop += 1

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    i_episode += 1

fig, (ax1) = plt.subplots(1)
fig.suptitle("Train Returns")
ax1.plot(np.linspace(0, len(agent.train_returns), len(agent.train_returns)), agent.train_returns)
plt.savefig("images/simple/train_returns_simple.png")


### EVALUATION
print("\n\nEVALUATION \n\n")

rews_after = evaluation(agent, eval_eps)
plot_hist_returns(rews_before, rews_after)
plotting_rews(rews_before, rews_after)


### OBSERVE AGENT LOOP

total_test_episodes = 10  
test_running_reward = 0

for ep in range(1, total_test_episodes+1):
    ep_reward = 0
    state = env.reset()
    
    for id_agent in env.agent_iter(max_ep_len):
        obs, rew, done, info = env.last()
        act = agent.select_action(obs) if not done else None

        env.step(act)
        env.render(mode='human')
        ep_reward += rew
        
        if done:
            break

    agent.buffer.clear()

    test_running_reward += ep_reward
    print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
    ep_reward = 0

env.close()