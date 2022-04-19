from pettingzoo.mpe import simple_v2
from PPO import PPO
import matplotlib.pyplot as plt
import numpy as np

env = simple_v2.env(max_cycles=25, continuous_actions=False)

max_ep_len = 400                    # max timesteps in one episode
max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)

update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

input_dim = 4
action_space = 5

agent = PPO(input_dim, action_space, lr_actor, lr_critic,  \
    gamma, K_epochs, eps_clip, c1=0.2, c2=0.2)

test_running_reward = 0

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

time_step = 0
i_episode = 0

while time_step <= max_training_timesteps:

    env.reset()
    obs, rew, done, info = env.last()

    current_ep_reward = 0 
    i_internal_loop = 0
    agent.tmp_return = 0

    for id_agent in env.agent_iter(max_ep_len):

        time_step += 1
        #print("time step", time_step)

        act = agent.select_action(obs) if not done else None
        env.step(act)

        obs, rew, done, info = env.last()
        #print(obs, rew, done, info)
        
        #if (i_internal_loop != 0):
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

#env.close()

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

    # clear buffer    
    agent.buffer.clear()

    test_running_reward += ep_reward
    print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
    ep_reward = 0

env.close()


fig, (ax1) = plt.subplots(1)
fig.suptitle("Train Returns")
ax1.plot(np.linspace(0, len(agent.train_returns), len(agent.train_returns)), agent.train_returns)
plt.show()

# ===========================================================================
# =============================== EVALUATION ================================
# ===========================================================================

print("\n\nEVALUATIOOOOOOOOOON \n\n")
env1 = simple_v2.env(max_cycles=25, continuous_actions=False)
env1.reset()
agent.ep_rewards = []
i = 0
for id_agent in env1.agent_iter(100):
    obs, rew, done, info = env1.last()
    act = agent.select_action(obs) if not done else None
    agent.ep_rewards.append(rew)

    env1.step(act)
    i += 1

env1.close()

fig, (ax1) = plt.subplots(1)
fig.suptitle("Rewards of agents in time for one episode")
ax1.plot(np.linspace(0, len(agent.ep_rewards), len(agent.ep_rewards)), agent.ep_rewards)
plt.show()