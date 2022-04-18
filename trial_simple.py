from pettingzoo.mpe import simple_v2
from PPO import PPO
import matplotlib.pyplot as plt
import numpy as np

action_space = 5

agent0_PPO = PPO(input_dim=4, action_dim=action_space, lr_actor=0.01, lr_critic=0.01,  \
    gamma=0.9, K_epochs=10, eps_clip=0.2, c1=0.2, c2=0.2)

env = simple_v2.env(max_cycles=25, continuous_actions=False)

# training loop
time_step = 0
i_episode = 0
max_ep_len = 900                    # max timesteps in one episode
update_timestep = max_ep_len * 4
print_freq = max_ep_len * 4  
max_training_timesteps = int(600)

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

# training loop
while time_step <= max_training_timesteps:

    env.reset()
    current_ep_reward = 0
    
    #agent0_PPO.tmp_return = 0

    for id_agent in env.agent_iter(max_ep_len):

        obs, rew, done, info = env.last()
        act = agent0_PPO.select_action(obs) if not done else None

        env.step(act)

        agent0_PPO.buffer.rewards.append(rew)
        agent0_PPO.buffer.is_terminals.append(done)
        #agent0_PPO.tmp_return += rew

        time_step +=1
        current_ep_reward += rew

        # update PPO agent
        if time_step % update_timestep == 0:
            print("\nUpdate agent 0")
            agent0_PPO.update()

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
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    i_episode += 1
    
    agent0_PPO.train_returns.append(agent0_PPO.tmp_return) 


#env.close()

total_test_episodes = 10  

test_running_reward = 0

for ep in range(1, total_test_episodes+1):
    ep_reward = 0
    state = env.reset()
    
    for id_agent in env.agent_iter(max_ep_len):
        obs, rew, done, info = env.last()
        act = agent0_PPO.select_action(obs) if not done else None

        env.step(act)
        ep_reward += rew
        
        if done:
            break

    # clear buffer    
    agent0_PPO.buffer.clear()

    test_running_reward +=  ep_reward
    print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
    ep_reward = 0

env.close()

"""
fig, (ax1) = plt.subplots(1)
fig.suptitle("Train Returns")
ax1.plot(np.linspace(0, len(agent0_PPO.train_returns), len(agent0_PPO.train_returns)), agent0_PPO.train_returns)
plt.show()

# ===========================================================================
# =============================== EVALUATION ================================
# ===========================================================================

print("\n\nEVALUATIOOOOOOOOOON \n\n")
env1 = simple_v2.env(max_cycles=25, continuous_actions=False)
env1.reset()
agent0_PPO.ep_rewards = []
i = 0
for id_agent in env1.agent_iter(100):
    obs, rew, done, info = env1.last()
    act = agent0_PPO.select_action(obs) if not done else None
    agent0_PPO.ep_rewards.append(rew)

    env1.step(act)
    i += 1

env1.close()

fig, (ax1) = plt.subplots(1)
fig.suptitle("Rewards of agents in time for one episode")
ax1.plot(np.linspace(0, len(agent0_PPO.ep_rewards), len(agent0_PPO.ep_rewards)), agent0_PPO.ep_rewards)
plt.show()"""