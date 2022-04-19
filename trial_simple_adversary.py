from pettingzoo.mpe import simple_adversary_v2
from PPO import PPO
import matplotlib.pyplot as plt
import numpy as np

action_space = 5

agent0_PPO = PPO(input_dim=10, action_dim=action_space, lr_actor=0.01, lr_critic=0.01,  \
    gamma=0.9, K_epochs=10, eps_clip=0.2, c1=0.2, c2=0.2)
agent1_PPO = PPO(input_dim=10, action_dim=action_space, lr_actor=0.01, lr_critic=0.01,  \
    gamma=0.9, K_epochs=10, eps_clip=0.2, c1=0.2, c2=0.2)
adversary0_PPO = PPO(input_dim=8, action_dim=action_space, lr_actor=0.01, lr_critic=0.01,  \
    gamma=0.9, K_epochs=10, eps_clip=0.2, c1=0.2, c2=0.2)

env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=False)

agents_dict = {"agent_0": agent0_PPO, "agent_1": agent1_PPO, "adversary_0": adversary0_PPO}
NUM_RESETS = 100
for i in range(NUM_RESETS):
    print("i=", i)
    env.reset()
    agent0_PPO.tmp_return = 0; agent1_PPO.tmp_return = 0; adversary0_PPO.tmp_return = 0
    j = 0
    for id_agent in env.agent_iter(100):
        #print("\ni=", i, "id_agent=", id_agent)
        #print("action_space=",env.action_space(agent))
        obs, rew, done, info = env.last()
        acting_agent = agents_dict[id_agent]
        act = acting_agent.select_action(obs) if not done else None
        #print("act=", act)

        env.step(act)
        if (j > 2):
            acting_agent.buffer.rewards.append(rew)
            acting_agent.tmp_return += rew
            acting_agent.buffer.is_terminals.append(done)
        j += 1

    agent0_PPO.train_returns.append(agent0_PPO.tmp_return) 
    agent1_PPO.train_returns.append(agent1_PPO.tmp_return) 
    adversary0_PPO.train_returns.append(adversary0_PPO.tmp_return) 
    # should I update at the end of one episode, at the end of N episodes, 
    # or after a certain num of time steps?
    # For now I update at the end of every episode

    #print("\nUpdate agent 0")
    #print(len(agent0_PPO.buffer.rewards))
    agent0_PPO.update()
    #print("\nUpdate agent 1")
    #print(len(agent1_PPO.buffer.rewards))
    agent1_PPO.update()
    #print("\nUpdate adversary 0")
    adversary0_PPO.update()
    #print("\nEND")

env.close()


fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle("Train Returns")
ax1.plot(np.linspace(0, len(agent0_PPO.train_returns), len(agent0_PPO.train_returns)), agent0_PPO.train_returns)
ax2.plot(np.linspace(0, len(agent1_PPO.train_returns), len(agent1_PPO.train_returns)), agent1_PPO.train_returns)
ax3.plot(np.linspace(0, len(adversary0_PPO.train_returns), len(adversary0_PPO.train_returns)), adversary0_PPO.train_returns)
plt.show()

# ===========================================================================
# =============================== EVALUATION ================================
# ===========================================================================

print("\n\nEVALUATIOOOOOOOOOON \n\n")
env1 = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=False)
env1.reset()
agent0_PPO.ep_rewards = []; agent1_PPO.ep_rewards = []; adversary0_PPO.ep_rewards = []
i = 0
for id_agent in env1.agent_iter(100):
    #print("id_agent=", id_agent)
    obs, rew, done, info = env1.last()
    acting_agent = agents_dict[id_agent]
    act = acting_agent.select_action(obs) if not done else None
    acting_agent.ep_rewards.append(rew)

    env1.step(act)
    i += 1

env1.close()

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle("Rewards of agents in time for one episode")
ax1.plot(np.linspace(0, len(agent0_PPO.ep_rewards), len(agent0_PPO.ep_rewards)), agent0_PPO.ep_rewards)
ax2.plot(np.linspace(0, len(agent1_PPO.ep_rewards), len(agent1_PPO.ep_rewards)), agent1_PPO.ep_rewards)
ax3.plot(np.linspace(0, len(adversary0_PPO.ep_rewards), len(adversary0_PPO.ep_rewards)), adversary0_PPO.ep_rewards)
plt.show()