from pettingzoo.mpe import simple_adversary_v2
from PPO import ActorCritic
env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=False)

env.reset()
observation, reward, done, info = env.last()
print(observation)

#agent1 = ActorCritic(input_dim=5, action_dim=5)
#agent2 = ActorCritic(input_dim=5, action_dim=5)
#adversary = ActorCritic(action_dim=5)
for agent in env.agent_iter():
    observation, reward, done, info = env.last()
#action = policy(observation, agent)
#env.step(action)

#if done:
#    env.reset()
