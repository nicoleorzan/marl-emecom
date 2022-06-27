import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("simple_adversary.csv")
n_episodes = len(df.episode.unique())

ag0 = df.groupby('t', as_index=False)['agent0_ret'].mean()
ag1 = df.groupby('t', as_index=False)['agent1_ret'].mean()
adv0 = df.groupby('t', as_index=False)['adv0_ret'].mean()

ag0_sd = df.groupby('t', as_index=False)['agent0_ret'].std()
ag1_sd = df.groupby('t', as_index=False)['agent1_ret'].std()
adv0_sd = df.groupby('t', as_index=False)['adv0_ret'].std()

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle("Rewards of agents in time for one episode")
#errorbar(xval, yval, xerr = 0.4, yerr = 0.5)
ax1.errorbar(np.linspace(0, len(ag0['agent0_ret']), len(ag0['agent0_ret'])), ag0['agent0_ret'], ag0_sd['agent0_ret'], label='mean_test_reward_agent_0')
ax2.errorbar(np.linspace(0, len(ag0['agent0_ret']), len(ag0['agent0_ret'])), ag1['agent1_ret'], ag1_sd['agent1_ret'], label='mean_test_reward_agent_1')
ax3.errorbar(np.linspace(0, len(ag0['agent0_ret']), len(ag0['agent0_ret'])), adv0['adv0_ret'], adv0_sd['adv0_ret'], label='mean_test_reward_adversary')

#plt.plot(np.linspace(0, len(ag0['agent0_ret']), len(ag0['agent0_ret'])), ag0['agent0_ret'], label='mean_test_reward_agent_0')
plt.show()