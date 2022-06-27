import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ret = pd.read_csv("avg_returns.csv")   
ret_comm = pd.read_csv("comm/avg_returnscomm.csv")   
coop = pd.read_csv("avg_cooperativeness.csv")
coop_comm = pd.read_csv("comm/avg_cooperativenesscomm.csv")

n_agents = 3
fig, ax = plt.subplots(n_agents)
fig.suptitle("AVG Cooperativity")
for ag_idx in range(n_agents):
    ax[ag_idx].plot(np.linspace(0, len(coop_comm.loc[ag_idx]), len(coop_comm.loc[ag_idx])), coop_comm.loc[ag_idx], color='blue', label='comm')
    ax[ag_idx].plot(np.linspace(0, len(coop.loc[ag_idx]), len(coop.loc[ag_idx])), coop.loc[ag_idx], color='red', label='no_comm')
    ax[ag_idx].grid()
    ax[ag_idx].set_ybound(0,1)
ax[0].legend()
plt.savefig("coop.png")    

n_agents = 3
fig, ax = plt.subplots(n_agents)
fig.suptitle("AVG Train returns")
for ag_idx in range(n_agents):
    ax[ag_idx].plot(np.linspace(0, len(ret_comm.loc[ag_idx]), len(ret_comm.loc[ag_idx])), ret_comm.loc[ag_idx], color='blue', label='comm')
    ax[ag_idx].plot(np.linspace(0, len(ret.loc[ag_idx]), len(ret.loc[ag_idx])), ret.loc[ag_idx], color='red', label='no_comm')
    ax[ag_idx].grid()
ax[0].legend()
plt.savefig("returns.png")    
