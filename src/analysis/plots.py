import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


df_no_comm = pd.read_csv("all_returns_no_comm.csv")
df_comm = pd.read_csv("comm/all_returns_comm.csv")
df_noise = pd.read_csv("comm/noise/all_returns_comm_noise.csv")
n_agents = 3

fig, ax = plt.subplots(n_agents, 1, figsize=(7, n_agents*3))
for ag in range(n_agents):
    print("ag=", ag)
    sns.lineplot(data=df_no_comm, x="episode", y="ret_ag"+str(ag), ax=ax[ag], label='no_comm')
    sns.lineplot(data=df_comm, x="episode", y="ret_ag"+str(ag), ax=ax[ag], label='comm')
    sns.lineplot(data=df_noise, x="episode", y="ret_ag"+str(ag), ax=ax[ag], label='noise')
    ax[ag].grid(alpha=0.6)
plt.show()


fig, ax = plt.subplots(n_agents, 1, figsize=(7, n_agents*3))
for ag in range(n_agents):
    print("ag=", ag)
    sns.lineplot(data=df_no_comm, x="episode", y="coop_ag"+str(ag), ax=ax[ag], label='no_comm')
    sns.lineplot(data=df_comm, x="episode", y="coop_ag"+str(ag), ax=ax[ag], label='comm')
    sns.lineplot(data=df_noise, x="episode", y="coop_ag"+str(ag), ax=ax[ag], label='noise')
    ax[ag].grid(alpha=0.6)
plt.show()
