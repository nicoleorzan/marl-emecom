import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


def plots(df_no_comm, df_comm, path=""):
    col = df_comm.loc[: , "ret_ag0":"ret_ag2"]
    df_comm['avg_rew'] = col.mean(axis=1)
    col = df_no_comm.loc[: , "ret_ag0":"ret_ag2"]
    df_no_comm['avg_rew'] = col.mean(axis=1)
    col = df_comm_RND.loc[: , "ret_ag0":"ret_ag2"]
    df_comm_RND['avg_rew'] = col.mean(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.lineplot(data=df_no_comm, x="episode", y="avg_rew", ax=ax, label='no_comm')
    sns.lineplot(data=df_comm, x="episode", y="avg_rew", ax=ax, label='comm')
    sns.lineplot(data=df_comm_RND, x="episode", y="avg_rew", ax=ax, label='comm_RND')
    ax.grid(alpha=0.6)
    plt.savefig(path+"group_reward.png")

    fig, ax = plt.subplots(n_agents, 1, figsize=(7, n_agents*3))
    for ag in range(n_agents):
        sns.lineplot(data=df_no_comm, x="episode", y="ret_ag"+str(ag), ax=ax[ag], label='no_comm')
        sns.lineplot(data=df_comm, x="episode", y="ret_ag"+str(ag), ax=ax[ag], label='comm')
        sns.lineplot(data=df_comm_RND, x="episode", y="ret_ag"+str(ag), ax=ax[ag], label='random_baseline')
        ax[ag].grid(alpha=0.6)
    plt.savefig(path+"agents_returns.png")

    fig, ax = plt.subplots(n_agents, 1, figsize=(7, n_agents*3))
    for ag in range(n_agents):
        sns.lineplot(data=df_no_comm, x="episode", y="coop_ag"+str(ag), ax=ax[ag], label='no_comm')
        sns.lineplot(data=df_comm, x="episode", y="coop_ag"+str(ag), ax=ax[ag], label='comm')
        sns.lineplot(data=df_comm_RND, x="episode", y="coop_ag"+str(ag), ax=ax[ag], label='random_baseline')
        ax[ag].grid(alpha=0.6)
    plt.savefig(path+"agents_cooperativity.png")


if __name__ == "__main__":

    df_no_comm = pd.read_csv("data_no_comm.csv")
    df_comm = pd.read_csv("comm/data_comm.csv")
    df_comm_RND = pd.read_csv("comm/random_baseline/data_comm.csv")
    n_agents = 3
    plots(df_no_comm, df_comm)
