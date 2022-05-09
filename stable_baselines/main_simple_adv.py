from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from pettingzoo.mpe import simple_adversary_v2
from stable_baselines3.common.utils import set_random_seed
import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt
from supersuit import pad_observations_v0
from save_gif import _label_with_episode_number
import imageio

cycles = 30
eval_eps = 1000
learning_steps = 500000
n_agents = 3

def evaluation(model, episodes):

    adv0r = np.zeros((episodes, cycles))
    agent0r = np.zeros((episodes, cycles))
    agent1r = np.zeros((episodes, cycles))
    for e in range(episodes):
        if (e%100 == 0):
            print("Episode:", e)
        adv0r[e], agent0r[e], agent1r[e] = evaluate_episode(model)

    return [adv0r, agent0r, agent1r]


def evaluate_episode(model):
    env = simple_adversary_v2.env(max_cycles=cycles, continuous_actions=False)
    env = pad_observations_v0(env)
    env.reset()
    agent0r = np.zeros(cycles); agent1r = np.zeros(cycles); adv0r = np.zeros(cycles)
    i = 0
    for agent in env.agent_iter():
        obs, reward, done, _ = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        if (i > 2):
            if (agent == 'adversary_0'):
                adv0r[int(i/n_agents)-2] = reward
            elif (agent == 'agent_0'):
                agent0r[int(i/n_agents)-3] = reward
            elif (agent == 'agent_1'):
                agent1r[int(i/n_agents)-4] = reward
        i += 1
    env.close()
    return adv0r, agent0r, agent1r


def observe(model, frames = None):
    env = simple_adversary_v2.env(max_cycles=cycles, continuous_actions=False)
    env = pad_observations_v0(env)
    env.reset()
    for _ in env.agent_iter():
        obs, _, done, _ = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        if frames != None:
            frame = env.render(mode="rgb_array")
            frames.append(_label_with_episode_number(frame, episode_num=i))
        else:
            env.render()

        if done:
                break
    env.close()
    if frames != None:
        imageio.mimwrite('agent.gif', frames, fps=120)

def plot_hist_returns(rews_before, rews_after):

    adv0r, agent0r, agent1r = rews_before
    returns_adv0b = np.sum(adv0r, axis=1)
    returns_ag0b = np.sum(agent0r, axis=1)
    returns_ag1b = np.sum(agent1r, axis=1)

    adv0r, agent0r, agent1r = rews_after
    returns_adv0a = np.sum(adv0r, axis=1)
    returns_ag0a = np.sum(agent0r, axis=1)
    returns_ag1a = np.sum(agent1r, axis=1)

    n_bins = 40
    fig, ax = plt.subplots(n_agents, 2, figsize=(20,8))
    fig.suptitle("Distribution of Returns", fontsize=25)
    ax[0,0].hist(returns_adv0b, bins=n_bins, range=[-120., 0.], label='adv')
    ax[0,1].hist(returns_adv0a, bins=n_bins, range=[-120., 0.], label='adv')
    ax[1,0].hist(returns_ag0b, bins=n_bins, range=[-30., 60.], label='ag0')
    ax[1,1].hist(returns_ag0a, bins=n_bins, range=[-30., 60.], label='ag0')
    ax[2,0].hist(returns_ag1b, bins=n_bins, range=[-30., 60.], label='ag1')
    ax[2,1].hist(returns_ag1a, bins=n_bins, range=[-30., 60.], label='ag1')
    print("Saving histogram..")
    plt.savefig("images/simple_adv/hist_rewards_simple_adv.png")


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


if __name__ == "__main__":

    set_random_seed(123)
    np.random.seed(123)

    # =================== SET UP ENVIRONMENT ====================
    # ===========================================================

    env = simple_adversary_v2.parallel_env(N=2, max_cycles=cycles, continuous_actions=False)
    env = pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')

    model = PPO(MlpPolicy, env, verbose=1, gamma=0.95, n_steps=256, ent_coef=0.09, \
        learning_rate=0.0006, vf_coef=0.05, max_grad_norm=0.9, gae_lambda=0.99, \
        n_epochs=5, clip_range=0.2, batch_size=256, seed=123)

    # eval prior to learing to show that learning worked
    print("\nEALUATION BEFORE LEARNING")
    rews_before = evaluation(model, eval_eps)

    # =================== LEARNING ======================
    # ===================================================
    print("\nLEARNING")
    model.learn(total_timesteps=learning_steps)
    print("SAVING MODEL...")
    model.save("policy_simple_adv")

    env.close()
    del model

    # =================== EVALUATION ====================
    # ===================================================
    print("\nEALUATION AFTER LEARNING")

    model = PPO.load("policy_simple_adv")
    
    rews_after = evaluation(model, eval_eps)

    plot_hist_returns(rews_before, rews_after)
    plotting_rews(rews_before, rews_after)
    print("done")

    # ===================== OBSERVE =====================
    # ===================================================
    print("OSERVE AGENT")
    frames = []
    for i in range(2):
        observe(model)
    print("finished observation")