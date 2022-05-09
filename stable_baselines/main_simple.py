from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from pettingzoo.mpe import simple_v2
import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt

cycles = 30
eval_eps = 1000
learning_steps = 100000

def evaluation(model, episodes):

    rews = np.zeros((episodes, cycles+1))
    for e in range(episodes):
        rews[e] = evaluate_episode(model)

    return rews

def evaluate_episode(model):
    env = simple_v2.env(max_cycles=cycles, continuous_actions=False)
    env.reset()
    rews_ep = []
    for _ in env.agent_iter():
        obs, reward, done, _ = env.last()
        rews_ep.append(reward)
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
    env.close()
    return rews_ep


def observe(model):
    env = simple_v2.env(max_cycles=cycles, continuous_actions=False)
    env.reset()
    for _ in env.agent_iter():
        obs, reward, done, _ = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        env.render()
        if done:
            break
    env.close()

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


if __name__ == "__main__":

    # =================== SET UP ENVIRONMENT ====================
    # ===========================================================

    env = simple_v2.parallel_env(max_cycles=cycles, continuous_actions=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')

    model = PPO(MlpPolicy, env, verbose=1, gamma=0.95, n_steps=256, ent_coef=0.0905168, \
        learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, \
        n_epochs=5, clip_range=0.3, batch_size=256)

    # eval prior to learing to show that learning worked
    print("\nEALUATION BEFORE LEARNING")

    rews_before = evaluation(model, eval_eps)

    # =================== LEARNING ======================
    # ===================================================
    print("\nLEARNING")
    model.learn(total_timesteps=learning_steps)
    print("SAVING MODEL...")
    model.save("policy_simple")

    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    #print("mean rew=", mean_reward)
    #print("std rew=", std_reward)

    env.close()
    del model

    # =================== EVALUATION ====================
    # ===================================================
    print("\nEALUATION AFTER LEARNING")

    model = PPO.load("policy_simple")

    rews_after = evaluation(model, eval_eps)
    print("Plotting hist returns...")
    plot_hist_returns(rews_before, rews_after)
    print("done.\nPlatting rewards...")
    plotting_rews(rews_before, rews_after)
    print("done.")

    # ===================== OBSERVE =====================
    # ===================================================
    #print("OSERVE AGENT")
    #observe(model)
