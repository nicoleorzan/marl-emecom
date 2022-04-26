from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from pettingzoo.mpe import simple_v2
import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy


cycles = 30
eval_eps = 1000
learning_steps = 50000

def evaluate_env(model):
    env = simple_v2.env(max_cycles=cycles, continuous_actions=False)
    env.reset()
    rews_ep = []
    for agent in env.agent_iter():
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
    env.close()


def plotting_rews(rews_b, rews_a): # rews is a LIST OF LISTS for num of episodes to plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    fig.suptitle("Rewards of agent in time for one episode", fontsize=20)
    arrays_b = [np.array(x) for x in rews_b]
    meanrews_b = [np.mean(k) for k in zip(*arrays_b)]
    sdrews_b = [np.std(k) for k in zip(*arrays_b)]
    arrays_a = [np.array(x) for x in rews_a]
    meanrews_a = [np.mean(k) for k in zip(*arrays_a)]
    sdrews_a = [np.std(k) for k in zip(*arrays_a)]
    
    for i in range(len(rews_a)):
        ax1.plot(np.linspace(0, len(rews_b[i]), len(rews_b[i])), rews_b[i], linestyle='dashed')
        ax2.plot(np.linspace(0, len(rews_a[i]), len(rews_a[i])), rews_a[i], linestyle='dashed')
    ax1.errorbar(np.linspace(0, len(rews_b[i]), len(rews_b[i])), meanrews_b, sdrews_b, color='black', linewidth=3, label="before_learning")
    ax2.errorbar(np.linspace(0, len(rews_a[i]), len(rews_a[i])), meanrews_a, sdrews_a, color='black', linewidth=3, label="after_learning")
    ax1.set_xlabel('Time', fontsize=18)
    ax2.set_xlabel('Time', fontsize=18)
    ax1.set_ylabel('Reward', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax1.legend(fontsize=18)
    ax2.legend(fontsize=18)
    plt.savefig("rewards_simple_v2.png")


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
    rews_list_before = []
    for i in range(eval_eps):
        rews_b = evaluate_env(model)
        rews_list_before.append(rews_b)

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

    rews_list_after = []
    for i in range(eval_eps):
        rews_a = evaluate_env(model)
        rews_list_after.append(rews_a)
    plotting_rews(rews_list_before, rews_list_after)

    # ===================== OBSERVE =====================
    # ===================================================
    print("OSERVE AGENT")
    observe(model)
