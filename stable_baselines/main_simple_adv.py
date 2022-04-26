from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from pettingzoo.mpe import simple_adversary_v2
from stable_baselines3.common.utils import set_random_seed
import supersuit as ss
import numpy as np
import matplotlib.pyplot as plt
from supersuit import pad_observations_v0

cycles = 30
eval_eps = 50
learning_steps = 100000

def evaluation(model, episodes):

    agent0r = np.zeros((episodes, cycles))
    agent1r = np.zeros((episodes, cycles))
    adv0r = np.zeros((episodes, cycles))
    for e in range(episodes):
        if (e%100 == 0):
            print("Episode:", e)
        agent0r[e], agent1r[e], adv0r[e] = evaluate_env(model)

    return agent0r, agent1r, adv0r


def evaluate_env(model):
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
                adv0r[int(i/3)-2] = reward
            elif (agent == 'agent_0'):
                agent0r[int(i/3)-3] = reward
            elif (agent == 'agent_1'):
                agent1r[int(i/3)-4] = reward
        i += 1
    env.close()
    return agent0r, agent1r, adv0r


def observe(model):
    env = simple_adversary_v2.env(max_cycles=cycles, continuous_actions=False)
    env = pad_observations_v0(env)
    env.reset()
    for _ in env.agent_iter():
        obs, _, done, _ = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        env.render()
    env.close()


def plotting_rews(rews_b, rews_a): # rews is a LIST OF LISTS for every agent of the num of episodes to plot
    
    agent0rb, agent1rb, adv0rb = rews_b
    agent0ra, agent1ra, adv0ra = rews_a
    
    fig, ax = plt.subplots(3, 2, figsize=(20,8))
    fig.suptitle("Rewards of agent in time for one episode", fontsize=25)

    meanrews_b = np.zeros((3, len(agent0rb[0])))
    sdrews_b = np.zeros((3, len(agent0rb[0])))

    meanrews_a = np.zeros((3, len(agent0ra[0])))
    sdrews_a = np.zeros((3, len(agent0ra[0])))

    meanrews_b[0] = [np.mean(k) for k in zip(*agent0rb)]
    meanrews_b[1] = [np.mean(k) for k in zip(*agent1rb)]
    meanrews_b[2] = [np.mean(k) for k in zip(*adv0rb)]
    sdrews_b[0] = [np.std(k) for k in zip(*agent0rb)]
    sdrews_b[1] = [np.std(k) for k in zip(*agent1rb)]
    sdrews_b[2] = [np.std(k) for k in zip(*adv0rb)]

    meanrews_a[0] = [np.mean(k) for k in zip(*agent0ra)]
    meanrews_a[1] = [np.mean(k) for k in zip(*agent1ra)]
    meanrews_a[2] = [np.mean(k) for k in zip(*adv0ra)]
    sdrews_a[0] = [np.std(k) for k in zip(*agent0ra)]
    sdrews_a[1] = [np.std(k) for k in zip(*agent1ra)]
    sdrews_a[2] = [np.std(k) for k in zip(*adv0ra)]


    for i in range(3):
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
    plt.savefig("rewards_simple_adversary_v2.png")


if __name__ == "__main__":

    set_random_seed(123)
    np.random.seed(123)

    # =================== SET UP ENVIRONMENT ====================
    # ===========================================================

    env = simple_adversary_v2.parallel_env(N=2, max_cycles=cycles, continuous_actions=False)
    env.seed(123)
    env = pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')

    model = PPO(MlpPolicy, env, verbose=1, gamma=0.95, n_steps=256, ent_coef=0.09, \
        learning_rate=0.0006, vf_coef=0.05, max_grad_norm=0.9, gae_lambda=0.99, \
        n_epochs=5, clip_range=0.3, batch_size=256, seed=123)

    # eval prior to learing to show that learning worked
    print("\nEALUATION BEFORE LEARNING")
    rews_before = evaluation(model, eval_eps)
    
    # =================== LEARNING ======================
    # ===================================================
    print("\nLEARNING")
    model.learn(total_timesteps=learning_steps)
    print("SAVING MODEL...")
    model.save("policy_simple")

    env.close()
    del model

    # =================== EVALUATION ====================
    # ===================================================
    print("\nEALUATION AFTER LEARNING")

    model = PPO.load("policy_simple")
    
    rews_after = evaluation(model, eval_eps)
    plotting_rews(rews_before, rews_after)
    print("done")

    # ===================== OBSERVE =====================
    # ===================================================
    print("OSERVE AGENT")
    for i in range(2):
        observe(model)
    print("finished observation")