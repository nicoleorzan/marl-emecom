import numpy as np
import pandas as pd 
import wandb

def eval(config, parallel_env, agents_dict, m, _print=True):
    observations = parallel_env.reset(None, None, m)

    if (_print == True):
        print("* Eval ===> Mult factor=", m)
        print("obs=", observations)

    done = False
    while not done:

        actions = {agent: agents_dict[agent].select_action(observations[agent], True) for agent in parallel_env.agents}
        out = {agent: agents_dict[agent].get_distribution(observations[agent]) for agent in parallel_env.agents}

        if (_print == True):
            print("actions=", actions)
            print("distributions", out)
        observations, _, done, _ = parallel_env.step(actions)

    return np.mean([actions["agent_"+str(idx)] for idx in range(config.n_agents)])


def save_stuff(config, parallel_env, agents_dict, df, m_min, m_max, avg_coop_time, experiment, ep_in):
    coop_min = eval(config, parallel_env, agents_dict, m_min, False)
    coop_max = eval(config, parallel_env, agents_dict, m_max, False)
    performance_metric = coop_max+(1.-coop_min)
    #print("[agent.tmp_actions_old for _, agent in agents_dict.items()]=",[agent.tmp_actions_old for _, agent in agents_dict.items()])
    avg_coop_time.append(np.mean([agent.tmp_actions_old for _, agent in agents_dict.items()]))
    if (config.wandb_mode == "online"):
        for ag_idx, agent in agents_dict.items():
            wandb.log({ag_idx+"_return_train": agent.return_episode_old.numpy()}, step=ep_in)
            wandb.log({ag_idx+"_coop_level_train": np.mean(agent.tmp_actions_old)}, step=ep_in)
        wandb.log({"episode": ep_in}, step=ep_in)
        wandb.log({"avg_return_train": np.mean([agent.return_episode_old.numpy() for _, agent in agents_dict.items()])}, step=ep_in)
        wandb.log({"avg_coop_train": avg_coop_time[-1]}, step=ep_in)
        wandb.log({"avg_coop_time_train": np.mean(avg_coop_time[-10:])}, step=ep_in)
        
        # insert some evaluation for m_min and m_max
        wandb.log({"mult_"+str(m_min)+"_coop": coop_min}, step=ep_in)
        wandb.log({"mult_"+str(m_max)+"_coop": coop_max}, step=ep_in)
        wandb.log({"performance_mult_("+str(m_min)+","+str(m_max)+")": performance_metric}, step=ep_in)

    if (config.save_data == True):
        df_ret = {"ret_ag"+str(i)+"_train": agents_dict["agent_"+str(i)].return_episode_old.numpy() for i in range(config.n_agents)}
        df_coop = {"coop_ag"+str(i)+"_train": np.mean(agents_dict["agent_"+str(i)].tmp_actions_old) for i in range(config.n_agents)}
        df_avg_coop = {"avg_coop_train": avg_coop_time[-1]}
        df_avg_coop_time = {"avg_coop_time_train": np.mean(avg_coop_time[-10:])}
        
        df_performance = {"coop_m"+str(m_min): coop_min, "coop_m"+str(m_max): coop_max, "performance_metric": performance_metric}
        df_dict = {**{'experiment': experiment, 'episode': ep_in}, **df_ret, **df_coop, **df_avg_coop, **df_avg_coop_time, **df_performance}
        df = pd.concat([df, pd.DataFrame.from_records([df_dict])])