from src.environments import pgg_parallel
from src.algos.Reinforce import Reinforce
from src.algos.DQN import DQN
import numpy as np
import torch
import wandb
import src.analysis.utils as U
from utils_train_reinforce_comm import eval1
from utils_train_reinforce import find_max_min

OBS_SIZE = 1
ACTION_SIZE = 2
DECAY_RATE = 0.999
RANDOM_BASELINE = False

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def setup_training_hyperparameters(args):
    print("inside setup training hyp")

    params = dict(
        num_game_iterations = 1,
        obs_size = OBS_SIZE,
        action_size = ACTION_SIZE,
        random_baseline = RANDOM_BASELINE
    )
    all_params = {**params, **vars(args)}
    print("all_params=",all_params)

    return all_params

def define_agents(config):
    agents = {}
    for idx in range(config.n_agents):
        if   (config.algorithm == "reinforce"):
            agents['agent_'+str(idx)] = Reinforce(config, idx)
        elif (config.algorithm == "dqn"):
            agents['agent_'+str(idx)] = DQN(config, idx)
    return agents

def train(args, repo_name):
    all_params = setup_training_hyperparameters(args)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)
    config = wandb.config
    print("config=", config)

    parallel_env = pgg_parallel.parallel_env(config)
    max_values = find_max_min(config, 4)
    idx_comm_agents = [j for j,val in enumerate(config.communicating_agents) if val==1]
    num_comm_agents = config.communicating_agents.count(1)

    agents = define_agents(config)

    #### TRAINING LOOP
    avg_norm_returns_train_list = []; avg_rew_time = []
    for epoch in range(config.n_epochs): 
        [agent.reset_batch() for _, agent in agents.items()]

        for i in range(config.batch_size):
            #print("\ni=", i)

            observations = parallel_env.reset()
            print("observations=", observations)
            
            [agent.reset_episode() for _, agent in agents.items()]

            done = False
            while not done:
                mf = parallel_env.current_multiplier
                print("mf=", mf.numpy()[0])

                messages = {}; actions = {}
                [agents[agent].set_observation(observations[agent]) for agent in parallel_env.agents]

                # speaking
                #print("\nspeaking")
                for agent in parallel_env.agents:
                    if (agents[agent].is_communicating):
                        messages[agent] = agents[agent].select_message(m_val=mf.numpy()[0]) # m value is given only to compute metrics

                # listening
                #print("\nlistening")
                if (num_comm_agents != 0):
                    message = torch.stack([v for _, v in messages.items()]).view(-1).to(device)
                    print("message=", message)
                    [agents[agent].get_message(message) for agent in parallel_env.agents if (agents[agent].is_listening)]

                # acting
                for agent in parallel_env.agents:
                    actions[agent] = agents[agent].select_action(m_val=mf.numpy()[0]) # m value is given only to compute metrics
                    print("actions=", actions)
                
                observations, rewards, done, _ = parallel_env.step(actions)
                rewards_norm = {key: value/max_values[float(parallel_env.current_multiplier[0])] for key, value in rewards.items()}
                #print("rewards=", rewards)
                #print("rewards_norm=", rewards_norm)

                for ag_idx, agent in agents.items():
                    
                    agent.buffer.rewards.append(rewards[ag_idx])
                    agent.return_episode_norm += rewards_norm[ag_idx]
                    agent.return_episode =+ rewards[ag_idx]

                    if (config.algorithm == "dqn"):
                        agent.append_to_replay(agent.state, agent.message_out, agent.state_to_act, agent.action, rewards[ag_idx], done)

                # break; if the episode is over
                if done:
                    break
            
        for ag_idx, agent in agents.items():
            if (agent.is_communicating):
                #print("\n",agent.buffer.actions_given_m)
                for m_val in config.mult_fact:
                    if (m_val in agent.sc_m):
                        agent.sc_m[m_val].append(U.calc_mutinfo(agent.buffer.actions_given_m[m_val], agent.buffer.messages_given_m[m_val], config.action_size, config.mex_size))
                    else:
                        agent.sc_m[m_val] = [U.calc_mutinfo(agent.buffer.actions_given_m[m_val], agent.buffer.messages_given_m[m_val], config.action_size, config.mex_size)]
                #print("\n\n==========>SC_m:",agent.sc_m)
                agent.sc.append(U.calc_mutinfo(agent.buffer.actions, agent.buffer.messages, config.action_size, config.mex_size))
            if (agent.is_listening):
                for i in idx_comm_agents:
                    agent.mutinfo_listening.append(U.calc_mutinfo(agent.buffer.actions, agents['agent_'+str(i)].buffer.messages, config.action_size, config.mex_size))


        # update agents
        #print("update")
        for ag_idx, agent in agents.items():
            agent.update()
        
        mex_distrib_given_m = {}; rewards_eval_m = {}; rewards_eval_norm_m = {}; actions_eval_m = {}
        for m in config.mult_fact:
            act_eval, mex_distrib, _, rewards_eval = eval1(config, parallel_env, agents, m, device, False)
            mex_distrib_given_m[m] = mex_distrib # distrib dei messaggei per ogni agente, calcolata con dato input
            rewards_eval_m[m] = rewards_eval
            rewards_eval_norm_m[m] = {key: value/max_values[m] for key, value in rewards_eval.items()}
            actions_eval_m[m] = act_eval
        #print("rewards_eval_m=",rewards_eval_m)
        avg_norm_return = np.mean([agent.return_episode_old_norm.numpy() for _, agent in agents.items()])
        avg_norm_returns_train_list.append(avg_norm_return)

        rew_values = [(np.sum([rewards_eval_m[m_val][ag_idx] for m_val in config.mult_fact])) for ag_idx, _ in agents.items()]
        
        avg_rew_time.append(np.mean(rew_values))
        #print(avg_rew_time)
        measure = np.mean(avg_rew_time[-10:])

        if (config.wandb_mode == "online" and epoch%10 == 0):
            for ag_idx, agent in agents.items():
                df_actions = {ag_idx+"actions_eval_m_"+str(i): actions_eval_m[i][ag_idx] for i in config.mult_fact}
                df_rew = {ag_idx+"rewards_eval_m"+str(i): rewards_eval_m[i][ag_idx] for i in config.mult_fact}
                df_rew_norm = {ag_idx+"rewards_eval_norm_m"+str(i): rewards_eval_norm_m[i][ag_idx] for i in config.mult_fact}
                df_agent = {**{
                    ag_idx+"_return_train_norm": agent.return_episode_old_norm,
                    ag_idx+"_return_train_"+str(mf[0]): agent.return_episode_old,
                    ag_idx+"gmm_means": agent.means,
                    ag_idx+"gmm_probabilities": agent.probs,
                    'epoch': epoch}, 
                    **df_actions, **df_rew, **df_rew_norm}
                
                if (config.communicating_agents[agent.idx] == 1.):
                    df_mex = {ag_idx+"messages_prob_distrib_m_"+str(i): mex_distrib_given_m[i][ag_idx] for i in config.mult_fact}
                    df_sc = {ag_idx+"sc": agent.sc_old[-1]}
                    df_sc_m = {ag_idx+"sc_given_m"+str(i): agent.sc_m[i][0] for i in config.mult_fact}
                    df_ent = {ag_idx+"_avg_mex_entropy": torch.mean(agent.entropy)}
                    df_agent = {**df_agent, **df_mex, **df_sc, **df_ent, **df_sc_m}
                if (config.listening_agents[agent.idx] == 1.):
                    df_listen = {ag_idx+"mutinfo_listening": agent.mutinfo_listening_old[-1]}
                    df_agent = {**df_agent, **df_listen}
                
                wandb.log(df_agent, step=epoch, commit=False)

            wandb.log({
                "epoch": epoch,
                "current_multiplier": mf,
                "avg_return_train": avg_norm_return,
                "avg_return_train_time": np.mean(avg_norm_returns_train_list[-10:]),
                "avg_rew_time": measure,
                "avg_loss": np.mean([agent.saved_losses[-1] for _, agent in agents.items()]),
                #"avg_loss_comm": np.mean([agent.saved_losses_comm[-1] for _, agent in agents.items()]),
                },
                step=epoch, commit=True)

        if (epoch%10 == 0):
            print("Epoch : {} \t Mult factor: {}  \t Measure: {} ".format(epoch, mf, measure))
    
    wandb.finish()
    

def training_function(args):

    name_gmm = "_noGmm"
    if (1 in args.gmm_):
        name_gmm = "_yesGmm"

    repo_name = str(args.n_agents) + "agents_" + "comm" + str(args.communicating_agents) + \
        "_list" + str(args.listening_agents) + name_gmm + "_unc" + str(args.uncertainties) + \
        "_mfact" + str(args.mult_fact) + "_algo_"+ args.algorithm +"_BEST"

    print("wandb: saving data in ", repo_name)
    train(args, repo_name)