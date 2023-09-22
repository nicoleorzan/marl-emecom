from src.environments import pgg_parallel_v0
from src.algos.DQN_comm import DQN_comm
import numpy as np
import torch.nn.functional as F
import optuna
from optuna.trial import TrialState
import torch
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
from src.algos.normativeagent import NormativeAgent
from src.utils.social_norm import SocialNorm
from src.utils.utils import pick_agents_idxs, introspective_rewards
from src.experiments_my_game.params import setup_training_hyperparams

torch.autograd.set_detect_anomaly(True)


def define_agents(config):
    agents = {}
    for idx in range(config.n_agents):
        if (config.is_dummy[idx] == 0):
            agents['agent_'+str(idx)] = DQN_comm(config, idx) 
        else: 
            agents['agent_'+str(idx)] = NormativeAgent(config, idx)
    return agents

def interaction_loop(config, parallel_env, active_agents, active_agents_idxs, social_norm, _eval=False, mf_input=None):
    # By default this is a training loop

    if (_eval == True):
        observations = parallel_env.reset(mf_input)
    else:
        observations = parallel_env.reset()

    social_norm.reset_comm()

    #print("observations=",observations)
    rewards_dict = {}
    actions_dict = {}
    #if (_eval == False):
    #    print("active_agents=", active_agents_idxs)
    mex_states = {}; mex_next_states = {}
    for idx_agent, agent in active_agents.items():
        agent.other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
        if (agent.is_dummy == True): 
            mex_next_states[idx_agent] = torch.cat((agent.other.reputation, torch.Tensor([parallel_env.current_multiplier])))
        else: 
            if (config.reputation_enabled == 1):
                mex_next_states[idx_agent] = torch.cat((agent.other.reputation, observations[idx_agent]))
            else: 
                mex_next_states[idx_agent] = observations[idx_agent]

    # set mex state
    actions = {}; messages = {}; 
    mex_states = mex_next_states
    for idx_agent, agent in active_agents.items():
        agent.state_message = mex_states[idx_agent]
    #print("mex_states=", mex_states)

    # output mex
    for agent in parallel_env.active_agents:
        m = active_agents[agent].select_message(_eval)
        messages[agent] = m
    #print("messages=", messages)

    # set next state
    states = {}; next_states = {}
    for idx_agent, agent in active_agents.items():
        mex = F.one_hot(messages["agent_"+str(agent.other.idx)].long(), num_classes=config.mex_size)[0]
        next_states[idx_agent] = torch.cat((mex_states[idx_agent], mex))
    #print("next_states=", next_states)

    done = False
    for i in range(config.num_game_iterations):
        #print("\n\ni=",i)

        actions = {}; states = next_states
        for idx_agent, agent in active_agents.items():
            print(agent.reputation)
            agent.state_act = states[idx_agent]
        
        # action
        for agent in parallel_env.active_agents:
            a = active_agents[agent].select_action(_eval)
            actions[agent] = a
        #print("actions=", actions)

        # reward
        _, rewards, done, _ = parallel_env.step(actions)
        if (config.introspective == True):
            rewards = introspective_rewards(config, active_agents, parallel_env, rewards, actions)
        if (_eval==True):
            for ag_idx in active_agents_idxs:       
                if "agent_"+str(ag_idx) not in rewards_dict.keys():
                    rewards_dict["agent_"+str(ag_idx)] = [rewards["agent_"+str(ag_idx)]]
                    actions_dict["agent_"+str(ag_idx)] = [actions["agent_"+str(ag_idx)]]
                else:
                    rewards_dict["agent_"+str(ag_idx)].append(rewards["agent_"+str(ag_idx)])
                    actions_dict["agent_"+str(ag_idx)].append(actions["agent_"+str(ag_idx)])
        #print("rewards=", rewards)

        social_norm.save_comm(mex_states, messages, actions, active_agents_idxs)
        social_norm.change_rep_mex(active_agents, active_agents_idxs)

        # next mex state
        mex_next_states = {}
        for idx_agent, agent in active_agents.items():
            #other = active_agents["agent_"+str(list(set(active_agents_idxs) - set([agent.idx]))[0])]
            if (agent.is_dummy == True): 
                mex_next_states[idx_agent] = torch.cat((agent.other.reputation, torch.Tensor([parallel_env.current_multiplier])))
            else: 
                if (config.reputation_enabled == 1):
                    mex_next_states[idx_agent] = torch.cat((agent.other.reputation, observations[idx_agent]))
                else: 
                    mex_next_states[idx_agent] = observations[idx_agent]
        #print("mex_next_states=",mex_next_states)
        
        # memory buffer mex dqn
        if (_eval == False):
            # save iteration
            for ag_idx, agent in active_agents.items():
                if (agent.is_dummy == False):
                    #print("mex_states[ag_idx], messages[ag_idx], rewards[ag_idx], mex_next_states[ag_idx]=",mex_states[ag_idx], messages[ag_idx], rewards[ag_idx], mex_next_states[ag_idx])
                    agent.append_to_replay_mex(mex_states[ag_idx], messages[ag_idx], rewards[ag_idx], mex_next_states[ag_idx], done)
                    agent.return_episode =+ rewards[ag_idx]
        
        # set mex state
        messages = {}; mex_states = mex_next_states
        for idx_agent, agent in active_agents.items():
            agent.state_message = mex_states[idx_agent]

        # next mex
        for agent in parallel_env.active_agents:
            m = active_agents[agent].select_message(_eval)
            messages[agent] = m
        #print("next_messages=", messages)

        # set next state
        next_states = {}
        for idx_agent, agent in active_agents.items():
            mex = F.one_hot(messages["agent_"+str(agent.other.idx)].long(), num_classes=config.mex_size)[0]
            next_states[idx_agent] = torch.cat((mex_states[idx_agent], mex))
        #print("next_states=", next_states)

        # memory buffer action dqn
        if (_eval == False):
            # save iteration
            for ag_idx, agent in active_agents.items():
                if (agent.is_dummy == False):
                    agent.append_to_replay_state(states[ag_idx], actions[ag_idx], rewards[ag_idx], next_states[ag_idx], done)
                    agent.return_episode =+ rewards[ag_idx]

        if done:
            if (_eval == True):
                avg_reward = {}; avg_coop = {}
                for ag_idx, agent in active_agents.items():
                    avg_coop[ag_idx] = torch.mean(torch.stack(actions_dict[ag_idx]).float())
                    avg_reward[ag_idx] = torch.mean(torch.stack(rewards_dict[ag_idx]))
            break

    if (_eval == True):
        return avg_reward, avg_coop

def objective(args, repo_name, trial=None):

    all_params = setup_training_hyperparams(args, trial)
    wandb.init(project=repo_name, entity="nicoleorzan", config=all_params, mode=args.wandb_mode)
    config = wandb.config
    print("config=", config)

    # define envupdate_
    parallel_env = pgg_parallel_v0.parallel_env(config)

    # define agents
    agents = define_agents(config)

    # define social norm
    social_norm = SocialNorm(config, agents)
    
    #### TRAINING LOOP
    coop_agents_mf = {}; rew_agents_mf = {}
    for epoch in range(config.n_episodes):
        #print("\n==========>Epoch=", epoch)

        # pick a pair of agents
        active_agents_idxs = pick_agents_idxs(config)
        active_agents = {"agent_"+str(key): agents["agent_"+str(key)] for key, _ in zip(active_agents_idxs, agents)}

        [agent.reset() for _, agent in active_agents.items()]

        parallel_env.set_active_agents(active_agents_idxs)

        # TRAIN
        #print("\n\n=====================================>TRAIN")
        interaction_loop(config, parallel_env, active_agents, active_agents_idxs, social_norm, _eval=False)

        # update agents
        #print("UPDATE!!")
        losses = {}
        for ag_idx, agent in active_agents.items():
            losses[ag_idx] = agent.update(epoch)

        # evaluation step
        #print("\n\n===================================>EVAL")
        for mf_input in config.mult_fact:
            avg_rew, avg_coop = interaction_loop(config, parallel_env, active_agents, active_agents_idxs, social_norm, True, mf_input)
            avg_coop_tot = torch.mean(torch.stack([cop_val for _, cop_val in avg_coop.items()]))

            avg_rep = np.mean([agent.reputation[0] for _, agent in agents.items() if (agent.is_dummy == False)])
            measure = avg_rep
            coop_agents_mf[mf_input] = avg_coop
            rew_agents_mf[mf_input] = avg_rew

        dff_coop_per_mf = dict(("avg_coop_mf"+str(mf), torch.mean(torch.stack([ag_coop for _, ag_coop in coop_agents_mf[mf].items()]))) for mf in config.mult_fact)
        dff_rew_per_mf = dict(("avg_rew_mf"+str(mf), torch.mean(torch.stack([ag_coop for _, ag_coop in rew_agents_mf[mf].items()]))) for mf in config.mult_fact)

        """Q = {}
        if (config.reputation_enabled == 0): 
            for ag_idx, agent in agents.items():
                if (agent.is_dummy == False):
                    if ag_idx not in Q:
                        Q[ag_idx] = torch.zeros(len(config.mult_fact), 2) # mult fact, poss actions
                        possible_states = torch.stack([torch.Tensor([mf]) for _, mf in enumerate(config.mult_fact)])
                        Q[ag_idx] = agent.get_action_values(possible_states).detach()
            stacked = torch.stack([val for ag_idx, val in Q.items()])
            avg_distrib = torch.mean(stacked, dim=0)
        
        else:
            possible_reputations = [0., 1.]
            for ag_idx, agent in agents.items():
                #print("\nag_idx=", ag_idx)
                if (agent.is_dummy == False):
                    if ag_idx not in Q:
                        Q[ag_idx] = torch.zeros(2, len(config.mult_fact), 2) # mult fact, poss rep, poss actions
                        #print("prob[ag_idx]=", prob[ag_idx])
                    for rep in possible_reputations:
                        possible_states = torch.stack([torch.Tensor([i, rep]) for i, _ in enumerate(config.mult_fact)])
                        #print("agent.get_action_values(possible_states).detach()=",agent.get_action_values(possible_states).detach())
                        Q[ag_idx][int(rep),:,:] = agent.get_action_values(possible_states).detach()
            #print("prob=", prob)
            stacked = torch.stack([val for _, val in Q.items()])
            #print("stacked=", stacked, stacked.shape)
            avg_distrib = torch.mean(stacked, dim=0)
        #print("avg_distrib=", avg_distrib)
        """

        if (config.optuna_):
            trial.report(measure, epoch)
            
            if trial.should_prune():
                print("is time to pruneee")
                wandb.finish()
                raise optuna.exceptions.TrialPruned()

        if (config.wandb_mode == "online" and float(epoch)%30. == 0.):
            for ag_idx, agent in active_agents.items():
                if (agent.is_dummy == False):
                    df_avg_coop = dict((ag_idx+"avg_coop_mf"+str(mf), coop_agents_mf[mf_input][ag_idx]) for mf in config.mult_fact)
                    df_avg_rew = {ag_idx+"avg_rew": avg_rew[ag_idx]}
                    #df_Q = {ag_idx+"Q[0,0]": Q[ag_idx][0,0], ag_idx+"Q[0,1]": Q[ag_idx][0,1], ag_idx+"Q[1,0]": Q[ag_idx][1,0], ag_idx+"Q[1,1]": Q[ag_idx][1,1]}
                    df_loss = {ag_idx+"loss": losses[ag_idx]}
                    df_agent = {**{
                        ag_idx+"_reputation": agent.reputation,
                        ag_idx+"epsilon": active_agents[str(ag_idx)].epsilon,
                        'epoch': epoch}, 
                        **df_avg_coop, **df_avg_rew, **df_loss
                        }
                else:
                    df_avg_coop = {ag_idx+"dummy_avg_coop": avg_coop[ag_idx]}
                    df_avg_rew = {ag_idx+"dummy_avg_rew": avg_rew[ag_idx]}
                    df_agent = {**{
                        ag_idx+"dummy_reputation": agent.reputation,
                        'epoch': epoch}, 
                        **df_avg_coop, **df_avg_rew
                        }
                if ('df_agent' in locals() ):
                    wandb.log(df_agent, step=epoch, commit=False)
            """if (config.reputation_enabled == 0): 
                dff_Q00 = {"avg_Q["+str(mf)+",0]": avg_distrib[imf,0] for imf, mf in enumerate(config.mult_fact) }
                dff_Q01 = {"avg_Q["+str(mf)+",1]": avg_distrib[imf,1] for imf, mf in enumerate(config.mult_fact) }
                dff_Q10 = dict(())
                dff_Q11 = dict(())
            else:
                dff_Q00 = {"avg_Q[0,"+str(mf)+",0]": avg_distrib[0,imf,0] for imf, mf in enumerate(config.mult_fact) }
                dff_Q01 = {"avg_Q[0,"+str(mf)+",1]": avg_distrib[0,imf,1] for imf, mf in enumerate(config.mult_fact) }
                dff_Q10 = {"avg_Q[1,"+str(mf)+",0]": avg_distrib[1,imf,0] for imf, mf in enumerate(config.mult_fact) }
                dff_Q11 = {"avg_Q[1,"+str(mf)+",1]": avg_distrib[1,imf,1] for imf, mf in enumerate(config.mult_fact) }
            """
            dff = {
                "epoch": epoch,
                "avg_rep": avg_rep,
                "avg_rew_time": measure,
                "avg_coop_from_agents": avg_coop_tot,
                "weighted_average_coop": torch.mean(torch.stack([avg_i for _, avg_i in avg_rew.items()])) # only on the agents that played, of course
                }
            if (config.non_dummy_idxs != []): 
                dff = {**dff, **dff_coop_per_mf, **dff_rew_per_mf}#, **dff_Q00, **dff_Q01, **dff_Q10, **dff_Q11}
            wandb.log(dff,
                step=epoch, commit=True)

        if (epoch%10 == 0):
            print("\nEpoch : {} \t Measure: {} ".format(epoch, measure))
            print("avg_rew=", {ag_idx:avg_i for ag_idx, avg_i in avg_rew.items()})
            #print("avg_coop_tot=", avg_coop_tot)
            print("coop_agents_mf=",coop_agents_mf)
            print("dff_coop_per_mf=",dff_coop_per_mf)
    
    wandb.finish()
    return measure


def train_dqn_comm(args):

    unc_string = "no_unc_"
    if (args.uncertainties.count(0.) != args.n_agents):
        unc_string = "unc_"

    repo_name = "EPGG_"+ str(args.n_agents) + "agents_" + \
        unc_string + args.algorithm + "_mf" + str(args.mult_fact) + \
        "_rep" + str(args.reputation_enabled) + "_COMM"
    
    if (args.addition != ""):
        repo_name += "_"+ str(args.addition)
    print("repo_name=", repo_name)

    # If optuna, then optimize or get best params. Else use given params
    if (args.optuna_ == 0):
        objective(args, repo_name)
    else:
        func = lambda trial: objective(args, repo_name, trial)

        # sql not optimized for paralel sync
        journal_name = repo_name + "_binary_"+str(args.binary_reputation)

        storage = JournalStorage(JournalFileStorage("optuna-journal"+journal_name+".log"))

        study = optuna.create_study(
            study_name=repo_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
            n_startup_trials=0, n_warmup_steps=40, interval_steps=3
            )
        )

        if (args.optimize):
            study.optimize(func, n_trials=100, timeout=1000)

        else:
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            
            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")

            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            
            print("Running with best params:")
            objective(args, repo_name+"_BEST", study.best_trial)