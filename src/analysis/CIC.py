import src.analysis.utils as U
import math
import numpy as np
import torch

def calc_cic(p_a_given_do_c, p_c, n_comm, n_acts):
    # Calculate the one-step causal influence of communication, i.e. the mutual information using p(a | do(c))
    p_ac = p_a_given_do_c * np.expand_dims(p_c, axis=1)  # calculate joint probability p(a, c)
    p_ac /= np.sum(p_ac)  # re-normalize
    p_a = np.mean(p_ac, axis=0)  # compute p(a) by marginalizing over c

    # Calculate mutual information
    cic = 0
    for c in range(n_comm):
        for a in range(n_acts):
            if p_ac[c][a] > 0:
                cic += p_ac[c][a] * math.log(p_ac[c][a] / (p_c[c] * p_a[a]))
    return cic

def get_p_a_given_do_c(agents, env, self=False):
    # Calculates p(a | do(c)) for both agents, i.e. the probability distribution over agent 1's actions given that
    # we intervene at agent 2 to send message c (and vice-versa)
    # If self = True, calculates p(a | do(c)) if we intervene at agent 1 to send message c, i.e. the effect of
    # agent 1's message on its own action (and similarly for agent 2)

    # Cache payoff matrices to ensure they are kept consistent
    #payoff_a = env.payoff_mat_a
    #payoff_b = env.payoff_mat_b
    p_a_given_do_c = [np.zeros((env.n_comm, env.n_acts)), np.zeros((env.n_comm, env.n_acts))]

    # For both agents
    for ag in range(2):
        # Iterated over this agent's possible messages
        for i in range(env.n_comm):
            _ = env.reset()  # get rid of any existing messages in the observation
            env.payoff_mat_a = payoff_a  # restore payoffs undone by .reset()
            env.payoff_mat_b = payoff_b
            ob_c, _ = env.step_c_single(i, ag)  # intervene in environment with message i
            if self:
                # Calculate p(a|do(c)) of same agent
                logits_c, logits_a, v = agents[ag].forward(torch.Tensor(ob_c[ag]))
            else:
                # Calculate p(a|do(c)) of other agent
                logits_c, logits_a, v = agents[1 - ag].forward(torch.Tensor(ob_c[1 - ag]))

            # Convert logits to probability distribution
            probs_a = F.softmax(logits_a, dim=0)
            p_a_given_do_c[ag][i, :] = probs_a.data.numpy()

    return p_a_given_do_c