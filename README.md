# MARL-EmeCom: Multi-Agent RL with Emergent Communication in Mixed-Motive Settings

**Paper:** *Learning in Public Goods Games: The Effects of Uncertainty and Communication on Cooperation* (Orzan et al. 2025)  
[📄 Read on SpringerLink](https://link.springer.com/article/10.1007/s00521-024-10530-6)

---

## Overview


This project studies **emergent communication in multi-agent reinforcement learning (MARL)** under **mixed incentives** and **uncertainty**.  
We extend the Public Goods Game into an **Extended Public Goods Game (EPGG)**, spanning cooperative, mixed, and competitive settings. The code reproduces the experiments from [our paper](https://link.springer.com/article/10.1007/s00521-024-10530-6).

**Key findings:**  
- Communication supports cooperation under **symmetric uncertainty**.  
- Under **asymmetric uncertainty**, agents may exploit communication.  
- Agents trained across multiple incentive environments learn richer strategies that **generalize** better to unseen settings.

## Project Features & Repository Structure

- **Environments**: Extended Public Goods Game (EPGG) with cooperative/mixed/competitive incentives.    
- **Uncertainty**: noisy observations of the incentive factor (Gaussian).
- **Emergent communication**: discrete (“cheap talk”) messages before acting.  
- **Algorithms**:  
  - **REINFORCE** (policy gradient)
  - **DQN** (deep Q-learning)  
- **Uncertainty modelling**: agents can optionally maintain a **Gaussian Mixture Model (GMM)** to infer hidden incentive structure.

**Code structure**:
- [`/envs`](envs): Extended Public Goods Game (EPGG) environments.
- [`/agents`](agents): Implementations of REINFORCE and DQN agents.
- [`/comm`](comm): Modules for emergent communication channels.
- [`/analysis`](analysis): Scripts for metrics (mutual information, speaker consistency, coordination).
- [`/experiments`](experiments): Configurations and training scripts to reproduce paper results.


## Getting Started / Implementation

### 1. Clone & Dependencies

```bash
git clone https://github.com/nicoleorzan/marl-emecom.git
cd marl-emecom
pip install -r requirements.txt
```
(The use of a virtual environment is suggested)

### 2. Training Agents

You can train agents either:
- Without communication
- With communication (a subset of agents sends discrete messages before action)

Example usage:

The launcher sets parameters inside `src/experiments_pgg_v0/caller_given_params.py`; you can edit them there, or pass them as input:
```
python caller_given_params.py --n_agents 2 --mult_fact 0.5 1.5 2.5 --uncertainties 0 0 --communicating_agents 1 1 --listening_agents 1 1 --gmm_ 0 --algorithm reinforce
```

Base run:
```
python src/experiments_pgg_v0/caller_given_params.py
```
