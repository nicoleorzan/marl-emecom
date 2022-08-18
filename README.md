# Multi-Agent Reinforcement Learning with Emergent Communication for Mixed-Motive Environments

## Repo content

In this repository I implemented two pettingzoo environments for two different versions of the Public Good Game, and PPO reinforcement learning agents that can solve the game with and wthout communication. 

Under the "src" folder you can find:
* in the "environments" the implementations of the two environments, with versions v0 and v1.
* in the "nets" folder you can find an implementation of the different kind of architectures for the agents playing the game
* in the "algos" folder you can find the implementations of the learning algorithm, which are different versions of the PPO (proximal olicy Optimization) algorithm. The different versions allow to train different kind of networks.
* in the "analysis" folder you can find the implementation of different function to analyse the bahavior of the RL agents during and after training
* in the "experiments_pgg_v0" and "experiments_pgg_v1" fodlers you can respectively find the training mains for the different kind of reinfrocement learning agents that solve the two games.

Different kind of agents with different abilities have been implemented: they can or cannot communicate, and agents that can or cannot memorize older actions and messages. All of them are trained using the Proximal Policy Optimization algorithm.


## How to use:

---
If in server, remember to load the correct python, as for example:

```bash
module load Python/3.8.6-GCCcore-10.2.0
```

Then create a virtual environment as follows:

```bash
python3 -m venv env

```
and activate it:

```bash
source env/bin/activate

```
Don't forget to update pip:


```bash
pip install --upgrade pip

```

Install the setup file as follows (the -e options allows to change the code while using it):

```bash
pip install -e .
```

Install the required packages:

```bash
pip install -r requirements.txt
```


