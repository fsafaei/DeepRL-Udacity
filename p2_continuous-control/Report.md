# Project 2: Continuos Control
##### Author: Farhad Safaei
---
### Methodology
#### Deep-Deterministic-Policy-Gradient(DDPG):
In this project a [ddpg-agent](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p2_continuous-control/ddpg_agent.py) is developed to accomplish the task with an policy-based deep-reinforcement-learning approach in a continuous space. The underlying methodology is based on the research [Continuous control with deep reinforcement learning ](https://arxiv.org/abs/1509.02971) by [DeepMind](https://deepmind.com/). In this method, the agent uses an Actor-Critic approach which is employed to reduce the variance problem in policy-based methods. However, the DDPG can be more considered as a proximate-DQN method instead of an actual Actor-Critic method. The reason is that the critic in DDPG is used to approximate the maximizer over the Q-values of the next states, not as a base-line for learning. The [model](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p2_continuous-control/model.py), implemented in [PyTorch](https://pytorch.org/), consists of the Actor, Critic, and an NN with three fully connected layers. 
The environment simulates a space with 20 robot arams. To improve the learning process, a Multi-Agent learning technique is used, so that the 20 agents in the environment can learn from each other'S experiments. (inspired from the model from[Henry Chan](https://github.com/kinwo/deeprl-continuous-control)).

#### Hyper parameters:
#
    GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR_ACTOR = 1e-3         # learning rate of the actor 
	LR_CRITIC = 1e-3        # learning rate of the critic
	WEIGHT_DECAY = 0.     	# L2 weight decay
	BATCH_SIZE = 1024       # minibatch size
	BUFFER_SIZE = int(1e6)  # replay buffer size
### Results
[Here](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p1_navigation/Navigation.ipynb) is shown how the agent interacts with the environment and how it learns to perform better. The environment is supposed to be solved when the agent achieves a minimum average score of 13.0. The average scores are shown during the agent training for 1000 episodes. The agent succeeded to solve the environment in 471 episodes by achieving the average score of 13.03.

![Training Process](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p1_navigation/navigation.png)

Finally, the smart agent plays the game for 20 rounds and achieves an average score of 15.45.

### Future works:
There have been a number of researches after the announcement of the DQNs, each of which addresses an issue in the original method. Six main ones are:
- [Double DQN](https://arxiv.org/abs/1509.06461): tackles with the overestimation of action values in DQN method.
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): improves the learning process by prioritizing the more important samples and choosing them with a higher probability, unlike the uniform sampling in DQN.
- [Dueling DQN](https://arxiv.org/abs/1511.06581): helps with the estimation of the value for each state, regardless of the corresponding action.
- Multi-Step Bootstrap Targets
- Distributional DQN
- Noisy DQN
An integration of all these six methods with the original DQN ,called
[Rainbow](https://arxiv.org/abs/1710.02298), was tested by researchers at Google DeepMind, recently. It succeeded to outperform all of the individual improvements and records the highest score on Atari 2600 games.