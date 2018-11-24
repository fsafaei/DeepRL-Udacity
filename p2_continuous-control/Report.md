# Project 2: Continues Control
##### Author: Farhad Safaei
---
### Methodology
#### Deep-Deterministic-Policy-Gradient(DDPG):
In this project a [ddpg-agent](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p2_continuous-control/ddpg_agent.py) is developed to accomplish the task with an policy-based deep-reinforcement-learning approach in a continuous space. The underlying methodology is based on the research [Continuous control with deep reinforcement learning ](https://arxiv.org/abs/1509.02971) by [DeepMind](https://deepmind.com/). In this method, the agent uses an Actor-Critic approach which is employed to reduce the variance problem in policy-based methods. However, the DDPG can be more considered as a proximate-DQN method instead of an actual Actor-Critic method. The reason is that the critic in DDPG is used to approximate the maximizer over the Q-values of the next states, not as a base-line for learning. The [model](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p2_continuous-control/model.py), implemented in [PyTorch](https://pytorch.org/), consists of the Actor, Critic, and an NN with three fully connected layers. 
The environment simulates a space with 20 robot arms. To improve the learning process, a Multi-Agent learning technique is used, so that the 20 agents in the environment can learn from each other’s experiments. (inspired from the model from [Henry Chan](https://github.com/kinwo/deeprl-continuous-control)).

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
[Here](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p2_continuous-control/Continuous_Control.ipynb) is shown how the agents interact with the environment and how they learn to perform better. The environment is supposed to be solved when the agents achieve a minimum average score of 30.0. The average scores are shown during the agents training for 150 episodes. The environment in solved in 29 episodes by achieving the average score of 30.20.

![Training Process](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p2_continuous-control/Continuous%20Control.png)


### Future works:
# Project 2: Continues Control
##### Author: Farhad Safaei
---
### Methodology
#### Deep-Deterministic-Policy-Gradient(DDPG):
In this project a [ddpg-agent](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p2_continuous-control/ddpg_agent.py) is developed to accomplish the task with an policy-based deep-reinforcement-learning approach in a continuous space. The underlying methodology is based on the research [Continuous control with deep reinforcement learning ](https://arxiv.org/abs/1509.02971) by [DeepMind](https://deepmind.com/). In this method, the agent uses an Actor-Critic approach which is employed to reduce the variance problem in policy-based methods. However, the DDPG can be more considered as a proximate-DQN method instead of an actual Actor-Critic method. The reason is that the critic in DDPG is used to approximate the maximizer over the Q-values of the next states, not as a base-line for learning. The [model](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p2_continuous-control/model.py), implemented in [PyTorch](https://pytorch.org/), consists of the Actor, Critic, and an NN with three fully connected layers. 
The environment simulates a space with 20 robot arms. To improve the learning process, a Multi-Agent learning technique is used, so that the 20 agents in the environment can learn from each other’s experiments. (inspired from the model from [Henry Chan](https://github.com/kinwo/deeprl-continuous-control)).

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
[Here](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p2_continuous-control/Continuous_Control.ipynb) is shown how the agents interact with the environment and how they learn to perform better. The environment is supposed to be solved when the agents achieve a minimum average score of 30.0. The average scores are shown during the agents training for 150 episodes. The environment in solved in 29 episodes by achieving the average score of 30.20.

![Training Process](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p2_continuous-control/Continuous%20Control.png)


### Future work:
- [Combined Reinforcement Learning via Abstract Representations](https://arxiv.org/pdf/1809.04506.pdf): tries to improve the stability by combining model-free and model-based approaches through a shared low-dimensional encoding of the environment.
- Fine tuning of the parameters might improves the convergence speed
- [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748) can be also evaluated over the problem.
