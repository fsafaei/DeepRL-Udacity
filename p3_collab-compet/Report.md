# Project 3: Collaboration and Competition
##### Author: Farhad Safaei
---
### Methodology
#### Multi-Agent Deep-Deterministic-Policy-Gradient(MADDPG):
In this project a [multi ddpg-agent]( https://github.com/fsafaei/DeepRL-Udacity/blob/master/p3_collab-compet/ddpg_agent.py) is developed to accomplish the task with a multi-agent policy-based deep-reinforcement-learning approach in a continuous space. The underlying methodology is based on the research [Continuous control with deep reinforcement learning ](https://arxiv.org/abs/1509.02971) by [DeepMind](https://deepmind.com/). In this method, the agent uses an Actor-Critic approach which is employed to reduce the variance problem in policy-based methods. However, the DDPG can be more considered as a proximate-DQN method instead of an actual Actor-Critic method. The reason is that the critic in DDPG is used to approximate the maximizer over the Q-values of the next states, not as a base-line for learning. The [model](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p3_collab-compet/model.py), implemented in [PyTorch](https://pytorch.org/), consists of the Actor, Critic, and an NN with three fully connected layers. Some minor changes are applied to the original model from Udacity, including the change in the NN layer sizes and adding dropout to avoid overfitting. Modifications in the model and hyper parameters are inspired from [Henry Chan]( https://github.com/kinwo/deeprl-tennis-competition)
The environment simulates two ping-pong players who can learn from each other’s experiments.
#### Hyper parameters:
#
    GAMMA = 0.99            # discount factor
	TAU = 2e-1              # for soft update of target parameters
	LR_ACTOR = 1e-4         # learning rate of the actor 
	LR_CRITIC = 3e-4        # learning rate of the critic
	WEIGHT_DECAY = 0.     	# L2 weight decay
	BATCH_SIZE = 512        # minibatch size
	BUFFER_SIZE = int(1e5)  # replay buffer size
### Results
[Here]( https://github.com/fsafaei/DeepRL-Udacity/blob/master/p3_collab-compet/Tennis.ipynb) is shown how the agents interact with the environment and how they learn to perform better. The environment is supposed to be solved when the agents achieve a minimum average score of 0.5. The average scores are shown during the agents training for 2000 episodes. The environment in solved in 204 episodes by achieving the average score of 0.51.

![Training Process]( https://github.com/fsafaei/DeepRL-Udacity/blob/master/p3_collab-compet/collab_compt.png)


### Future works:

- Further improvements on the DDPG method such as [Combined Reinforcement Learning via Abstract Representations](https://arxiv.org/pdf/1809.04506.pdf) and [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)
- Applying [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments]( https://arxiv.org/pdf/1706.02275.pdf)
- Hyper parameter tuning 
