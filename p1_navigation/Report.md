# Project 1: Navigation
##### Author: Farhad Safaei
---
### Methodology
##### Deep-Q-Netwoks:
In this project a [dqn-agent](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p1_navigation/dqn_agent.py) is developed to accomplish the task with a value-based deep-reinforcement-learning approach. The underlaying methodology is based on [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) by [DeepMind](https://deepmind.com/) that succeeded to play many Atari video games better than humans in 2015. The [Q-network model](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p1_navigation/model.py), implemented in [PyTorch](https://pytorch.org/), consists of three fully connected layers.
##### Hypermarameters:
#
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
### Results
[Here](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p1_navigation/Navigation.ipynb) is shown how the agent interacts with the environment and how it learns to perform better. The environment is supposed to be solved when the agent achieves a minimum average score of 13.0. The average scores are shwon during the agent training for 1000 episodes. The agent succeeded to solve the environment in 471 episodes by achieveing the average score of 13.03.

![Training Process]()

Finally, the smart agent plays the game for 20 rounds and achieves an average score of 15.45.
