# Project 1: Navigation
##### Author: Farhad Safaei
---
### Methodology
#### Deep-Q-Networks:
In this project a [dqn-agent](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p1_navigation/dqn_agent.py) is developed to accomplish the task with a value-based deep-reinforcement-learning approach. The underlying methodology is based on [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) by [DeepMind](https://deepmind.com/) that succeeded to play many Atari video games better than humans in 2015. The [Q-network model](https://github.com/fsafaei/DeepRL-Udacity/blob/master/p1_navigation/model.py), implemented in [PyTorch](https://pytorch.org/), consists of three fully connected layers.
#### Hyper parameters:
#
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
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