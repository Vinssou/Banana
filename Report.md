# Project 1: Navigation
## Introduction
For this project, I trained an agent to navigate and collect yellow bananas, and avoid the blue bananas in a large, square world.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
## Overview
The agent that has been implemented is a Dueling Double Deep Q Network (D3QN), using a prioritized replay buffer.
First we will see the model architecture, then I will present to you how the neural network is used, and finally we will explore the replay buffer.
## Model Architecture
### Dueling
The model is composed of two parallel layers. Those two layers have the same input which is the state. Those two layers are composed of fully connected layers, drop outs layers. Each fully connected layer is terminated by an activated function. We use the Relu activation function.
The first parallel layers are used to predict the state value, which is a single scalar. The second is used to predict the advantage values. Its output dimension size is the action space size. The two outputs are added and the average of the advantage is subtracted.
[Reference: Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
### Drop out
Some dropout layers are used to maximize the training of all neurons. During training some neurons will be randomly disabled, which will favor the training of other neurons.
## Double Network
### Target Network
In the traditional DQN we use only one neural network to compute the target using the next state. Here we use an additional neural network (the target) to compute the target. This network has the same architecture as the other (the local) network that we are training. The target network is updated regularly from the parameters of the local network. There are two ways to update the parameter, by applying a soft update or a hard update.
[Reference: Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
An hyper parameter has been added to be able to select the type of update to select.
### Hard Update
Every number of frames, chosen by a hyper parameter the weight from the local network are copied over the target network.
### Soft Update
We update the target network from the local regularly using an interpolation with local weight.
## Training
### Batching data
Iterating through all states, actions, rewards, next_states and dones to train the network can be CPU time consuming. By batching all data together to train our network speed up the training process.
### Exploration versus Exploitation
To train our network we need to have a good balance between exploring the environment, meaning picking an action randomly and exploiting our network to predict the right action.
We use a linear interpolation starting from exploration to exploitation.
[Reference: Deep Exploration via Bootstrapped DQN](https://papers.nips.cc/paper/2016/file/8d8818c8e140c64c743113f563cf750f-Paper.pdf)
## Prioritized Replay Buffer
Every experience is stored in a memory to be used to train the model. Each entry is an experience defined by a state, an action, the reward, the next state and a boolean to know if it is the last step of the episode.
An approach that works is to store a constant number of experiences and pick randomly a batch of experiences.
### Finding the experiences that matter
To increase the convergence of the neural network we use a probability distribution to select more often the experience that matters most. We are using the error defined by the difference of the target qvalue and the current qvalue to prioritize the experience.
### Optimization with a Sum tree
Using a sum tree allows us to find the experience that has the biggest error. The sum is a binary tree where the parent node is equal to the sum of its direct children.
We use all the leaves to store the experiences. Each leaf stores its priority aside its experience (state, action, rewards, next state, done). When adding an experience in the tree, the priority is updated according to the experience error. Then we update all the parents nodes' values until we reach the root. By definition the root contains the sum of all priority of all the leaves of the tree.
The tree has been implemented in an optimal way, using continuous memory data structure and iterative algorithms to navigate inside the tree. And to maximise it, it uses internally a buffer where the size is a power of 2, to use all leaves.
### Adapting the weight
To avoid a big impact on the experiences that will be selected more often, we use a formula to compute the weight. The weights are considered when computing the loss to stabilize the algorithm.
[Reference: PRIORITIZED EXPERIENCE REPLAY ](https://arxiv.org/pdf/1511.05952.pdf)
## Results and Hyperparameters
### Hyperparameters
```
Hyper Parameters:
gamma: 0.99
tau_soft: 0.001
tau_hard: 100
use_soft_update: True
learning_rate: 0.0005
learning_step_count: 4
eps_min: 0.03
eps_deacy: 0.99995
buffer_size: 10000
batch_size: 64
use_prioritized_buffer: True
memory_epsilon: 0.01
memory_alpha: 0.4
memory_beta: 0.4
capacity:  16384
```
 
Note `capacity` is computed from the buffer_size.
 
The D3QN implemented has a lot of hyper parameter, we can have a priority replay buffer or just a simple replay buffer, we can have a soft update target or a hard update, each of them have different parameters, the learning rate, number of hidden layers, and their number of neurons etc.. The amount of parameters is huge, and their combination makes the algorithm hard to tune.
Developing a tool that tries the algorithm with a various set of parameters could help to tune the algorithm.
 
The epsilon decay was tuned to reach the exploitation mode quite quickly and keep taking 3% of action randomly. A simple linear function has been used here, but a logarithmic function could have helped to decrease the randomness slower at the end.
The soft update seems to bring better results than the hard update, it is hard to compare them because they also have their own parameters that can be tuned separately.

The dueling architecture helped to converge faster.
 
![alt text](https://github.com/Vinssou/Banana/blob/master/score.png)
 
It was hard to take advantage of the replay buffer.
I think the environment is quite simple, and so the prioritiized replay buffer couldn't take any advantages.
 
![alt text](https://github.com/Vinssou/Banana/blob/master/score_prioritized.png)
 
### Test
The output of the last cell ran with the load model
```
0 Eps: 0.03 Score: 18.0
1 Eps: 0.03 Score: 17.0
2 Eps: 0.03 Score: 17.0
3 Eps: 0.03 Score: 19.0
4 Eps: 0.03 Score: 15.0
5 Eps: 0.03 Score: 18.0
6 Eps: 0.03 Score: 13.0
7 Eps: 0.03 Score: 18.0
8 Eps: 0.03 Score: 23.0
9 Eps: 0.03 Score: 8.0
```
 
## Conclusion
The prioritized replay buffer didn't bring any speed up in the convergence. I should try to solve other problems, to see if I could get the gain that was experimented in various papers.
 
I should support GPU for my code to be able to run more experiments and so be able to tune it more rigorously.
 
 

