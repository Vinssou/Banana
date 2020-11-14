

# Project 1: Navigation

## Introduction

For this project, you will train an agent to navigate and collect yellow bananas, and avoiding the blue bananas in a large, square world. 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


## Overview

The agent that has been implemented is a Dueling Double Deep Q Network (D3QN), using a prioritized replay buffer.

First we will see the model architecture, then I will present you how the neural network is used, and finally we will explore the replay buffer.


## Model Architecture

### Dueling
The model is composed of two parrallel layers. Those two layers have the same input which is the state. Those two layers are composed the fully connected layers, drop outs layers. Each fully connected layers are terminated by an activated function. we use the Relu activated function.

The first parralel layers are used to predict the state value, which is a single scalar. The second is used to predict the advantage values. Its output dimension size is the action space size. The two outputs are added and the average of the advantage is substracted.

[Reference:](https://arxiv.org/pdf/1511.06581.pdf)

### Drop out

Some dropout layers are used to maximize the training of all neurones. During training some neurons will be randomly disabled, which will favorite the traiing of other neurons.

## Double Network 

### Target Network
In the traditional DQN we use only one neural network to compute the target using the next state. Here we use an additional neural network (the target) to compute the target. This network has the same architecture than the other (the local) network that we are training. The target netwok is updated regurlaly from the parameter of the local network. There is two ways to update the parameter, by applying a soft update or a hard update.
[Reference:](https://arxiv.org/pdf/1509.06461.pdf)

An hyper parameter has been added to be able to select the type of update to select.

### Hard Update
Every number of frame, choosen by a hyper parameter the weight from the local network are copied over the target network.

### Soft Update
We update the target network from the local regurlaly using an interpolation with local weight.

## Training

### Batching data
Iterating through all states, actions, rewards, next_states and dones to train the network can be CPU time consuming. By batching all data together to train our network speed up the training process.

### Exploration versus Exploitation
To train our network we need to have a good balance between exploring the environment, meaning picking an action randomly and exploitating our network to predict the right action.

We use a linear interpolation starting from exploration to exploitation. 
[Reference:](https://papers.nips.cc/paper/2016/file/8d8818c8e140c64c743113f563cf750f-Paper.pdf)

## Memory Buffer

Every experience is store in a memory to be used to train the model. Each entry is an experience defined by a state, an action, the reward, the next state and a boolean to know if it is the last step of the episode. 
An approach that works is to store a constant number of experiences and pick randomly a batch of experiences.

### Finding the experiences that matter
To increase the convergence of the neural network we use a probability distribution to select more often the experience that matter most. We are using the error defined by the difference of the target qvalue and the current qvalue to prioritize the experience.
[Reference:](https://arxiv.org/pdf/1511.05952.pdf)

### Finding it quick with a Sum tree
Using a sum tree allows to find the experience that have the biggest error. The sum is a binary tree where the parent node is equal to the sum of its direct children.
We use all the leaves to store the experiences. Each leaf store its priority asside its experience (state, action, rewards, next state, done). When adding an experience in the tree, the priority is updated according to the experience error. Then we update all then parents nodes value until we reach the root. By definition the root contains the sum of all priority of the entire tree.

## Conclusion


The D3QN implemented has a lot of hyper parameter, we can have a priority replay buffer or just a simple replay buffer, we can have a soft update target or a hard update with, each of them have different parameters, the learning rate, number of hidden layers, and their number of neurons etc.. The amount of parameters is huge, and their combination makes the algorithm hard to tune.

Developping a tool that try the algorithm with a various set of parameter could help to tune the algorithm.





