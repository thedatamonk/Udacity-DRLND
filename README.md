# Udacity Deep Reinforcement learning Nanodegree
## Problem statement
**Project Navigation**: In this project, we have to train an agent to navigate (and collect bananas!) in a large, square world. This environment is provided by [Unity Machine Learning agents] (https://github.com/Unity-Technologies/ml-agents).

![Unity Banana environment](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

**NOTE:**
1. This project was completed in the Udacity Workspace, but the project can also be completed on a local Machine. Instructions on how to download and setup Unity ML environments can be found in [Unity ML-Agents Github repo](https://github.com/Unity-Technologies/ml-agents). 
2. The environment provided by Udacity is similar to, but **not identical to** the Banana Collector environment on the [Unity ML-Agents Github page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). 

## Environment
The **state space** has `37` dimensions each of which is a continuous variable. It includes the agent's velocity, along with ray-based perception of objects around the agent's forward direction.
The **action space** contains the following `4` legal actions: 
- move forward `0`
- move backward `1`
- turn left `2`
- turn right `3`

A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.
The task is **episodic**, and in order to solve the environment, your agent must get an average score of `+13` over `100` consecutive episodes.

## Getting started
1. Download the environment from one of the links below. You need to only select the environment that matches your operating sytem:
 - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
 - Max OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
 - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
 
 (*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
 
 (*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), the please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

## Dependencies
1. Python 3.6
2. Pytorch
3. Unity ML-Agents

## Running the code
1. After installing all dependcies, clone this repository in your local system.
2. Make sure you have **Jupyter** installed. To install Jupyter:
```
python3 -m pip install --upgrade pip
python3 -m pip install jupyter
```
3. The main code exists in `Navigation.ipynb`. This file contains two training methods, `dqn` for **VanillaDQN** and `doubledqn` for **Double DQN**. Call these methods to train the model from scratch. Or reload pretrained models provided in the `./checkpoints` directory using `load_state_dict` method.

## Solution
1. Two methods are implememted to solve this problem: **VanillaDQN** and **DoubleDQN**. Both of them use **experience replay** and **target networks** to improve training.
2. Soft target network update is used to in VanillaDQN but it didn't turn out to be useful in DoubleDQN.

## Results
![DQN Score](https://github.com/thedatamonk/Unity-Banana-Navigation/blob/master/plots/dqn_score.png)

**Vanilla DQN** was able to solve the environment in approximately `1600` episodes. As is evident, after `450` episodes, the score did not improve much (it kept oscillating between `8` and `10`). After `1200` episodes, the score finally starts improving.

![DoubleDQN Score](https://github.com/thedatamonk/Unity-Banana-Navigation/blob/master/plots/doubledqn_score.png)

**Double DQN** was able to solve the environment in about `700` episodes, which is a great improvement over **VanillaDQN**. Also in this case, the score consistently improves for all episodes of training. 
