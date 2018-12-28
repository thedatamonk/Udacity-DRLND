import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)     # replay buffer size
BATCH_SIZE = 32            # minibatch size
GAMMA = 0.99               # discount factor
TAU = 1e-3                 # for soft update of target parameters
LR = 1e-4                 # learning rate
UPDATE_EVERY = 4             # how many actions to choose between each stochastic update
UPDATE_TARGET_EVERY = 100   # how often to update the target network
# TRAINING_STARTS = 10000      # after how many time steps training should start
# to initialise the device (gpu or cpu) for training the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DoubleDQNAgent():
    """
        This class defines an object of Double DQN agent that interacts 
        with the environment and learns from it by updating its online and 
        target network.
    """
    
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        """"
            Defining the online and target Q-networks which are exact copies
            of each other. 
            ============
            Online network's parameters are represented by theta
            Target network's parameters are represented by theta_dot
            
            The logic behind using two networks of the same type is that 
            doing so breaks the correlation between the target (expected)
            action values of the current states and the current estimates of
            the action values of the current states.
            
            We select the best (greedy) action from the Online (theta) network
            while obtain the action value of the corresponding greedy action from 
            the target network (theta_dot).
        """
        
        self.qnetwork_online = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_online.parameters(), lr=LR)
        
        # initialise replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        
        """
        Initialise t_step_online and t_step_target
        ==========
        t_step_online: after how many time steps we should update the online
        network. Popular belief is to update the online network every gradient
        time step. But researches (cite them) have shown that agent learns better
        if trained at regular intervals
        
        t_step_target: after how many time steps we should update the target
        network. So let's say if this value is 10 then we will update its paramters 
        at t = 10, 20, 30 .... At other time steps, the parameters' value will
        remain same as the last updated values.
        """
        
        # time step to keep track of the target update
        self.t_step_target = 0
        self.losses = []
        
        # time step to keep track of the online network update
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # add experience in replay memory
        
        self.memory.add(state, action, reward, next_state, done)
    
        # update every UPDATE_ONLINE_EVERY time steps
#         self.t_step_online = (self.t_step_online + 1) % UPDATE_ONLINE_EVERY
        
#         if self.t_step_online == 0:
#             # if enough samples are there in the memory
#             if len(self.memory) > BATCH_SIZE:
#                 experiences = self.memory.sample()
#                 # now the learn function is called after every UPDATE_ONLINE_EVERY
#                 # time steps
#                 self.learn(experiences, GAMMA)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
        
            
    def act(self, state, eps=0.):
        """
        returns an action for the given state as per current policy.
        
        Params
        ======
              state: current_state
              eps: epsilon, for epsilon-greedy action selection
             
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_online.eval()
        
        # no need to compute gradients
        with torch.no_grad():
            action_values = self.qnetwork_online(state)
        self.qnetwork_online.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        """
        In order to calculate the loss (TD-error), we need to calculate 2 values:           ===========
        1. Target values 
        2. current Q-values for each state.
        
        Now, in order to break the correlation between current Q-values and the
        target Q-values, we require two set of weights. One is called local or
        online weights that are updated at every UPDATE_ONLINE_EVERY steps while
        the target weights are updated with the current online weights at
        UPDATE_TARGET_EVERY steps.
        
        """
        states, actions, rewards, next_states, dones = experiences
        
        
        _, greedy_actions = torch.max(self.qnetwork_online(next_states).detach(), dim=1)
        greedy_actions = torch.unsqueeze(greedy_actions, 1)
        # we select the action values corresponding to the greedy_actions we selected from
        # the local/online network
        
        q_greedy_targets = self.qnetwork_target(next_states).gather(1, greedy_actions)
        
        q_greedy_targets = (1 - dones) * q_greedy_targets
        q_targets = rewards + gamma * (q_greedy_targets)
        
        q_current_est = self.qnetwork_online(states).gather(1, actions)
        
        
        # backpropagation step
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_targets, q_current_est)
        
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss)
        # -----------------update the target network----------------- #
        self.t_step_target = (self.t_step_target + 1) % UPDATE_TARGET_EVERY
        
        if self.t_step_target == 0:
#             self.soft_update(self.qnetwork_online, self.qnetwork_target, TAU)
            self.update_target(self.qnetwork_online, self.qnetwork_target)
               


    def update_target(self, online_model, target_model):
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(online_param.data)
        
        
    def soft_update(self, online_model, target_model, tau):
        """
        Soft update model parameters.
        target_model = tau * online_model + (1 - tau) * target_model
        """
        
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    
    def sample(self):
        # this function will be only called when there are enough experience
        # samples in the Replay Buffer. So we do not have to check for this
        # in the sample function
        
        experiences = random.sample(self.memory, k = self.batch_size)
        
        """
        The following code decouples all components (s, a, r, s', d) from the
        experiences list and joins all components of the same type together
        """
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)
        
        
