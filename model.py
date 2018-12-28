import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    
    """DQN to estimate the Q-values of the input states"""
    
    def __init__(self, state_size, action_size, seed):
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # in this Environment state_size is 37 dimensional
        # and action_size is 4 dimensional
        self.state_size = state_size
        
        self.action_size = action_size
        
        # state_size ====> 64
        self.fc1 = nn.Linear(in_features = state_size, out_features = 150)
        
        # 64 ======> 64
        self.fc2 = nn.Linear(in_features = 150, out_features = 150)
        
        # 64 =======> action_size
        
        self.fc3 = nn.Linear(in_features = 150, out_features = 150)
        
        self.action_layer = nn.Linear(in_features = 150, out_features = action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values"""
        
        x = F.relu(F.dropout(self.fc1(state)))
        x = F.relu(F.dropout(self.fc2(x)))
        q_values = self.action_layer(x)
        
        return q_values