import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork1(nn.Module):

    def __init__(self, state_size, action_size, seed,adv_type = 'avg', fc1_units=128, fc2_units=64,fc3_units = 256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork1, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_adv = nn.Linear(fc2_units, fc3_units)
        self.fc_value = nn.Linear(fc2_units, fc3_units)
        self.adv = nn.Linear(fc3_units, action_size)
        self.value = nn.Linear(fc3_units, 1)
        self.adv_type = adv_type

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))# state_size -> fc1_units
        x = F.relu(self.fc2(x))# fc1_units -> fc2_units
        x_adv = F.relu(self.fc_adv(x))# fc2_units -> fc3_units
        x_adv = F.relu(self.adv(x_adv))# fc3_units -> action_size F.relu(self.adv(x_adv))
        x_value = F.relu(self.fc_value(x))# fc2_units -> fc3_units
        x_value = F.relu(self.adv(x_value))# fc3_units -> action_size F.relu(self.adv(x_value))
        if self.adv_type == 'avg':
          advAverage = torch.mean(x_adv, dim=1, keepdim=True)
          q =  x_value + x_adv - advAverage
        else:
          advMax,_ = torch.max(x_adv, dim=1, keepdim=True)
          q =  x_value + x_adv - advMax
        return q