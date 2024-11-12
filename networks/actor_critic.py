import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.args = args
        self.agent_id = agent_id
        self.action_dim = args.action_shape
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.softmax = nn.Softmax(dim=-1)
        self.action_out = nn.Linear(64, args.action_shape[agent_id]*self.max_action)

    def forward(self, x):
        x = x.reshape((x.shape[0],self.args.obs_shape[self.agent_id]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.action_out(x)
        x = x.reshape((x.shape[0], self.args.action_shape[self.agent_id], self.max_action))
        prob = self.softmax(x)
        return prob



class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.max_action = self.args.high_action
        if self.args.algorithm == 'IQL':
            self.fc1 = nn.Linear(self.args.obs_shape[0] + self.args.action_shape[0], 64)
        else:
            self.fc1 = nn.Linear(sum(self.args.obs_shape) + sum(self.args.action_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        if self.args.algorithm != 'IQL':
            state = torch.cat(state, dim=1)
            action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value




