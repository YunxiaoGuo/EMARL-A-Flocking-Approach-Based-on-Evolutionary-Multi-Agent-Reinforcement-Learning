import os
import torch
import numpy as np
from torch.distributions import Categorical

class Agent:
    def __init__(self, agent_id, args,algorithm):
        self.args = args
        self.agent_id = agent_id
        self.policy = algorithm(args, agent_id)

    def select_action(self, o):
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
        pi = self.policy.actor_network(inputs).squeeze(0)
        # print('{} : {}'.format(self.name, pi))
        u = Categorical(pi.squeeze(0)).sample()
        u = u.cpu().numpy()
        return u.copy()

    def evolve_learn(self,transitions, other_agents, senior_agent_list, junior_agent_list):
        self.policy.train(transitions, other_agents, senior_agent_list, junior_agent_list)

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

