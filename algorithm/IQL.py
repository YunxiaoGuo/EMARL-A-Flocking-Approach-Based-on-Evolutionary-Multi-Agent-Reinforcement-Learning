import torch
import os
from networks.actor_critic import Actor, Critic
from torch.distributions import Categorical
class algorithm:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        # create the network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)
        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = os.path.join(self.args.save_dir,self.args.algorithm)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # 加载模型
        if os.path.exists(self.model_path + '/499_actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/499_actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/499_critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/499_actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/499_critic_params.pkl'))
    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            if isinstance(transitions[key],torch.Tensor):
                transitions[key] = transitions[key].clone()
            else:
                transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]
        o = transitions['o_%d' % self.agent_id]
        u = transitions['u_%d' % self.agent_id]
        o_next = transitions['o_next_%d' % self.agent_id]
        # Calculate the target Q value function
        with torch.no_grad():
            prob_next = self.actor_target_network(o_next)
            u_next = Categorical(prob_next.squeeze(0)).sample()
            q_next = self.critic_target_network(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()
        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()
        # the actor loss
        action_prob = self.actor_network(o)
        u = Categorical(action_prob.squeeze(0)).sample()
        actor_loss = - self.critic_network(o, u).mean()
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1
    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = self.args.save_dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        actor_model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(actor_model_path):
            os.makedirs(actor_model_path)
        torch.save(self.actor_network.state_dict(), actor_model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


