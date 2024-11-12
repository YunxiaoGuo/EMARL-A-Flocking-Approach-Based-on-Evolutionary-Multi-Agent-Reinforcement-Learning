import torch
import os
from networks.actor_critic import Actor, Critic
from torch.distributions import Categorical
from torch.nn.functional import one_hot
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
        self.actor_model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.actor_model_path):
            os.mkdir(self.actor_model_path)
        # 加载模型
        if os.path.exists(self.actor_model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.actor_model_path + '/actor_params.pkl'))
        if os.path.exists(self.model_path + '/critic_params.pkl'):
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
    # get baseline
    def get_baseline(self,q_value,u_i,action_prob):
        u_1 = one_hot(u_i[:, 0], num_classes=self.args.high_action)
        u_2 = one_hot(u_i[:, 1], num_classes=self.args.high_action)
        prob = torch.sum(action_prob[:, 0] * u_1, dim=-1) *torch.sum(action_prob[:, 1] * u_2, dim=-1)
        prob = prob.unsqueeze(1)
        baseline = torch.sum(q_value * prob, dim=-1)
        return baseline.unsqueeze(1)

    # get ratio
    def get_log_prob(self,u_old,action_prob):
        log_prob = Categorical(action_prob.squeeze(0)).log_prob(u_old)
        joint_log_prob = torch.sum(log_prob,dim=-1)
        return joint_log_prob

    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            if isinstance(transitions[key],torch.Tensor):
                transitions[key] = transitions[key].clone()
            else:
                transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
        # Calculate the target Q value function
        with torch.no_grad():
            u_next = u
            prob_next = self.actor_target_network(o_next[self.agent_id])
            u_next[self.agent_id] = Categorical(prob_next.squeeze(0)).sample()
            q_next = self.critic_target_network(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()
        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()
        # the actor loss
        action_prob = self.actor_network(o[self.agent_id])
        u_old = u[self.agent_id].clone()
        u[self.agent_id] = Categorical(action_prob.squeeze(0)).sample()
        baseline = self.get_baseline(q_value,u[self.agent_id],action_prob)
        advantage = (q_value - baseline).detach()
        log_prob = self.get_log_prob(u_old,action_prob)
        actor_loss = - torch.mean(advantage * torch.exp(log_prob.unsqueeze(1)))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model()
        self.train_step += 1
    def save_model(self):
        torch.save(self.actor_network.state_dict(), self.actor_model_path + '/'  + 'actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  self.model_path + '/' + 'critic_params.pkl')


