# -*- coding: utf-8 -*-
# author: Yunxiao Guo
# This is the flocking environment of paper:
# Cooperation and Competition: Flocking with Evolutionary Multi-Agent Reinforcement Learning. ICONIP (1) 2022: 271-283

#导入必要包
import gym
from gym import spaces
import numpy as np
import os
import sys
import argparse
from envs.action_space import MultiAgentActionSpace
from envs.observation_space import MultiAgentObservationSpace
#将当前路径添加到系统路径，以便后续算法调用
ROOT_DIR = os.path.abspath('./envs')
sys.path.append(ROOT_DIR)




#创建集群(flocking)环境
class flocking_env():
    def __init__(self,args):
        #智能体设置
        self.args = args
        self.n = self.args.n #智能体数量  n
        self.v_max = self.args.v_max #智能体线速度上限 v_max
        self.omega_max = self.args.omega_max #智能体角速度上限 ω_max
        self.delta_v = 3 #智能体单步更新线速度值  Δv
        self.delta_omega = np.pi/24 #智能体单步更新角速度值  Δω
        self.delta_t = 1 #时间步长 Δt
        self.k = self.args.num_obs_agent #周围可观测智能体数量 k
        self.high_action = 5
        self.n_action = 2
        #环境设置
        self.axis_lim = 1000 #XOY空间边界
        self.obs_n = 2 #障碍物数量
        self.obs_pos = [[200,100],[100,350]] #障碍物坐标
        self.obs_rad = [50,30]  #障碍物半径

        #奖励设置
        self.beta_sep = 1
        self.beta_ali = 1
        self.beta_coh = 1
        self.d_sep = 1
        self.d_coh = 1



        #划定状态、动作空间
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(low=np.array([0, 0, -4 * np.pi, -self.v_max, -self.omega_max]),
                        high=np.array([self.axis_lim, self.axis_lim, 4 * np.pi, self.v_max, self.omega_max])) for _ in range(self.k)])
        self.action_space = MultiAgentActionSpace([spaces.Tuple([spaces.Discrete(self.high_action) for i in range(self.n_action)])])
        #获取两点间距离
    def get_dis(self,r_1,r_2):
        return np.sqrt(((r_1[0]-r_2[0])/self.axis_lim)**2+((r_1[1]-r_2[1])/self.axis_lim)**2)

    #获取距离矩阵
    def get_dis_mat(self,x,y):
        D = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i+1,self.n):
                D[i,j] = self.get_dis([x[i],y[i]],[x[j],y[j]])
                D[j,i] = D[i,j]
        return D

    #角度归一化
    def theta_reg(self,theta):
        #功能：将角度θ限制在[-Π,+Π]上
        theta += np.pi
        while theta > 2*np.pi:
            theta -= 2*np.pi
        while theta < 0:
            theta += 2*np.pi
        return theta - np.pi

    #获取状态值
    def get_state(self):
        #把给定的x,y,v,omega,theta等序列，转化为对应状态空间的状态值
        self.D = -self.get_dis_mat(self.x,self.y)
        state = [[] for i in range(self.n)]
        for i in range(self.n):
            index = np.array(self.D[i,:]).argsort()[-self.k:][::-1]#返回距离最近的k个智能体的索引
            for j in index:
                state[i].append([self.x[j],self.y[j],self.theta[j],self.v[j],self.omega[j]])
        return state

    #获取相对状态
    def get_rstate(self):
        rstate = [[] for i in range(self.n)]
        for i in range(self.n):
            index = np.array(self.D[i,:]).argsort()[-(self.k+1):][::-1]#返回距离最近的k个智能体的索引
            for j in index:
                if i != j:
                    rstate[i].append(np.array([self.x[i]-self.x[j],self.y[i]-self.y[j],self.theta[i]-self.theta[j],self.v[i]-self.v[j],self.omega[i]-self.omega[j]]))
        rstate = [np.array(rstate[i]).reshape((1,self.observation_space[0].shape[0]*self.args.num_obs_agent)) for i in range(self.n)]
        return np.array(rstate)

    #重置环境
    def reset(self):
        self.x = np.random.uniform(low=self.axis_lim * 2/5, high=self.axis_lim * 3 / 5, size=self.n)
        self.y = np.random.uniform(low=self.axis_lim * 2/5, high=self.axis_lim * 3 / 5, size=self.n)
        self.v = np.random.uniform(low=-self.v_max, high=self.v_max, size=self.n)
        self.omega = np.random.uniform(low=-self.omega_max, high=self.omega_max, size=self.n)
        self.theta = np.random.uniform(low=-np.pi, high=np.pi, size=self.n)
        self.state = self.get_state()
        self.rstate = self.get_rstate()
        return self.rstate

    #Separation Reward
    def get_sep_rwd(self):
        sep_rwd = []
        for i in range(self.n):
            r_sep = 0
            for j in range(self.k):
                dis_r = self.get_dis([self.x[i],self.y[i]],[self.state[i][j][0],self.state[i][j][1]])
                r_sep -= 1/self.k*np.exp(-self.beta_sep*dis_r) if dis_r < self.d_sep else 0
            sep_rwd.append(r_sep)
        return sep_rwd

    #Alignment Reward
    def get_ali_rwd(self):
        ali_rwd = []
        for i in range(self.n):
            v_x = self.v[i]*np.cos(self.theta[i])
            v_y = self.v[i]*np.sin(self.theta[i])
            r_ali = 0
            for j in range(1,self.k):
                r_ali += 1/self.k*np.exp(self.beta_ali*self.get_dis([v_x,v_y],[self.state[i][j][3]*np.cos(self.state[i][j][2]),self.state[i][j][3]*np.cos(self.state[i][j][2])]))
            ali_rwd.append(r_ali)
        return ali_rwd

    #Cohesion Reward
    def get_coh_rwd(self):
        coh_rwd = []
        for i in range(self.n):
            x_bar = np.mean([self.state[i][j][0] for j in range(self.k)])
            y_bar = np.mean([self.state[i][j][1] for j in range(self.k)])
            mean_dis = self.get_dis([self.x[i],self.y[i]],[x_bar,y_bar])
            r_coh = np.exp(-self.beta_coh*mean_dis) if mean_dis >= self.d_coh else 0
            coh_rwd.append(r_coh)
        return coh_rwd

    #奖励函数
    def get_rwd(self):
        sep_rwd = np.array(self.get_sep_rwd())
        ali_rwd = np.array(self.get_ali_rwd())
        coh_rwd = np.array(self.get_coh_rwd())
        reward = np.sum([sep_rwd,ali_rwd,coh_rwd],axis=0)
        #print('##################',sep_rwd/reward,ali_rwd/reward,coh_rwd/reward,'###############')
        return reward.tolist()

    #执行下一步动作
    def step(self,action):
        info = []
        done = False
        for i in range(self.n):
            self.v[i] += (action[i][0]-2)*self.delta_v
            self.v[i] = np.clip(self.v[i],-self.v_max,self.v_max)
            self.omega[i] += (action[i][1]-2)*self.delta_omega
            self.omega[i] = np.clip(self.omega[i],-self.omega_max,self.omega_max)
            self.theta[i] += self.theta_reg(self.omega[i]*self.delta_t)
            self.x[i] += self.v[i]*np.cos(self.theta[i])*self.delta_t
            self.y[i] += self.v[i] * np.sin(self.theta[i]) * self.delta_t
            info.append([self.x[i],self.y[i],self.theta[i],self.v[i],self.omega[i]])
        self.state = self.get_state()# 用于奖励计算
        self.rstate = self.get_rstate()
        reward = self.get_rwd()
        return self.rstate, reward, done, info




