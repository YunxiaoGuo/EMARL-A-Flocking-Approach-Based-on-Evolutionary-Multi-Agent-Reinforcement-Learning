import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from common.agent import Agent
from tester import evaluate
from visualization import visualization
from common.replay_buffer import Buffer
from parameters.parameters import env_config
from algorithm import SEMARL,MADDPG,COMA,IQL,SQDDPG
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_evolve_rank(agents,env,args):
    s = env.reset()
    reward = []
    for time_step in range(args.max_epi_len):
        actions = []
        with torch.no_grad():
            for agent_id, agent in enumerate(agents):
                actions.append(agent.select_action(s[agent_id]))
        s_next, r, done, info = env.step(actions)
        reward.append(r)
    reward = np.array(reward)
    fitness = np.sum(reward,axis=0)
    agents_rank = sorted(range(fitness.shape[0]),key=lambda i:fitness[i],reverse=True)
    senior_agent_list = agents_rank[0:args.n_senior]
    junior_agent_list = agents_rank[-args.n_junior:]
    return senior_agent_list, junior_agent_list

def training(args):
    # Initialize environment
    env,args = env_config(args)
    # Initialize algorithm
    method = {
        'MADDPG': MADDPG,
        'SEMARL': SEMARL,
        'COMA': COMA,
        'SQDDPG': SQDDPG,
        'IQL': IQL
    }
    buffer = Buffer(args)
    algorithm = method[args.algorithm].algorithm
    agents = []
    returns = []
    for i in range(args.n):
        agent = Agent(i, args,algorithm)
        agents.append(agent)
    for episode in tqdm(range(args.max_epi)):
        s = env.reset()
        bufferINFO = []
        for time_step in range(args.max_epi_len):
            actions = []
            u = []
            with torch.no_grad():
                for agent_id, agent in enumerate(agents):
                    action = agent.select_action(s[agent_id])
                    actions.append(action)
                    u.append(action)
            s_next, r, done, info = env.step(actions)
            info = np.array(info)
            bufferINFO.append(info)
            buffer.store_episode(s, u, r, s_next)
            s = s_next
            if buffer.current_size >= args.batch_size:
                transitions = buffer.sample(args.batch_size)
                if args.algorithm == 'SEMARL':
                    senior_agent_list, junior_agent_list = get_evolve_rank(agents,env,args)
                    for agent in agents:
                        other_agents = agents.copy()
                        other_agents.remove(agent)
                        agent.evolve_learn(transitions, other_agents, senior_agent_list, junior_agent_list)
                else:
                    for agent in agents:
                        other_agents = agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)
        if episode > 0 and episode % args.evaluate_rate == 0:
            print('\nEvaluation Stage...')
            returns.append(evaluate(env,agents,args))
            visualization.real_time_reward(args,returns)
            if done == True:
                break




