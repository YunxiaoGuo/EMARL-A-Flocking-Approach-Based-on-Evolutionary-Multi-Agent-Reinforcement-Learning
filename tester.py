import os
import torch
import numpy as np
from tqdm import tqdm
from envs import flocking_env
from visualization import visualization
from parameters.parameters import get_paras


def evaluate(env,agents,args):
    returns = []
    for episode in tqdm(range(args.evaluate_episodes)):
        s = env.reset()
        rewards = 0
        for time_step in range(args.evaluate_episode_len):
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(agents):
                    action = agent.select_action(s[agent_id])
                    actions.append(action)
            s_next, r, done, info = env.step(actions)
            rewards += r[0]
            s = s_next
        returns.append(rewards)
    return sum(returns) / args.evaluate_episodes

def sto_act_test():
    args = get_paras()
    env = flocking_env.flocking_env(args)

    save_path = args.data_path + '/' + 'flocking data/' + str(args.n) + 'agents' +'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    reward = []
    for epi in tqdm(range(args.max_epi)):
        s = env.reset()
        trace_store = []
        for time_step in (range(args.max_epi_len)):
            a_sto = [(np.random.randint(0,4), np.random.randint(0,4)) for _ in range(args.n)] #生成随机动作
            s_, r, done, info = env.step(a_sto)
            trace_store.append(info)
            reward.append(np.mean(r))

        trace_store = np.array(trace_store)
        np.save(save_path + 'trace_' + str(epi), trace_store)
        visualization.trace_drawer(trace_store,epi,args)
        visualization.var_drawer(trace_store, epi, args)
    visualization.draw_reward(reward,args,1)