import argparse
import numpy as np
from envs import flocking_env
#获取Agent、环境参数
def get_paras():
    parser = argparse.ArgumentParser("Parameters for flocking environments")
    parser.add_argument("-max-episode-len", "--max_epi_len", type=int, default=100,
                        help="maximum episode length")
    parser.add_argument("-max-episodes", "--max_epi",type=int, default=1000,
                        help="number of time steps")
    parser.add_argument("-n-agents", "--n",type=int, default=15,
                        help="number of flocking agents")
    parser.add_argument("--n-senior", type=int, default=5,
                        help="number of senior agents")
    parser.add_argument("--n-junior", type=int, default=5,
                        help="number of junior agents")
    parser.add_argument("--p-evolve", type=float, default=0.8,
                        help="number of junior agents")
    parser.add_argument("-device", "--device", type=str, default='cuda',
                        help="number of flocking agents")
    parser.add_argument("-v-max", "--v_max", type=int, default=10,
                        help="maximum of the line velocity for agents")
    parser.add_argument("-omega-max", "--omega_max", type=int, default=np.pi / 4,
                        help="maximum of the angular velocity for agents")
    parser.add_argument("-data-path", "--data_path", type=str, default='./',
                        help="relative path for data storing")
    parser.add_argument("-result-path", "--result_path", type=str, default='./results/',
                        help="relative path for results storing")
    parser.add_argument("-buffer-size", "--buffer_size", type=int, default=1000,
                        help="The maximum volume of the experience replay buffer")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="parameter for updating the target network")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--algorithm", type=str, default='MADDPG',
                        help="The training algorithm")
    parser.add_argument("--save-dir", type=str, default="./model",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--lr-actor", type=float, default=1e-4,
                        help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3,
                        help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="epsilon greedy")
    parser.add_argument("--num-obs-agent", type=int, default=5,
                        help="number of observable agents")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--noise_rate", type=float, default=0.1,
                        help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--evaluate-rate", type=int, default=10,
                        help="how often to evaluate model")
    parser.add_argument("--evaluate-episode-len", type=int, default=100,
                        help="length of episodes for evaluating")
    parser.add_argument("--evaluate-episodes", type=int, default=10,
                        help="number of episodes for evaluating")
    args = parser.parse_args()
    return args

def env_config(args):
    env = flocking_env.flocking_env(args)
    args.high_action = env.high_action
    args.obs_shape = [env.observation_space[0].shape[0]*args.num_obs_agent for agent_id in range(args.n)]
    args.action_shape = [env.n_action for agent_id in range(args.n)]
    return env,args