import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import os
matplotlib.use('Agg')


def trace_drawer(info,episode,args):
    trace_path = args.result_path + 'traces/' + str(args.n) + 'agents' + '/'
    if not os.path.exists(trace_path):
        os.makedirs(trace_path)
    pd.options.display.notebook_repr_html = False  # 表格显示
    plt.rcParams['figure.dpi'] = 350  # 图形分辨率
    sns.set_theme(style='darkgrid')  # 图形主题
    for i in range(info.shape[1]):
        df=pd.DataFrame(dict(X=info[:,i,0],Y=info[:,i,1]))
        sns.scatterplot(x=df['X'], y=df['Y'])
        plt.plot(info[:,i,0], info[:,i,1])#,label='Follower%d'%i)

    #plt.legend()
    plt.savefig(trace_path + 'flocking_' + str(episode)+ '.pdf', format='pdf')
    plt.clf()


def var_drawer(info,episode,args):
    var_path = args.result_path + 'variable/' + str(args.n) + 'agents' + '/'
    if not os.path.exists(var_path):
        os.makedirs(var_path)
    pd.options.display.notebook_repr_html = False  # 表格显示
    plt.rcParams['figure.dpi'] = 350  # 图形分辨率
    plt.figure()
    ax = plt.gca()
    for i in range(info.shape[1]):
        plt.plot(range(info.shape[0]),info[:,i,0],)#,label='Follower%d'%i)
    ax.set_xlabel('Timestep (s)', fontsize=14)
    ax.set_ylabel('x (m)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(var_path + 'flocking_x_' + str(episode)+ '.pdf', format='pdf')
    plt.clf()

    plt.figure()
    ax = plt.gca()
    for i in range(info.shape[1]):
        plt.plot(range(info.shape[0]),info[:,i,1],)#,label='Follower%d'%i)
    ax.set_xlabel('Timestep (s)', fontsize=14)
    ax.set_ylabel('y (m)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(var_path + 'flocking_y_' + str(episode)+ '.pdf', format='pdf')
    plt.clf()

    plt.figure()
    ax = plt.gca()
    for i in range(info.shape[1]):
        plt.plot(range(info.shape[0]), info[:, i, 2], )  # ,label='Follower%d'%i)
    ax.set_xlabel('Timestep (s)', fontsize=14)
    ax.set_ylabel('Θ (°)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(var_path + 'flocking_theta_' + str(episode) + '.pdf', format='pdf')
    plt.clf()

    plt.figure()
    ax = plt.gca()
    for i in range(info.shape[1]):
        plt.plot(range(info.shape[0]), info[:, i, 3], )  # ,label='Follower%d'%i)
    ax.set_xlabel('Timestep (s)', fontsize=14)
    ax.set_ylabel('v (m/s)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(var_path + 'flocking_v_' + str(episode) + '.pdf', format='pdf')
    plt.clf()

    plt.figure()
    ax = plt.gca()
    for i in range(info.shape[1]):
        plt.plot(range(info.shape[0]), info[:, i, 4], )  # ,label='Follower%d'%i)
    ax.set_xlabel('Timestep (s)', fontsize=14)
    ax.set_ylabel('$\omega$ (rad/s)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(var_path + 'flocking_w_' + str(episode) + '.pdf', format='pdf')
    plt.clf()


def draw_reward(reward,args,num):
    reward_path = args.result_path + 'reward/' + str(args.n) + 'agents' + '/'
    if not os.path.exists(reward_path):
        os.makedirs(reward_path)
    pd.options.display.notebook_repr_html = False  # 表格显示
    plt.rcParams['figure.dpi'] = 350  # 图形分辨率
    plt.figure()
    ax = plt.gca()
    plt.plot(range(len(reward)), reward)  # ,label='Follower%d'%i)
    ax.set_xlabel('Timestep (s)', fontsize=14)
    ax.set_ylabel('reward', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.savefig(reward_path + 'training_curve' + str(num) + '.pdf', format='pdf')
    plt.clf()


def real_time_reward(args, returns):
    plt.rcParams['figure.dpi'] = 350  # 图形分辨率
    plt.figure()
    ax = plt.gca()
    plt.plot(range(len(returns)), returns)
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Average Reward', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(args.result_path + '/real_time_avearage_reward.pdf', format='pdf')
    plt.clf()
    plt.close('all')