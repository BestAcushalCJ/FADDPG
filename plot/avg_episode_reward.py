import pandas as pd
import random
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def formatunm(x, pos):
    return '$%.1f$10^{4}$'%(x/10000)

# 训练
csv1 = pd.read_csv("/Users/bestacushlacj/Desktop/train/avg_episode_reward_1.csv")
csv2 = pd.read_csv("/Users/bestacushlacj/Desktop/train/avg_episode_reward_2.csv")
csv3 = pd.read_csv("/Users/bestacushlacj/Desktop/train/avg_episode_reward_3.csv")
csv4 = pd.read_csv("/Users/bestacushlacj/Desktop/train/avg_episode_reward_4.csv")
# DDPG result
plt.plot(csv1.Step, csv1.Value,color='black',label='abnormal situation')
plt.plot(csv2.Step, csv2.Value, color='cyan', label='normal situation')
# plt.plot(csv3.Step, csv3.Value, color='blue', label='FADDPG')
# plt.plot(csv4.Step, csv4.Value, color='red', label='curiosity-FADDPG')
plt.xlabel('global step')
plt.ylabel('avg_episode_reward')
plt.legend()
plt.savefig('test.png')
plt.show()

# csv1 = pd.read_csv("/Users/bestacushlacj/Desktop/train/2d_avg_episode_reward_1.csv")
# csv2 = pd.read_csv("/Users/bestacushlacj/Desktop/train/2d_avg_episode_reward_2.csv")
# csv3 = pd.read_csv("/Users/bestacushlacj/Desktop/train/2d_avg_episode_reward_3.csv")
# plt.plot(csv1.Step, csv1.Value, color='cyan', label='DDPG')
# plt.plot(csv2.Step, csv2.Value, color='blue', label='FADDPG')
# plt.plot(csv3.Step, csv3.Value, color='red', label='curiosity-FADDPG')
# plt.xlabel('global step')
# plt.ylabel('avg_episode_reward')
# plt.legend()
# plt.savefig('test.png')
# plt.show()

