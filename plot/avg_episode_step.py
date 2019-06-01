import pandas as pd
import random
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

csv2 = pd.read_csv("/Users/bestacushlacj/Desktop/train/avg_episode_steps_1.csv")
csv3 = pd.read_csv("/Users/bestacushlacj/Desktop/train/avg_episode_steps_2.csv")
# csv4 = pd.read_csv("/Users/bestacushlacj/Desktop/train/avg_episode_reward_4.csv")

plt.plot(csv2.Step, csv2.Value, color='cyan', label='DDPG')
plt.plot(csv3.Step, csv3.Value, color='blue', label='FADDPG')
# plt.plot(csv4.Step, csv4.Value, color='red', label='curiosity-FADDPG')
plt.xlabel('global step')
plt.ylabel('avg_episode_step')
plt.savefig('test.png')
plt.show()