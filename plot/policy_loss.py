import pandas as pd
import random
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


csv = pd.read_csv("/Users/bestacushlacj/Desktop/train/policy_loss_1.csv")
# DDPG result
plt.plot(csv.Step, csv.Value, color='cyan', label='DDPG')
# plt.plot(csv3.Step, csv3.Value, color='blue', label='FADDPG')
# plt.plot(csv4.Step, csv4.Value, color='red', label='curiosity-FADDPG')
plt.xlabel('global step')
plt.ylabel('policy_loss')
plt.savefig('test.png')
plt.show()

