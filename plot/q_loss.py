import pandas as pd
import random
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt



csv = pd.read_csv("/Users/bestacushlacj/Desktop/train/q_loss_1.csv")
csv2 = pd.read_csv("/Users/bestacushlacj/Desktop/train/q_loss_2.csv")
csv3 = pd.read_csv("/Users/bestacushlacj/Desktop/train/q_loss_3.csv")
# DDPG result
plt.plot(csv.Step, csv.Value, color='cyan', label='DDPG')
plt.plot(csv2.Step, csv2.Value, color='blue', label='FADDPG')
plt.plot(csv3.Step, csv3.Value, color='red', label='curiosity-FADDPG')
plt.xlabel('global step')
plt.ylabel('q_loss')
plt.legend()
# plt.title("torcs experiment")
plt.savefig('test.png')
plt.show()


# csv1 = pd.read_csv("/Users/bestacushlacj/Desktop/train/2d_q_loss_1.csv")
# csv2 = pd.read_csv("/Users/bestacushlacj/Desktop/train/2d_q_loss_2.csv")
# csv3 = pd.read_csv("/Users/bestacushlacj/Desktop/train/2d_q_loss_3.csv")
# csv4 = pd.read_csv("/Users/bestacushlacj/Desktop/train/2d_q_loss_4.csv")
#
# plt.plot(csv1.Step, csv1.Value, color='cyan', label='DDPG')
# plt.plot(csv2.Step, csv2.Value, color='blue', label='FADDPG')
# plt.plot(csv3.Step, csv3.Value, color='red', label='curiosity-FADDPG')
# # plt.plot(csv4.Step, csv4.Value, color='red', label='beta=0.001')
# plt.xlabel('global step')
# plt.ylabel('q_loss')
# plt.legend()
# plt.savefig('test.png')
# plt.show()

# 调参实验
# csv1 = pd.read_csv("/Users/bestacushlacj/Desktop/plt/q_value_loss_1.csv")
# csv2 = pd.read_csv("/Users/bestacushlacj/Desktop/plt/q_value_loss_2.csv")
# csv3 = pd.read_csv("/Users/bestacushlacj/Desktop/plt/q_value_loss_3.csv")
# plt.plot(csv1.Step, csv1.Value, color='cyan', label='rho=0.85')
# plt.plot(csv3.Step, csv3.Value, color='red', label='rho=0.9')
# plt.plot(csv2.Step, csv2.Value, color='blue', label='rho=0.95')
# plt.xlabel('global step')
# plt.ylabel('q_loss')
# plt.legend()
# plt.savefig('test.png')
# plt.show()