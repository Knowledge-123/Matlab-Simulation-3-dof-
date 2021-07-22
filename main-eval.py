"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env import ArmEnv
from rl import DDPG
import numpy as np
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 去除cpu tensorflow版本报错

MAX_EPISODES = 4000
MAX_EP_STEPS = 200
#ON_TRAIN = True
ON_TRAIN = False


# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)


def eval():
    rl.restore()
    # env.render()
    # env.viewer.set_vsync(True)
    # env.viewer1.set_vsync(True)
    s = env.reset()
    i = 0
    while True:
        i += 1
        # env.render()
        env.plot()
        a = rl.choose_action(s)
        s, r, done = env.step(a, 0, ON_TRAIN)

        if i % 10 == 0:
            print(i)

        if i > 2000:
            break



if __name__ == '__main__':
    with open("data.csv", "w+", newline='') as csvfile:
        csvfile.close()
    eval()
