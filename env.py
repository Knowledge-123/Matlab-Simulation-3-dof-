from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pyglet
import csv



class ArmEnv(object):
    viewer = None
    viewer1 = None
    dt = .1    # refresh rate
    action_bound = [-1, 1]
    goal3D = {'x': 100., 'y': 400., 'z': 40., 'r': 40.,
              'h': 80.}  # xy为体中心点,以此向上偏移20，确定上表面中心点
    goal = {'x': 200., 'y': 200., 'r': 40., 'h': 80}  # 此值为初始值，没有实际意义，后续会有新的赋值
    obstacle = {'x': 100, 'y': 50., 'l': 80}  # obstacal为包裹目标立方体
    state_dim = 22
    action_dim = 2  # 3个自由度
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(1, 1, 1, projection='3d')  # 为了画出连续的效果

    def __init__(self):
        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'][0] = 425.0  # /4        # 3 arms length 机械臂所有长度/4
        self.arm_info['l'][1] = 392.43  # /4
        #self.arm_info['l'][2] = 93/4
        self.arm_info['r'] = np.pi/6    # 3 angles information

        self.arm_info1 = np.zeros(
            1, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info1['l'] = 70  # 1 arm length
        self.arm_info1['r'] = np.pi / 6  # 1 angle information
        self.flag = 0

        self.a4l = 125.4  # /4
        self.a3l = 93  # /4
        self.a2l = 0
        self.a1l = 0
        self.a3z = 0
        self.a2z = 0
        self.a1z = 0
        self.weizi = []
        self.offset = 40

        self.on_goal = 0

        self.KP = 3
        self.KI = 0
        self.KD = 1
        self.u = 0
        self.e_all = 0
        self.e_last = 0

        self.on_touch = 0

    def step(self, action, step=0, ON_TRAIN=True):

        # move the goal
        if ON_TRAIN == False:
            if self.goal3D['x'] < 120*4 and self.flag == 0:
                self.goal3D['x'] += 1
            else:
                self.goal3D['x'] -= 1
                self.flag = 1
                if self.goal3D['x'] < -120*4:
                    self.flag = 0

        # 预先三维转化二维
        # 三维转二维
        lenth = np.sqrt(
            np.square(self.goal3D['x']) + np.square(self.goal3D['y']))
        self.goal['x'] = 200 + lenth
        self.goal['y'] = self.goal3D['z'] + 200
        # 障碍物位置更新
        self.obstacle['x'] = self.goal['x']
        self.obstacle['y'] = self.goal['y']

        # 角度转换
        if self.goal3D['x'] == 0:
            if self.goal3D['y'] >= 0:
                # np.arctan(109.0 / 4 / lenth)
                self.f = np.pi / 2 - np.arctan(109.0 / lenth)
            else:
                # np.arctan(109.0 / 4 / lenth)
                self.f = 3 * np.pi / 2 - np.arctan(109.0 / lenth)
        else:
            # np.arctan(109.0 / 4 / lenth)
            self.f = np.arctan(
                self.goal3D['y'] / self.goal3D['x']) - np.arctan(109.0 / lenth)

        # 计算机械臂平面位置
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a0xy = np.array([200., 200.])  # a0 start (x0, y0)
        a1xy = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + \
            a0xy  # a1 end and a2 start (x1, y1)
        a2xy = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]
                        ) * a2l + a1xy  # a2 end (x2, y2)
        a3xy = np.array([1, 0]) * self.a3l + a2xy  # a3 end (x3, y3)
        finger = np.array([0, -1]) * self.a4l + a3xy  # a4 end (x4, y4)

        self.weizi = [a1xy[0]-a0xy[0], a1xy[1] -
                      a0xy[1], a2xy[0]-a1xy[0], a2xy[1]-a1xy[1]]
        # print(self.weizi)

        # 坐标变换，从工作平面转至俯视平面
        self.arm_info1['l'] = np.sqrt((a2xy[0] - 200) ** 2)

        self.a2l = a2xy[0] - 200
        self.a1l = a1xy[0] - 200

        self.a4z = finger[1] - 200
        self.a3z = a3xy[1] - 200
        self.a2z = a2xy[1] - 200
        self.a1z = a1xy[1] - 200
        # PID 控制
        #
        #         # 更新夹角位置（获得输出）
        self.arm_info1['r'][0] += self.u*self.dt
        self.arm_info1['r'][0] %= np.pi * 2  # normalize
        # self.arm_info['r'][0] = np.clip(self.arm_info['r'][0], *[0.1*np.pi, 0.9*np.pi]) #限定旋转角度

        # 计算偏差 deata
        if self.goal3D['x'] >= 0:
            deata = self.f % (2 * np.pi) - self.arm_info1['r'][0]
        else:
            deata = self.f + np.pi - self.arm_info1['r'][0]

        # pid 控制
        self.u = self.KP*deata + self.KI * \
            self.e_all + self.KD*(deata-self.e_last)
        self.e_all += deata  # 误差的累加和
        self.e_last = deata  # 前一个误差值

        # DDPG 控制

        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        # normalize features 归一化
        dist1 = [(self.goal['x'] - a1xy[0]) / 400,
                 (self.goal['y'] + self.offset - a1xy[1]) / 400]
        dist2 = [(self.goal['x'] - a2xy[0]) / 400,
                 (self.goal['y'] + self.offset - a2xy[1]) / 400]
        dist3 = [(self.goal['x'] - a3xy[0]) / 400,
                 (self.goal['y'] + self.offset - a3xy[1]) / 400]  # 目标物和机械臂2末端的距离坐标
        distm = [(self.goal['x'] - finger[0]) / 400,
                 (self.goal['y'] + self.offset - finger[1]) / 400]  # 目标物和末端执行器的距离坐标，且目标点与GOAL中心点向上偏置offset=40
        # 目标物和末端执行器的距离长度，没有归一化
        dist0 = np.sqrt(
            pow(self.goal['x'] - finger[0], 2) + pow(self.goal['y'] + self.offset - finger[1], 2))

        dist4 = [(self.obstacle['x'] - a1xy[0]) / 400,
                 (self.obstacle['y'] - a1xy[1]) / 400]
        dist5 = [(self.obstacle['x'] - a2xy[0]) / 400,
                 (self.obstacle['y'] - a2xy[1]) / 400]
        dist6 = [(self.obstacle['x'] - a3xy[0]) / 400,
                 (self.obstacle['y'] - a3xy[1]) / 400]

        # 构建奖惩函数
        #r = - np.sqrt(dist0[0] ** 2 + dist0[1] ** 2) + 0 * np.sqrt(dist1[0] ** 2 + dist1[1] ** 2)\
        #   + 10 / (np.sqrt(dist0[0] ** 2 + dist0[1] ** 2))#偏差倒数，奖励和位置的倒数成反比 放大系数可适当增大

        r = - np.sqrt(distm[0] ** 2 + distm[1] ** 2) + \
            0.5 / (np.sqrt(distm[0] ** 2 + distm[1] ** 2))

        # add obstacle
        # 在三个机械臂上采样，各取4个代表点
        f1 = (a2xy - a3xy) / 5 + a3xy
        f2 = 2 * (a2xy - a3xy) / 5 + a3xy
        f3 = 3 * (a2xy - a3xy) / 5 + a3xy
        f4 = 4 * (a2xy - a3xy) / 5 + a3xy

        a1 = (a1xy - a0xy) / 5 + a0xy
        a2 = 2 * (a1xy - a0xy) / 5 + a0xy
        a3 = 3 * (a1xy - a0xy) / 5 + a0xy
        a4 = 4 * (a1xy - a0xy) / 5 + a0xy

        e1 = (a2xy - a1xy) / 5 + a1xy
        e2 = 2 * (a2xy - a1xy) / 5 + a1xy
        e3 = 3 * (a2xy - a1xy) / 5 + a1xy
        e4 = 4 * (a2xy - a1xy) / 5 + a1xy

        # 构建罚函数
        R = 15
        # R = 0 ## 暂不考虑避障，同时，由于末端执行器的长度小于目标宽度的一半，使得其在接近与远离之间跳动。
        if self.obstacle['x'] - self.obstacle['l'] / 2 < f1[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < f1[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < f2[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < f2[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < f3[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < f3[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < f4[0] < self.obstacle['x'] + self.obstacle['l']/2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < f4[1] < self.obstacle['y'] + self.obstacle['l']/2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < a1[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < a1[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < a2[0] < self.obstacle['x'] + self.obstacle['l']/2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < a2[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < a3[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < a3[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < a4[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < a4[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < e1[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < e1[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < e2[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < e2[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < e3[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < e3[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < e4[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < e4[1] < self.obstacle['y'] + self.obstacle['l']/2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < a2xy[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < a2xy[1] < self.obstacle['y'] + self.obstacle['l']/2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < a1xy[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < a1xy[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1
        if self.obstacle['x'] - self.obstacle['l'] / 2 < a3xy[0] < self.obstacle['x'] + self.obstacle['l'] / 2 and \
                self.obstacle['y'] - self.obstacle['l'] / 2 < a3xy[1] < self.obstacle['y'] + self.obstacle['l'] / 2:
            r += -R
            self.on_touch = 1

        # done and reward
        if self.goal['x'] - self.goal['r'] < finger[0] < self.goal['x'] + self.goal['r']:
            if self.goal['y'] - self.goal['h']/2 < finger[1] < self.goal['y'] + 40 + self.goal['h']/2:
                r += 5
                self.on_goal += 1
                if step > 500:
                    if dist0 <= self.goal['r']/2:
                        self.on_goal += 1
                        r += 1000
                    if dist0 <= self.goal['r']/4:
                        self.on_goal += 1
                        r += 2000
                    if dist0 <= self.goal['r']/6:
                        self.on_goal += 1
                        r += 5000
                    if dist0 <= self.goal['r']/10:
                        r += 10000
                        self.on_goal += 1
                if self.on_goal >= 50:
                    done = True
        else:
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy / 200, a2xy / 200, finger / 200, dist1 + dist2 + dist3 + distm,
                           [1. if self.on_goal else 0.], dist4 + dist5 + dist6, [0. if self.on_touch else 1.]))
        return s, r, done

    def reset(self):

        self.ax.scatter3D(0, 0, 0, cmap='Blues')
        # plt.ion()
        # plt.show() #画出3维图
        # if self.goal3D['x'] < 150*4:
        #    self.goal3D['x'] += 1
        # else:
        #    self.goal3D['x'] = -150*4

        self.goal3D['x'] = np.random.rand()*600
        #self.goal3D['x'] = (np.random.rand()-0.5)*180.
        #self.goal3D['y'] = (np.random.rand()-0.5)*180.
        #self.goal3D['z'] = (np.random.rand()-0.5)*180.

        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        self.arm_info1['r'] = 2 * np.pi * np.random.rand(1)
        self.on_goal = 0
        self.on_touch = 0

        self.u = 0
        self.e_all = 0
        self.e_last = 0

        # 计算goal的矢量夹角
        if self.goal3D['x'] == 0:
            if self.goal3D['y'] >= 0:
                self.f = np.pi/2
            else:
                self.f = 3*np.pi/2
        else:
            self.f = np.arctan(self.goal3D['y']/self.goal3D['x'])

        # 三维转二维
        lenth = np.sqrt(
            np.square(self.goal3D['x']) + np.square(self.goal3D['y']))
        self.goal['x'] = lenth + 200
        self.goal['y'] = self.goal3D['z'] + 200
        # 障碍物位置更新
        self.obstacle['x'] = self.goal['x']
        self.obstacle['y'] = self.goal['y']

        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a0xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + \
            a0xy  # a1 end and a2 start (x1, y1)
        a2xy = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]
                        ) * a2l + a1xy  # a2 end (x2, y2)
        a3xy = np.array([1, 0]) * self.a3l + a2xy  # a3 end (x3, y3)
        finger = np.array([0, -1]) * self.a4l + a3xy  # a4 end (x4, y4)

        self.arm_info1['l'] = np.sqrt(np.square(a2xy[0]-200))

        # normalize features
        dist1 = [(self.goal['x'] - a1xy[0]) / 400,
                 (self.goal['y'] + 20 - a1xy[1]) / 400]
        dist2 = [(self.goal['x'] - a2xy[0]) / 400,
                 (self.goal['y'] + 20 - a2xy[1]) / 400]
        dist3 = [(self.goal['x'] - a3xy[0]) / 400,
                 (self.goal['y'] + 20 - a3xy[1]) / 400]
        distm = [(self.goal['x'] - finger[0]) / 400,
                 (self.goal['y'] + 20 - finger[1]) / 400]

        dist4 = [(self.obstacle['x'] - a1xy[0]) / 400,
                 (self.obstacle['y'] - a1xy[1]) / 400]
        dist5 = [(self.obstacle['x'] - a2xy[0]) / 400,
                 (self.obstacle['y'] - a2xy[1]) / 400]
        dist6 = [(self.obstacle['x'] - a3xy[0]) / 400,
                 (self.obstacle['y'] - a3xy[1]) / 400]

        # state
        s = np.concatenate((a1xy / 200, a2xy / 200, finger / 200, dist1 + dist2 + dist3 + distm, [
                           1. if self.on_goal else 0.], dist4 + dist5 + dist6, [0. if self.on_touch else 1.]))
        return s

    def plot(self):

        biasl = 109.0  # / 4
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']

        a1rf = self.arm_info1['r'][0]  # radian, angle
        a0xy = np.array([0, 0])
        a1xy = np.array([np.cos(a1rf), np.sin(a1rf)]) * \
            np.cos(a1r) * a1l + a0xy  # a1 end and a2 start (x1, y1)
        # a1 end and a2 start (x1, y1)
        a2xy = np.array([np.cos(a1rf), np.sin(a1rf)]) * \
            np.cos(a1r+a2r) * a2l + a1xy
        # a2 end and a3 start (x2, y2)
        a3xy = np.array([np.cos(a1rf + np.pi / 2),
                        np.sin(a1rf + np.pi / 2)]) * biasl + a2xy
        a4xy = np.array([np.cos(a1rf), np.sin(a1rf)]) * self.a3l + a3xy
        a5xy = a4xy  # a3 end (x3, y3)

        a0z = [0]
        a1z = [self.a1z]
        a2z = [self.a2z]
        a3z = [self.a2z]
        a4z = [self.a3z]
        a5z = [self.a3z-self.a4l]

        a0 = np.concatenate((a0xy, a0z))
        a1 = np.concatenate((a1xy, a1z))
        a2 = np.concatenate((a2xy, a2z))
        a3 = np.concatenate((a3xy, a3z))
        a4 = np.concatenate((a4xy, a4z))
        a5 = np.concatenate((a5xy, a5z))

        A = np.array([a0, a1, a2, a3, a4])
        # print(A)

        x1 = A[:, 0][:2]
        y1 = A[:, 1][:2]
        z1 = A[:, 2][:2]
        x2 = A[:, 0][1:3]
        y2 = A[:, 1][1:3]
        z2 = A[:, 2][1:3]
        x3 = A[:, 0][2:4]
        y3 = A[:, 1][2:4]
        z3 = A[:, 2][2:4]
        x4 = A[:, 0][3:5]
        y4 = A[:, 1][3:5]
        z4 = A[:, 2][3:5]

        # 末端执行器
        xm = np.array([a4xy[0], a5xy[0]])
        ym = np.array([a4xy[1], a5xy[1]])
        zm = np.array([a4z[0], a5z[0]])

        # 目标物的坐标
        xa = self.goal3D['x']
        ya = self.goal3D['y']
        za = self.goal3D['z']
        l = self.goal3D['h'] / 2

        xd = np.array([xa - l, xa + l, xa + l, xa - l, xa - l, xa + l, xa + l, xa - l,
                       xa - l, xa - l, xa - l, xa - l, xa + l, xa + l, xa + l, xa + l])
        yd = np.array([ya - l, ya - l, ya + l, ya + l, ya + l, ya + l, ya - l, ya - l,
                       ya - l, ya + l, ya + l, ya - l, ya - l, ya - l, ya + l, ya + l])
        zd = np.array([za - l, za - l, za - l, za - l, za + l, za + l, za + l, za + l,
                       za - l, za - l, za + l, za + l, za + l, za - l, za - l, za + l])

        xa = self.goal3D['x']
        ya = self.goal3D['y']
        za = self.goal3D['z']
        l = self.goal3D['h'] / 2

        xdo = np.array([xa - l, xa + l, xa + l, xa - l, xa - l, xa + l, xa + l, xa - l,
                        xa - l, xa - l, xa - l, xa - l, xa + l, xa + l, xa + l, xa + l])
        ydo = np.array([ya - l, ya - l, ya + l, ya + l, ya + l, ya + l, ya - l, ya - l,
                        ya - l, ya + l, ya + l, ya - l, ya - l, ya - l, ya + l, ya + l])
        zdo = np.array([za - l, za - l, za - l, za - l, za + l, za + l, za + l, za + l,
                        za - l, za - l, za + l, za + l, za + l, za - l, za - l, za + l])

        self.ax.set_xlim(-1000, 1000)  # 编辑坐标轴范围
        self.ax.set_ylim(-1000, 1000)
        self.ax.set_zlim(0, 1000)
        lines1 = self.ax.plot3D(x1, y1, z1, 'red', lw=5)
        lines2 = self.ax.plot3D(x2, y2, z2, 'red', lw=5)
        lines3 = self.ax.plot3D(x3, y3, z3, 'green', lw=5)
        lines4 = self.ax.plot3D(x4, y4, z4, 'red', lw=5)
        lines5 = self.ax.plot3D(xm, ym, zm, 'green', lw=5)
        lines11 = self.ax.plot3D(xd, yd, zd, 'blue', lw=2)
        lines22 = self.ax.plot3D(xdo, ydo, zdo, 'blue', lw=1)

        plt.pause(0.01)
        self.ax.lines.remove(lines1[0])
        self.ax.lines.remove(lines2[0])
        self.ax.lines.remove(lines3[0])
        self.ax.lines.remove(lines4[0])
        self.ax.lines.remove(lines5[0])
        self.ax.lines.remove(lines11[0])
        self.ax.lines.remove(lines22[0])

        #lenth1 = [np.sqrt(a1xy[0] ** 2 + a1xy[1] ** 2)]
        #a1 = np.concatenate((lenth1, a1z))

        #lenth2 = [np.sqrt(a2xy[0] ** 2 + a2xy[1] ** 2)]
        #a2 = np.concatenate((lenth2, a2z))

        #B = np.concatenate(([a1rf], a1, a2, [xa, ya, za]))
        B = np.concatenate(([a1rf], self.weizi, [xa, ya, za]))

        with open("data.csv", "a+", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(B)

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()
        if self.viewer1 is None:
            self.viewer1 = Round(self.arm_info, self.arm_info1, self.goal3D)
        self.viewer1.render()

    def sample_action(self):
        return (np.random.rand(2)-0.5)  # three radians


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal):

        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=500, height=400,
                                     resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.a4l = 125.4  # /4
        self.a3l = 93  # /4
        self.goal_info = goal
        self.center_coord = np.array([200, 200])
        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [goal['x'] - goal['r'], goal['y'] - goal['h']/2,  # location
                     goal['x'] - goal['r'], goal['y'] + goal['h']/2,
                     goal['x'] + goal['r'], goal['y'] + goal['h']/2,
                     goal['x'] + goal['r'], goal['y'] - goal['h']/2]),
            ('c3B', (86, 109, 249) * 4))  # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))
        self.arm3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))
        self.arm4 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (0, 255, 0) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['r'], self.goal_info['y'] -
            self.goal_info['h']/2,
            self.goal_info['x'] + self.goal_info['r'], self.goal_info['y'] -
            self.goal_info['h']/2,
            self.goal_info['x'] + self.goal_info['r'], self.goal_info['y'] +
            self.goal_info['h']/2,
            self.goal_info['x'] - self.goal_info['r'], self.goal_info['y'] + self.goal_info['h']/2)

        # update arm
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + \
            a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * \
            a2l + a1xy_  # a2 end and a3 start (x2, y2)
        a3xy_ = np.array([1, 0]) * self.a3l + a2xy_  # a3 end (x3, y3)
        a4xy_ = np.array([0, -1]) * self.a4l + a3xy_  # a4 end (x4, y4)

        a1tr, a2tr = np.pi / 2 - \
            self.arm_info['r'][0], np.pi / 2 - \
            self.arm_info['r'][0] - self.arm_info['r'][1]
        a3tr = np.pi/2
        a4tr = np.pi

        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        xy21_ = a2xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy22_ = a2xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy31 = a3xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy32 = a3xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc

        xy31_ = a3xy_ + np.array([np.cos(a4tr), np.sin(a4tr)]) * self.bar_thc
        xy32_ = a3xy_ + np.array([-np.cos(a4tr), np.sin(a4tr)]) * self.bar_thc
        xy41 = a4xy_ + np.array([-np.cos(a4tr), np.sin(a4tr)]) * self.bar_thc
        xy42 = a4xy_ + np.array([np.cos(a4tr), np.sin(a4tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
        self.arm3.vertices = np.concatenate((xy21_, xy22_, xy31, xy32))
        self.arm4.vertices = np.concatenate((xy31_, xy32_, xy41, xy42))


class Round(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, arm_info1, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Round, self).__init__(width=400, height=400,
                                    resizable=False, caption='flat', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.arm_info1 = arm_info1
        self.a3l = 93  # /4
        self.goal_info = goal
        #self.obstacle_info = obstacle
        self.center_coord = np.array([200, 200])
        self.batch = pyglet.graphics.Batch()  # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [goal['x'] - goal['h'] / 2, goal['y'] - goal['h'] / 2,  # location
                     goal['x'] - goal['h'] / 2, goal['y'] + goal['h'] / 2,
                     goal['x'] + goal['h'] / 2, goal['y'] + goal['h'] / 2,
                     goal['x'] + goal['h'] / 2, goal['y'] - goal['h'] / 2]),
            ('c3B', (86, 109, 249) * 4))  # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,  # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))  # color
        self.arm3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (0, 255, 0) * 4,))
        self.arm4 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))
        self.arm5 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (0, 255, 0) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            200 + self.goal_info['x'] - self.goal_info['h'] /
            2, 200 + self.goal_info['y'] - self.goal_info['h'] / 2,
            200 + self.goal_info['x'] + self.goal_info['h'] /
            2, 200 + self.goal_info['y'] - self.goal_info['h'] / 2,
            200 + self.goal_info['x'] + self.goal_info['h'] /
            2, 200 + self.goal_info['y'] + self.goal_info['h'] / 2,
            200 + self.goal_info['x'] - self.goal_info['h'] / 2, 200 + self.goal_info['y'] + self.goal_info['h'] / 2)

        # update arm

        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1l = np.fabs(self.arm_info1['l'][0])  # radius, arm length
        # print(a1l)
        biasl = 109.0  # /4
        a1rf = self.arm_info1['r'][0]  # radian, angle

        a0xy = self.center_coord  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1rf), np.sin(a1rf)]) * \
            a1l + a0xy  # a1 end and a2 start (x1, y1)
        # a2 end and a3 start (x2, y2)
        a3xy_ = np.array(
            [np.cos(a1rf + np.pi / 2), np.sin(a1rf + np.pi / 2)]) * biasl + a1xy_
        a4xy_ = np.array([np.cos(a1rf), np.sin(a1rf)]) * self.a3l + a3xy_
        a5xy_ = a4xy_  # a3 end (x3, y3)

        a1tr = np.pi / 2 - a1rf
        xy01 = a0xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a0xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        a3tr = -a1rf
        xy21_ = a1xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy22_ = a1xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy31 = a3xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy32 = a3xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc

        a4tr = np.pi/2 - a1rf
        xy31_ = a3xy_ + np.array([-np.cos(a4tr), np.sin(a4tr)]) * self.bar_thc
        xy32_ = a3xy_ + np.array([np.cos(a4tr), -np.sin(a4tr)]) * self.bar_thc
        xy41 = a4xy_ + np.array([np.cos(a4tr), -np.sin(a4tr)]) * self.bar_thc
        xy42 = a4xy_ + np.array([-np.cos(a4tr), np.sin(a4tr)]) * self.bar_thc

        a5tr = - np.pi / 2 + a1rf
        xy41_ = a4xy_ + np.array([-np.cos(a5tr), np.sin(a5tr)]) * self.bar_thc
        xy42_ = a4xy_ + np.array([np.cos(a5tr), -np.sin(a5tr)]) * self.bar_thc
        xy51 = a5xy_ + np.array([np.cos(a5tr), -np.sin(a5tr)]) * self.bar_thc
        xy52 = a5xy_ + np.array([-np.cos(a5tr), np.sin(a5tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm3.vertices = np.concatenate((xy21_, xy22_, xy31, xy32))
        self.arm4.vertices = np.concatenate((xy31_, xy32_, xy41, xy42))
        self.arm5.vertices = np.concatenate((xy41_, xy42_, xy51, xy52))

        # convert the mouse coordinate to goal's coordinate 鼠标位置
    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x - 200
        self.goal_info['y'] = y - 200


if __name__ == '__main__':
    env = ArmEnv()
    for _ in range(200):
        env.reset()
        for _ in range(200):
            # env.render()
            # env.plot()
            env.step(env.sample_action())
